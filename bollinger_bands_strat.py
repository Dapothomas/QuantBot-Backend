import os
import json
import time
import hmac
import math
import hashlib
import requests
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# Load environment variables from .env file
load_dotenv()

class SecureConfigManager:
    @staticmethod
    def load_config() -> Dict:
        try:
            config = {
                'binance_api_key': os.getenv('BINANCE_API_KEY'),
                'binance_secret_key': os.getenv('BINANCE_SECRET_KEY'),
                'trading_symbol': os.getenv('TRADING_SYMBOL', 'BTCUSDT'),
                'initial_balance': float(os.getenv('INITIAL_BALANCE', 10000)),
                'risk_percentage': float(os.getenv('RISK_PERCENTAGE', 1)),
                'testnet': os.getenv('BINANCE_TESTNET', 'False').lower() == 'true',
                'k_period': int(os.getenv('STOCHASTIC_K_PERIOD', 15)),
                'd_period': int(os.getenv('STOCHASTIC_D_PERIOD', 5))
            }
            
            if not config['binance_api_key'] or not config['binance_secret_key']:
                raise ValueError("API keys must be configured in .env file")
            
            return config
        except Exception as e:
            print(f"Configuration loading error: {e}")
            raise

class BinanceClient:
    def __init__(self, api_key: str, secret_key: str, testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"

    def place_order(self, symbol: str, side: str, quantity: float, order_type="MARKET"):
        endpoint = f"{self.base_url}/v3/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timestamp": int(time.time() * 1000),
        }
        params["signature"] = self._generate_signature(params)

        headers = {"X-MBX-APIKEY": self.api_key}
        response = requests.post(endpoint, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()

    def _generate_signature(self, params: Dict) -> str:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'), 
            query_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

    def get_historical_klines(self, symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical klines/candlestick data from Binance

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '15m')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        endpoint = f"{self.base_url}/api/v3/klines"

        # Convert dates to timestamps (milliseconds)
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

        all_klines = []
        current_ts = start_ts
        total_days = (end_ts - start_ts) / (1000 * 60 * 60 * 24)
        days_processed = 0
        last_update_time = time.time()

        print(f"\nStarting data collection for {symbol} from {start_date} to {end_date}")
        print(f"Collecting data in {interval} intervals\n")

        while current_ts < end_ts:
            try:
                current_time = time.time()
                days_processed = (current_ts - start_ts) / (1000 * 60 * 60 * 24)
                progress = (days_processed / total_days) * 100 if total_days > 0 else 100

                # Update progress every 2 seconds
                if current_time - last_update_time >= 2:
                    print(f"\rProgress: {progress:.2f}% | Processing date: {datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')}", end="")
                    last_update_time = current_time

                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_ts,
                    'endTime': min(current_ts + (1000 * 60 * 60 * 24), end_ts),  # Get 1 day at a time
                    'limit': 1000
                }

                response = requests.get(endpoint, params=params)
                response.raise_for_status()

                klines = response.json()
                if not klines:
                    break

                all_klines.extend(klines)
                current_ts = klines[-1][0] + 1  # Next timestamp after last received

                time.sleep(0.1)  # Respect rate limits

            except requests.exceptions.HTTPError as e:
                print(f"\nHTTP error occurred: {e.response.status_code} - {e.response.text}")
                raise
            except requests.exceptions.RequestException as e:
                print(f"\nError fetching data: {e}")
                if hasattr(response, 'status_code') and response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                raise
            
        print("\nData collection completed!")

        if not all_klines:
            raise ValueError("No data retrieved for the specified period")

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'quote_Volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
        ])

        # Convert types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'quote_Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"\nProcessed {len(df)} data points")
        return df
class StochasticOscillatorTrader:
    def __init__(self, config: Dict, binance_client: Optional[BinanceClient] = None, initial_balance: float = 10000):
        self.config = config
        self.client = binance_client
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.risk_percentage = config['risk_percentage']
        self.commission_rate = 0.0005
        self.Open_position = None
        self.trade_log = []
        self.commission = 0.001
        self.slippage = 0.0005
        self.k_period = 15  #GOOD self.k_period = 14,19,15,16,17 BEST 15
        self.d_period = 5   #GOOD self.d_period = 5,5            BEST 5
        
        

    def calculate_stochastic_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
        Low_min = data['Low'].rolling(window=self.k_period).min()
        High_max = data['High'].rolling(window=self.k_period).max()
        
        data['%K'] = (data['Close'] - Low_min) / (High_max - Low_min) * 100
        data['%D'] = data['%K'].rolling(window=self.d_period).mean()
        
        data['Signal'] = 0
        data.loc[(data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20), 'Signal'] = 1
        data.loc[(data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80), 'Signal'] = -1
        
        return data
    
    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price based on trade direction"""
        multiplier = 1 + (self.slippage if side == 'buy' else -self.slippage)
        return price * multiplier

    def calculate_position_size(self, entry_price: float) -> float:
        max_risk_amount = self.balance
        position_size = max_risk_amount / entry_price
        return min(position_size, self.balance / entry_price)

    # def apply_trading_costs(self, trade_value: float) -> float:
    #     commission = trade_value * self.commission_rate
    #     slippage_cost = trade_value * self.slippage
    #     return trade_value - commission - slippage_cost

    def get_live_price(self, symbol: str) -> float:
        endpoint = f"{self.client.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(endpoint, params=params).json()
        return float(response["price"])

    def convert_timestamp(self, timestamp):
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

    def backtest_strategy(self, data: pd.DataFrame) -> Dict:
        initial_balance = self.initial_balance
        current_balance = initial_balance
        trades = []
        Open_position = None
        max_drawdown = 0
        peak_balance = initial_balance
        total_trades = 0
        winning_trades = 0
        
        # Progress tracking variables
        total_rows = len(data)
        last_update_time = time.time()
        
        for i in range(1, len(data)):
            # Calculate and show progress every 10 seconds
            current_time = time.time()
            current_percentage = (i / total_rows) * 100
            
            if current_time - last_update_time >= 10 or current_percentage == 100:
                print(f"Progress: {current_percentage:.2f}% complete | "
                      f"Current Balance: ${current_balance:.2f} | "
                      f"Total Trades: {total_trades} | "
                      f"Winning Trades: {winning_trades}")
                last_update_time = current_time
                
            current_row = data.iloc[i]
            
            if Open_position is None and current_row['Signal'] == 1:
                entry_price = self.apply_slippage(current_row['Close'], 'buy')
                position_size = self.calculate_position_size(entry_price)
                cost = entry_price * position_size * (1 + self.commission)
                
                if cost <= current_balance:
                    Open_position = {
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'entry_time': current_row.name,
                        'stop_loss': entry_price * 0.99,
                        'take_profit': entry_price * 1.1
                    }
                    total_trades += 1
            
            elif Open_position is not None:
                exit_condition = False
                exit_price = current_row['Close']
                
                if current_row['Signal'] == -1:
                    exit_price = self.apply_slippage(current_row['Close'], 'sell')
                    exit_condition = True
                    trade_result = 'win' if exit_price > Open_position['entry_price'] else 'loss'
                else:
                    if current_row['Close'] <= Open_position['stop_loss']:
                        exit_price = Open_position['stop_loss']
                        exit_condition = True
                        trade_result = 'loss'
                    elif current_row['Close'] >= Open_position['take_profit']:
                        exit_price = Open_position['take_profit']
                        exit_condition = True
                        trade_result = 'win'
    
                if exit_condition:
                    balance_before = current_balance
                    position_value = exit_price * Open_position['position_size']
                    commission_cost = position_value * self.commission
                    trade_profit = (position_value - commission_cost) - (Open_position['entry_price'] * Open_position['position_size'])
                    current_balance += trade_profit
    
                    if trade_result == 'win':
                        winning_trades += 1
    
                    peak_balance = max(peak_balance, current_balance)
                    current_drawdown = (peak_balance - current_balance) / peak_balance
                    max_drawdown = max(max_drawdown, current_drawdown)
    
                    trades.append({
                        'entry_time': Open_position['entry_time'],
                        'exit_time': current_row.name,
                        'entry_price': Open_position['entry_price'],
                        'exit_price': exit_price,
                        'position_size': Open_position['position_size'],
                        'profit': trade_profit,
                        'balance_before': balance_before,
                        'balance_after': current_balance,
                        'trade_result': trade_result
                    })
                    Open_position = None
    
        # Calculate final statistics
        total_return = ((current_balance - initial_balance) / initial_balance) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        average_profit = sum(trade['profit'] for trade in trades) / len(trades) if trades else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'final_balance': current_balance,
            'trades': trades,
            'average_profit': average_profit
        }

    def paper_trade(self):
        print("\nStarting Paper Trading...")
        while True:
            try:
                live_price = self.get_live_price(self.config['trading_symbol'])
                print(f"\n[Market Update] Current Price: {live_price}")

                latest_data = self.fetch_latest_candle_data()
                latest_data = self.calculate_stochastic_oscillator(latest_data)

                current_k = latest_data["%K"].iloc[-1]
                current_d = latest_data["%D"].iloc[-1]
                prev_k = latest_data["%K"].iloc[-2]
                prev_d = latest_data["%D"].iloc[-2]

                print(f"%K: {current_k:.2f}, %D: {current_d:.2f}")
                print(f"Previous %K: {prev_k:.2f}, Previous %D: {prev_d:.2f}")
                print(f"Available Balance: ${self.balance:.2f}")

                print("\n[Trading Conditions]")
                # Buy condition checks
                if self.Open_position is None:
                    print("Checking BUY conditions:")
                    print(f"1. Is %K crossing above %D? {(current_k > current_d) and (prev_k <= prev_d)}")
                    print(f"2. Is %K beLow 20? {current_k < 20}")

                    if not ((current_k > current_d) and (prev_k <= prev_d)):
                        print("❌ No buy signal: %K is not crossing above %D")
                    elif not (current_k < 20):
                        print("❌ No buy signal: %K is not beLow 20 (not oversold)")

                    if (current_k > current_d) and (prev_k <= prev_d) and (current_k < 20):
                        position_size = self.calculate_position_size(live_price)
                        trade_cost = live_price * position_size
                        fee = trade_cost * self.commission_rate

                        if trade_cost + fee > self.balance:
                            print("❌ No buy execution: Insufficient balance")
                        else:
                            self.balance -= (trade_cost + fee)
                            self.Open_position = {
                                "entry_price": live_price,
                                "size": position_size,
                                "buy_time": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            print(f"✅ BUY EXECUTED! {position_size} {self.config['trading_symbol']} at {live_price}")
                            print(f"New Balance: ${self.balance:.2f}")

                # Sell condition checks
                else:
                    print("Checking SELL conditions:")
                    print(f"1. Is %K crossing beLow %D? {(current_k < current_d) and (prev_k >= prev_d)}")
                    print(f"2. Is %K above 80? {current_k > 80}")

                    if not ((current_k < current_d) and (prev_k >= prev_d)):
                        print("❌ No sell signal: %K is not crossing beLow %D")
                    elif not (current_k > 80):
                        print("❌ No sell signal: %K is not above 80 (not overbought)")

                    if (current_k < current_d) and (prev_k >= prev_d) and (current_k > 80):
                        sell_value = live_price * self.Open_position["size"]
                        fee = sell_value * self.commission_rate
                        profit = (sell_value - (self.Open_position["entry_price"] * self.Open_position["size"])) - fee

                        self.balance += (sell_value - fee)
                        self.trade_log.append({
                            "entry_price": self.Open_position["entry_price"],
                            "exit_price": live_price,
                            "profit": profit,
                            "size": self.Open_position["size"],
                            "buy_time": self.Open_position["buy_time"],
                            "sell_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })

                        print(f"✅ SELL EXECUTED! {self.Open_position['size']} {self.config['trading_symbol']} at {live_price}")
                        print(f"Profit from Trade: ${profit:.2f}")
                        print(f"New Balance: ${self.balance:.2f}")
                        self.Open_position = None

                
                # time.sleep(30)
                now = datetime.now(timezone.utc)
                # Calculate seconds until the next quarter hour
                minutes = now.minute
                seconds = now.second
                wait_seconds = ((15 - (minutes % 15)) * 60) - seconds
                wait_seconds = max(wait_seconds, 1)  # Ensure at least 1 second wait
                print(f"\nWaiting {wait_seconds} seconds for the next update...")
                time.sleep(wait_seconds) #for 15 minutes

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

    def fetch_latest_candle_data(self):
        end_time = int(time.time() * 1000)
        # start_time = end_time - (self.k_period * 2 * 60 * 1000)
        start_time = end_time - (self.k_period * 2 * 15 * 60 * 1000)  # Adjust for 15-minute intervals
        historical_data = self.client.get_historical_klines(
            # self.config['trading_symbol'], "1m", start_time, end_time
            self.config['trading_symbol'], "15m", start_time, end_time  # Change to 15m
        )

        df = pd.DataFrame(historical_data, columns=[
            "Time", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
            "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore"
        ])

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")
        df.set_index("Time", inplace=True)
        return df

def main():
    try:
        parser = argparse.ArgumentParser(description="Run Stochastic Oscillator Strategy")
        parser.add_argument("--mode", choices=["backtest", "live"], required=True)
        args = parser.parse_args()

        config = SecureConfigManager.load_config()
        binance_client = BinanceClient(config["binance_api_key"], config["binance_secret_key"], config["testnet"])
        trader = StochasticOscillatorTrader(config, binance_client)
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_SECRET_KEY')
        data_fetcher = BinanceClient(api_key, api_secret)
        symbol = 'BTCUSDT'
        interval = '15m'
        start_date = '2023-01-01'
        end_date = '2023-12-31'

        if args.mode == "backtest":
            print(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
            df = data_fetcher.get_historical_klines(symbol, interval, start_date, end_date)
            
            # Calculate indicators before running backtest
            print("\nCalculating indicators...")
            backtester = StochasticOscillatorTrader(config, initial_balance=10000)
            df_with_signals = backtester.calculate_stochastic_oscillator(df)
            
            # Run backtest
            print("\nRunning backtest...")
            results = backtester.backtest_strategy(df_with_signals)

            # Print results
            print("\n=== Backtest Results ===")
            print(f"Initial Balance: ${backtester.initial_balance:,.2f}")
            print(f"Final Balance: ${results['final_balance']:,.2f}")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Winning Trades: {results['winning_trades']}")
            print(f"Win Rate: {results['win_rate']:.2f}%")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Average Profit per Trade: ${results['average_profit']:,.2f}")

            # Save trades to CSV
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv('binance_backtest_trades.csv', index=False)
            print("\nTrade history saved to 'binance_backtest_trades.csv'")

        elif args.mode == "live":
            print("\n=== Starting Live Paper Trading ===")
            trader.paper_trade()

    except FileNotFoundError:
        print("Error: CSV file not found.")
    except pd.errors.EmptyDataError:
        print("Error: Empty CSV file.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()