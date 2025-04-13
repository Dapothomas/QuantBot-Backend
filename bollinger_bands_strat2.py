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
                'd_period': int(os.getenv('STOCHASTIC_D_PERIOD', 5)),
                'position_sizing': os.getenv('POSITION_SIZING', 'percentage'),
                'fixed_position_size': float(os.getenv('FIXED_POSITION_SIZE', 0.1)),
                'position_size_percentage': int(os.getenv('POSITION_SIZE_PERCENTAGE', 90))
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
        endpoint = f"{self.base_url}/api/v3/order"
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
        return response.json()

    def _generate_signature(self, params: Dict) -> str:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.secret_key.encode('utf-8'), 
            query_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

    def get_historical_klines(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[List]:
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time
        }
        response = requests.get(endpoint, params=params)
        return response.json()

class StochasticOscillatorTrader:
    def __init__(self, config: Dict, binance_client: Optional[BinanceClient] = None):
        self.config = config
        self.client = binance_client
        self.initial_balance = config['initial_balance']
        self.balance = self.initial_balance
        self.risk_percentage = config['risk_percentage']
        self.commission_rate = 0.0005
        self.open_position = None
        self.trade_log = []
        self.slippage = 0.0003
        self.k_period = 15  #GOOD self.k_period = 14,19,15,16,17 BEST 15
        self.d_period = 5   #GOOD self.d_period = 5,5            BEST 5
        self.position_sizing = config.get('position_sizing', 'percentage')  # Default: percentage
        self.fixed_position_size = config.get('fixed_position_size', 0.1)  # Default: 0.1 BTC/ETH
        self.position_size_percentage = config.get('position_size_percentage', 90)  # Default: 90% of balance
        
        

    def calculate_stochastic_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
        print("\nDebug - Stochastic Calculation:")
        print(f"Data shape: {data.shape}")
        print(f"Last few prices:")
        print(data[["High", "Low", "Close"]].tail())

        low_min = data['Low'].rolling(window=self.k_period).min()
        high_max = data['High'].rolling(window=self.k_period).max()

        data['%K'] = (data['Close'] - low_min) / (high_max - low_min) * 100
        data['%D'] = data['%K'].rolling(window=self.d_period).mean()

        print("\nCalculated values for last few candles:")
        print(data[['Close', '%K', '%D']].tail())

        data['Signal'] = 0
        data.loc[(data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20), 'Signal'] = 1
        data.loc[(data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80), 'Signal'] = -1

        return data

    def calculate_position_size(self, entry_price: float) -> float:
        if self.position_sizing == 'fixed':
            # Calculate the maximum affordable position size based on current balance
            max_affordable = self.balance / entry_price
            
            # Return the smaller of fixed size or maximum affordable
            return min(self.fixed_position_size, max_affordable)
        else:
            # Use percentage of balance (default)
            available_balance = self.balance * (self.position_size_percentage / 100)
            position_size = available_balance / entry_price
            return min(position_size, self.balance / entry_price)

    def apply_trading_costs(self, trade_value: float) -> float:
        commission = trade_value * self.commission_rate
        slippage_cost = trade_value * self.slippage
        return trade_value - commission - slippage_cost

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
        open_position = None
        max_drawdown = 0
        peak_balance = initial_balance
        total_trades = 0
        winning_trades = 0

        for i in range(1, len(data)):
            current_row = data.iloc[i]
            
            if open_position is None and current_row['Signal'] == 1:
                entry_price = current_row['Close']
                position_size = self.calculate_position_size(entry_price)
                
                # Calculate the cost of entry including fees
                entry_cost = entry_price * position_size
                entry_fee = entry_cost * self.commission_rate
                total_entry_cost = entry_cost + entry_fee
                
                # Ensure we have enough balance for the trade
                if total_entry_cost <= current_balance:
                    # Deduct the cost from current balance
                    current_balance -= total_entry_cost
                    
                    # Store the timestamp directly from the index
                    entry_time = current_row.name
                    
                    open_position = {
                        'entry_price': entry_price,
                        'position_size': position_size,
                        'entry_time': entry_time,  # Store the pandas Timestamp object directly
                        'entry_k': current_row['%K'],
                        'entry_d': current_row['%D'],
                        'stop_loss': entry_price * 0.99,
                        'take_profit': entry_price * 1.1,
                        'entry_cost': total_entry_cost
                    }
                    total_trades += 1
            
            elif open_position is not None:
                exit_condition = False
                
                if current_row['Signal'] == -1:
                    exit_price = current_row['Close']
                    exit_condition = True
                    trade_result = 'win' if exit_price > open_position['entry_price'] else 'loss'
                else:
                    if current_row['Close'] <= open_position['stop_loss']:
                        exit_price = open_position['stop_loss']
                        exit_condition = True
                        trade_result = 'loss'
                    elif current_row['Close'] >= open_position['take_profit']:
                        exit_price = open_position['take_profit']
                        exit_condition = True
                        trade_result = 'win'

                if exit_condition:
                    # Calculate exit value including fees
                    exit_value = exit_price * open_position['position_size']
                    exit_fee = exit_value * self.commission_rate
                    net_exit_value = exit_value - exit_fee
                    
                    balance_before = current_balance
                    # Add the net exit value to current balance
                    current_balance += net_exit_value
                    
                    # Calculate profit/loss for the trade
                    trade_profit = net_exit_value - open_position['entry_cost']

                    if trade_result == 'win':
                        winning_trades += 1

                    peak_balance = max(peak_balance, current_balance)
                    current_drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
                    max_drawdown = max(max_drawdown, current_drawdown)

                    # Store the exit timestamp directly
                    exit_time = current_row.name

                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': exit_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'entry_k': open_position['entry_k'],
                        'entry_d': open_position['entry_d'],
                        'exit_k': current_row['%K'],
                        'exit_d': current_row['%D'],
                        'position_size': open_position['position_size'],
                        'profit': trade_profit,
                        'balance_before': balance_before,
                        'balance_after': current_balance,
                        'stop_loss': open_position['stop_loss'],
                        'take_profit': open_position['take_profit'],
                        'trade_result': trade_result
                    })
                    open_position = None

        total_return = ((current_balance - initial_balance) / initial_balance) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'final_balance': current_balance,
            'trades': trades
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
                if self.open_position is None:
                    print("Checking BUY conditions:")
                    print(f"1. Is %K crossing above %D? {(current_k > current_d) and (prev_k <= prev_d)}")
                    print(f"2. Is %K below 20? {current_k < 20}")

                    if not ((current_k > current_d) and (prev_k <= prev_d)):
                        print("❌ No buy signal: %K is not crossing above %D")
                    elif not (current_k < 20):
                        print("❌ No buy signal: %K is not below 20 (not oversold)")

                    if (current_k > current_d) and (prev_k <= prev_d) and (current_k < 20):
                        position_size = self.calculate_position_size(live_price)
                        trade_cost = live_price * position_size
                        fee = trade_cost * self.commission_rate
                        total_cost = trade_cost + fee

                        if total_cost > self.balance:
                            print("❌ No buy execution: Insufficient balance")
                            print(f"  Required: ${total_cost:.2f}, Available: ${self.balance:.2f}")
                            print(f"  Maximum affordable position: {(self.balance / live_price):.6f} {self.config['trading_symbol']}")
                        else:
                            self.balance -= total_cost
                            self.open_position = {
                                "entry_price": live_price,
                                "size": position_size,
                                "entry_cost": total_cost,
                                "buy_time": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            print(f"✅ BUY EXECUTED! {position_size} {self.config['trading_symbol']} at {live_price}")
                            print(f"  Cost: ${total_cost:.2f} (incl. ${fee:.2f} fee)")
                            print(f"  New Balance: ${self.balance:.2f}")

                # Sell condition checks
                else:
                    print("Checking SELL conditions:")
                    print(f"1. Is %K crossing below %D? {(current_k < current_d) and (prev_k >= prev_d)}")
                    print(f"2. Is %K above 80? {current_k > 80}")

                    if not ((current_k < current_d) and (prev_k >= prev_d)):
                        print("❌ No sell signal: %K is not crossing below %D")
                    elif not (current_k > 80):
                        print("❌ No sell signal: %K is not above 80 (not overbought)")

                    if (current_k < current_d) and (prev_k >= prev_d) and (current_k > 80):
                        sell_value = live_price * self.open_position["size"]
                        fee = sell_value * self.commission_rate
                        net_value = sell_value - fee
                        profit = net_value - self.open_position["entry_cost"]
                        profit_percent = (profit / self.open_position["entry_cost"]) * 100

                        self.balance += net_value
                        self.trade_log.append({
                            "entry_price": self.open_position["entry_price"],
                            "exit_price": live_price,
                            "profit": profit,
                            "profit_percent": profit_percent,
                            "size": self.open_position["size"],
                            "buy_time": self.open_position["buy_time"],
                            "sell_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })

                        print(f"✅ SELL EXECUTED! {self.open_position['size']} {self.config['trading_symbol']} at {live_price}")
                        print(f"  Net Value: ${net_value:.2f} (after ${fee:.2f} fee)")
                        print(f"  Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)")
                        print(f"  New Balance: ${self.balance:.2f}")
                        self.open_position = None

                
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
        start_time = end_time - (self.k_period * 2 * 15 * 60 * 1000)

        print("\nDebug - Data Fetch:")
        print(f"Start time: {datetime.fromtimestamp(start_time/1000)}")
        print(f"End time: {datetime.fromtimestamp(end_time/1000)}")

        historical_data = self.client.get_historical_klines(
            self.config['trading_symbol'], "15m", start_time, end_time
        )

        print(f"Number of candles received: {len(historical_data)}")

        df = pd.DataFrame(historical_data, columns=[
            "Time", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
            "TakerBuyBaseVolume", "TakerBuyQuoteVolume", "Ignore"
        ])

        # Convert columns to float
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        # Clean anomalous data
        for index in range(len(df)):
            row = df.iloc[index]
            avg_price = (float(row['Open']) + float(row['Close'])) / 2

            # If Low is too far from average (e.g., more than 50% deviation)
            if float(row['Low']) < avg_price * 0.5:
                df.at[index, 'Low'] = min(float(row['Open']), float(row['Close']))

            # If High is too far from average
            if float(row['High']) > avg_price * 1.5:
                df.at[index, 'High'] = max(float(row['Open']), float(row['Close']))

        # Convert time and set index
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")
        df.set_index("Time", inplace=True)

        print("\nCleaned last few candles:")
        print(df[["Open", "High", "Low", "Close"]].tail())

        return df

def main():
    try:
        parser = argparse.ArgumentParser(description="Run Stochastic Oscillator Strategy")
        parser.add_argument("--mode", choices=["backtest", "live"], required=True)
        args = parser.parse_args()

        config = SecureConfigManager.load_config()
        config['testnet'] = False
        binance_client = BinanceClient(config["binance_api_key"], config["binance_secret_key"], config["testnet"])
        trader = StochasticOscillatorTrader(config, binance_client)

        if args.mode == "backtest":
    # Load and prepare the data
            df = pd.read_csv("SOLANA_DATA_YEAR_2024_2025.csv")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
        
            # Convert price columns to numeric
            price_columns = ["Open", "High", "Low", "Close", "Volume"]
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=price_columns, inplace=True)
        
            # Run the backtest
            analysis_data = trader.calculate_stochastic_oscillator(df)
            backtest_results = trader.backtest_strategy(analysis_data)

            if backtest_results['trades']:
                trades_df = pd.DataFrame(backtest_results['trades'])

                # Format numeric columns
                numeric_columns = ['entry_price', 'exit_price', 'entry_k', 'entry_d', 
                                 'exit_k', 'exit_d', 'profit', 'balance_before', 
                                 'balance_after', 'stop_loss', 'take_profit']
                for col in numeric_columns:
                    trades_df[col] = trades_df[col].round(2)

                # Ensure timestamps are in the correct format
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
                trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

                # Save to CSV
                trades_df.to_csv('backtest_results.csv', index=False)
                print("\nSaved detailed backtest trades to backtest_results.csv")

            print("\n=== Backtest Results ===")
            print(f"Total Return: {backtest_results['total_return']:.2f}%")
            print(f"Total Trades: {backtest_results['total_trades']}")
            print(f"Winning Trades: {backtest_results['winning_trades']}")
            print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
            print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
            print(f"Final Balance: ${backtest_results['final_balance']:,.2f}")

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

# .env template additions:
# STOCHASTIC_K_PERIOD=14
# STOCHASTIC_D_PERIOD=3