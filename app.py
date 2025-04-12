from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
from dotenv import load_dotenv
from bollinger_bands_strat2 import StochasticOscillatorTrader, SecureConfigManager

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# List of available data files
data_files = {
    'btc_2023': 'BTC_Year_2023_data.csv',
    'btc_2024': 'BTC_2024_DATA.csv',
    'btc_2022': 'BTC_Year_2022_data.csv',
    'eth_2023': 'Ethereum_Data_2023.csv',
    'eth_2024': 'Ethereum_Data_2024_2025.csv',
    'sol_2023': 'SOLANA_DATA_YEAR_2023.csv',
    'sol_2024': 'SOLANA_DATA_YEAR_2024_2025.csv',
}

# Add a health check route for Render
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "ok", "message": "Stochastic Oscillator Trading Bot API is running"})

@app.route('/api/available-data', methods=['GET'])
def get_available_data():
    """Return a list of available data files for backtesting"""
    return jsonify({
        'data_options': [
            {'id': key, 'name': f"{key.split('_')[0].upper()} {key.split('_')[1]}"} 
            for key in data_files.keys()
        ]
    })

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run a backtest with the provided parameters"""
    try:
        # Get parameters from the request
        params = request.json
        
        # Load data
        data_key = params.get('data_source', 'btc_2023')
        if data_key not in data_files:
            return jsonify({'error': 'Invalid data source'}), 400
            
        # Read the data
        df = pd.read_csv(data_files[data_key])
        
        # Set datetime index if timestamp column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Create configuration
        config = {
            'initial_balance': float(params.get('initial_balance', 10000)),
            'risk_percentage': float(params.get('risk_percentage', 1)),
            'trading_symbol': params.get('trading_symbol', 'BTCUSDT'),
            'k_period': int(params.get('k_period', 15)),
            'd_period': int(params.get('d_period', 5))
        }
        
        # Initialize trader
        trader = StochasticOscillatorTrader(config)
        
        # Calculate technical indicators
        df = trader.calculate_stochastic_oscillator(df)
        
        # Run backtest
        results = trader.backtest_strategy(df)
        
        # Convert trades to serializable format
        trades_serializable = []
        for trade in results['trades']:
            trade_serializable = {
                'entry_time': trade['entry_time'].isoformat() if hasattr(trade['entry_time'], 'isoformat') else str(trade['entry_time']),
                'exit_time': trade['exit_time'].isoformat() if hasattr(trade['exit_time'], 'isoformat') else str(trade['exit_time']),
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'profit': float(trade['profit']),
                'trade_result': trade['trade_result']
            }
            trades_serializable.append(trade_serializable)
        
        # Add chart data
        chart_data = []
        for index, row in df.iterrows():
            if isinstance(index, pd.Timestamp):
                timestamp = index.isoformat()
            else:
                timestamp = str(index)
            
            data_point = {
                'timestamp': timestamp,
                'close': float(row['Close']),
                'k': float(row['%K']) if not pd.isna(row['%K']) else None,
                'd': float(row['%D']) if not pd.isna(row['%D']) else None,
                'signal': int(row['Signal']) if not pd.isna(row['Signal']) else 0
            }
            chart_data.append(data_point)
        
        # Prepare the response
        response = {
            'summary': {
                'total_return': float(results['total_return']),
                'win_rate': float(results['win_rate']),
                'max_drawdown': float(results['max_drawdown']),
                'total_trades': int(results['total_trades']),
                'winning_trades': int(results['winning_trades']),
                'final_balance': float(results['final_balance'])
            },
            'trades': trades_serializable,
            'chart_data': chart_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy-info', methods=['GET'])
def get_strategy_info():
    """Return information about the trading strategy"""
    return jsonify({
        'name': 'Stochastic Oscillator Strategy',
        'description': 'This strategy uses the Stochastic Oscillator to identify overbought and oversold conditions. It buys when the %K line crosses above the %D line while in oversold territory (below 20) and sells when the %K line crosses below the %D line while in overbought territory (above 80).',
        'parameters': [
            {
                'name': 'K Period',
                'description': 'The lookback period for calculating the %K line',
                'default': 15
            },
            {
                'name': 'D Period',
                'description': 'The lookback period for calculating the %D line (smoothing of %K)',
                'default': 5
            }
        ],
        'entry_rules': 'Enter long when %K crosses above %D while %K is below 20 (oversold condition)',
        'exit_rules': 'Exit when %K crosses below %D while %K is above 80 (overbought condition)',
    })

if __name__ == '__main__':
    # Use environment variables for host and port if available
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) 