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

# Historical context for each dataset
historical_context = {
    'btc_2022': "2022 was Bitcoin's bear market year, with major crashes following the collapse of Terra Luna in May and FTX in November. BTC fell from ~$47,000 to under $17,000, a 65% decline. This dataset captures trading during extreme market stress.",
    'btc_2023': "2023 was a recovery year for Bitcoin following the 2022 bear market. BTC started at ~$16,500 and ended around $42,000, rising 150%. The market showed resilience despite regulatory crackdowns and banking issues in March.",
    'btc_2024': "2024 began with Bitcoin's ETF approval in January, creating significant momentum. Bitcoin reached new all-time highs above $73,000 in March following its fourth halving, then entered a consolidation phase.",
    'eth_2023': "Ethereum in 2023 emerged from its post-Merge phase, showing stability after transitioning to Proof of Stake in 2022. ETH price ranged from ~$1,200 to ~$2,400, with increased focus on Layer 2 scaling solutions.",
    'eth_2024': "2024 saw Ethereum gain momentum with the Dencun upgrade in March, significantly reducing Layer 2 transaction costs. ETH price action was influenced by Bitcoin's ETF approval and speculation about Ethereum's own ETF prospects.",
    'sol_2023': "Solana experienced a remarkable recovery in 2023 after FTX's collapse threatened the ecosystem in 2022. SOL rose from $10 to over $100 by year-end, a 1000% increase, regaining developer and user trust.",
    'sol_2024': "Solana continued its strong momentum in 2024 with increasing adoption, improved stability, and expansion of its DeFi and NFT ecosystems. SOL maintained competition with Ethereum while focusing on performance and reliability."
}

# Add a health check route for Render
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "ok", "message": "Stochastic Oscillator Trading Bot API is running"})

@app.route('/api/available-data', methods=['GET'])
def get_available_data():
    """Return a list of available data files for backtesting with historical context"""
    data_options = []
    for key in data_files.keys():
        option = {
            'id': key, 
            'name': f"{key.split('_')[0].upper()} {key.split('_')[1]}",
            'context': historical_context.get(key, "")
        }
        data_options.append(option)
        
    return jsonify({'data_options': data_options})

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
            'd_period': int(params.get('d_period', 5)),
            'position_sizing': params.get('position_sizing', 'percentage'),
            'fixed_position_size': float(params.get('fixed_position_size', 0.1)),
            'position_size_percentage': float(params.get('position_size_percentage', 90))
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