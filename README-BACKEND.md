# Stochastic Oscillator Trading Bot - Backend

This is the backend API for the Stochastic Oscillator Trading Bot. It provides endpoints for backtesting cryptocurrency trading strategies based on the Stochastic Oscillator technical indicator.

## Features

- Flask REST API for running cryptocurrency backtests
- Implementation of the Stochastic Oscillator trading strategy
- Support for multiple data sources (BTC, ETH, SOL)
- Detailed trade analysis and performance metrics

## Setup Instructions

1. Clone this repository:
   ```
   git clone <your-private-repo-url>
   cd trading-bot-backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file based on the provided `.env.example`:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file with your configuration.

6. Start the server:
   ```
   python app.py
   ```
   The API will run on http://localhost:5000 by default.

## API Endpoints

- `GET /api/available-data`: Get list of available data sources
- `POST /api/backtest`: Run a backtest with specified parameters
- `GET /api/strategy-info`: Get information about the trading strategy

## Data Files

The repository includes CSV files with historical price data for various cryptocurrencies:
- Bitcoin (BTC): 2022-2024
- Ethereum (ETH): 2023-2024
- Solana (SOL): 2023-2024

## Connect with Frontend

The backend API is designed to work with the [Stochastic Oscillator Trading Bot Frontend](https://github.com/yourusername/trading-bot-frontend), which provides a user-friendly interface for interacting with this API.

## Security Notes

- This repository should remain private as it may contain sensitive data
- API keys and secrets should never be committed to the repository
- Use environment variables for all sensitive information 