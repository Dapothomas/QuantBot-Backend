# Stochastic Oscillator Trading Bot Web Application

This web application allows you to visualize and test a cryptocurrency trading bot that uses the Stochastic Oscillator indicator for trading decisions.

## Features

- Interactive backtesting with configurable parameters
- Real-time chart visualization with trade markers
- Detailed trade history and performance metrics
- Strategy explanation and documentation
- Dark mode support and responsive design

## Project Structure

- `app.py`: Flask backend API
- `frontend/`: React frontend application
  - `src/`: Source files
    - `components/`: React components
    - `utils/`: Utility functions
  - `public/`: Static assets
- `*.py`: Python strategy and backtest files

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install backend dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the Flask API server:
   ```
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

## Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Configure your backtest parameters
3. Click "Run Backtest" to see the results
4. View the price chart, trade markers, and performance metrics
5. Toggle dark mode using the theme button in the header

## Technologies Used

- **Backend**: Flask, Python, Pandas, NumPy
- **Frontend**: React, Tailwind CSS, Lightweight Charts
- **Data Visualization**: Lightweight Charts by TradingView

## Future Enhancements

- Add more trading strategies
- Implement live trading capabilities
- Add user authentication and profile management
- Include more technical indicators

## License

MIT 