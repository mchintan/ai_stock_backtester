# Stock Portfolio Monte Carlo Simulator

This application provides a web interface for analyzing stock portfolios using Monte Carlo simulations. It allows users to:
- Input multiple stock tickers
- Analyze historical price data over the last 5 years
- Calculate key statistics (median returns, standard deviation, range)
- Run Monte Carlo simulations with configurable parameters
- Perform convergence analysis
- Compare multiple portfolios
- Analyze dividend performance and projections

## Local Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
uvicorn main:app --reload
```

4. Open the web interface in your browser:
```
http://localhost:8000
```

## Deployment Instructions

### Deploy to Render.com (Recommended)

1. Fork this repository to your GitHub account

2. Create a new Web Service on Render.com:
   - Go to https://dashboard.render.com
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository you just forked

3. Configure the Web Service:
   - Name: stock-portfolio-simulator (or your preferred name)
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Select the free plan

4. Click "Create Web Service"

The application will be automatically deployed and available at your Render.com URL.

## Features
- Historical stock data analysis
- Statistical calculations
- Monte Carlo simulations
- Convergence analysis
- Interactive visualizations
- Configurable simulation parameters
- Multiple portfolio comparison
- Dividend analysis and projections

## Technologies Used
- FastAPI (Backend)
- Python with pandas, numpy, and yfinance
- Plotly.js for visualizations
- Bootstrap for UI
- JavaScript for frontend interactivity 