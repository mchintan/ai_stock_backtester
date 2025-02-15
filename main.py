from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
import concurrent.futures
import asyncio
from enum import Enum
from scipy.optimize import minimize

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Mount static directory
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)  # Create static directory if it doesn't exist
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create templates
templates = Jinja2Templates(directory=str(static_dir))

class PortfolioGroup(BaseModel):
    name: str
    tickers: List[str]

# Add new models for optimization
class OptimizationObjective(str, Enum):
    SHARPE_RATIO = "sharpe_ratio"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    MAX_SORTINO = "max_sortino"
    RISK_PARITY = "risk_parity"  # Add risk parity option

class StockRequest(BaseModel):
    portfolio_groups: List[PortfolioGroup]
    num_simulations: int = 1000
    time_horizon: int = 60
    start_year: int = 2010
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    min_weight: float = Field(0.0, ge=0.0, le=1.0)
    max_weight: float = Field(1.0, ge=0.0, le=1.0)

# Add caching for historical data fetching
@lru_cache(maxsize=100)
def fetch_cached_stock_data(ticker: str, start_date: datetime, end_date: datetime):
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)

# Modify fetch_historical_data to use parallel processing
async def fetch_historical_data(tickers: List[str], start_year: int = 2000) -> Dict:
    end_date = datetime.now()
    start_date = datetime(start_year, 1, 1)
    
    price_dfs = {}
    dividend_data = {}
    
    # Use ThreadPoolExecutor for parallel data fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        futures = {executor.submit(fetch_cached_stock_data, ticker, start_date, end_date): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                hist = future.result()
                if not hist.empty:
                    # Get the close prices and resample to month-end
                    close_prices = hist['Close'].resample('ME').last()
                    price_dfs[ticker] = close_prices
                    
                    # Get dividend data
                    dividends = hist['Dividends'].resample('ME').sum()  # Monthly dividend sums
                    if not dividends.empty and dividends.sum() > 0:
                        annual_dividends = dividends.resample('YE').sum()  # Changed from 'Y' to 'YE'
                        latest_price = close_prices.iloc[-1]
                        latest_annual_div = dividends.rolling(12).sum().iloc[-1]  # Rolling 12-month dividend
                        
                        # Calculate dividend growth rate with safety checks
                        if len(annual_dividends) > 1 and annual_dividends.iloc[0] > 0:
                            growth_rate = ((annual_dividends.iloc[-1] / annual_dividends.iloc[0]) ** (1 / max(1, len(annual_dividends)-1))) - 1
                            # Ensure growth rate is within reasonable bounds
                            growth_rate = min(max(growth_rate, -0.5), 0.5)  # Cap between -50% and +50%
                        else:
                            growth_rate = 0.0
                        
                        dividend_data[ticker] = {
                            'dividend_yield': float(latest_annual_div / latest_price) if latest_price > 0 else 0.0,
                            'dividend_growth': float(growth_rate),
                            'dividend_history': [float(d) for d in annual_dividends.values],
                            'dividend_years': [int(d.year) for d in annual_dividends.index],
                            'monthly_dividends': [float(d) for d in dividends.values],
                            'dividend_months': [d.strftime("%Y-%m-%d") for d in dividends.index]
                        }
                    else:
                        dividend_data[ticker] = {
                            'dividend_yield': 0.0,
                            'dividend_growth': 0.0,
                            'dividend_history': [],
                            'dividend_years': [],
                            'monthly_dividends': [0.0] * len(close_prices),
                            'dividend_months': [d.strftime("%Y-%m-%d") for d in close_prices.index]
                        }
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")
    
    # Calculate portfolio-level dividend metrics with safety checks
    if dividend_data and price_dfs:
        total_portfolio_value = sum(price_dfs[ticker].iloc[-1] for ticker in price_dfs.keys())
        if total_portfolio_value > 0:
            portfolio_dividend_data = {
                'portfolio_yield': min(
                    sum(
                        dividend_data[ticker]['dividend_yield'] * price_dfs[ticker].iloc[-1] / total_portfolio_value 
                        for ticker in dividend_data.keys()
                    ), 
                    1.0
                ),
                'weighted_growth': min(
                    max(
                        sum(
                            dividend_data[ticker]['dividend_growth'] * price_dfs[ticker].iloc[-1] / total_portfolio_value 
                            for ticker in dividend_data.keys()
                        ), 
                        -0.5
                    ), 
                    0.5
                ),
                'annual_income': float(
                    sum(
                        dividend_data[ticker]['dividend_yield'] * price_dfs[ticker].iloc[-1] 
                        for ticker in dividend_data.keys()
                    )
                ),
                'individual_dividends': dividend_data
            }
        else:
            portfolio_dividend_data = {
                'portfolio_yield': 0.0,
                'weighted_growth': 0.0,
                'annual_income': 0.0,
                'individual_dividends': dividend_data
            }
    else:
        portfolio_dividend_data = {
            'portfolio_yield': 0.0,
            'weighted_growth': 0.0,
            'annual_income': 0.0,
            'individual_dividends': {}
        }
    
    return {
        'prices': pd.DataFrame(price_dfs),
        'dividends': portfolio_dividend_data
    }

def calculate_statistics(df: pd.DataFrame) -> Dict:
    """Calculate key statistics from historical data."""
    # Calculate monthly returns
    returns = df.pct_change().dropna()
    
    # Calculate cumulative portfolio performance (equal weight)
    portfolio_values = (1 + returns.mean(axis=1)).cumprod()
    
    # Annualize factor for monthly data
    annualize_factor = 12
    
    stats = {
        'individual': {},
        'portfolio': {}
    }
    
    # Individual stock statistics
    for column in returns.columns:
        stats['individual'][column] = {
            'median_monthly_return': float(returns[column].median()),
            'annual_return': float((1 + returns[column].mean()) ** annualize_factor - 1),
            'annual_std_dev': float(returns[column].std() * np.sqrt(annualize_factor)),
            'monthly_return_range': [float(returns[column].min()), float(returns[column].max())],
            'total_return': float((1 + returns[column]).prod() - 1),
            'sharpe_ratio': float((returns[column].mean() * annualize_factor) / (returns[column].std() * np.sqrt(annualize_factor)))
        }
    
    # Portfolio statistics (equal weight)
    portfolio_returns = returns.mean(axis=1)
    stats['portfolio'] = {
        'median_monthly_return': float(portfolio_returns.median()),
        'annual_return': float((1 + portfolio_returns.mean()) ** annualize_factor - 1),
        'annual_std_dev': float(portfolio_returns.std() * np.sqrt(annualize_factor)),
        'monthly_return_range': [float(portfolio_returns.min()), float(portfolio_returns.max())],
        'total_return': float((1 + portfolio_returns).prod() - 1),
        'sharpe_ratio': float((portfolio_returns.mean() * annualize_factor) / (portfolio_returns.std() * np.sqrt(annualize_factor))),
        'historical_performance': {
            'dates': [d.strftime("%Y-%m-%d") for d in returns.index],
            'values': [float(v) for v in portfolio_values.values],
            'returns': [float(r) for r in portfolio_returns.values]
        }
    }
    
    return stats

def run_monte_carlo(df: pd.DataFrame, num_simulations: int, time_horizon: int) -> Dict:
    """Run Monte Carlo simulations."""
    daily_returns = df.pct_change().dropna()
    
    # Calculate mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    # Initialize simulation results
    simulation_results = []
    
    # Run simulations
    for _ in range(num_simulations):
        # Generate random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            time_horizon
        )
        
        # Calculate cumulative portfolio value (equal weight)
        portfolio_values = (1 + simulated_returns.mean(axis=1)).cumprod()
        simulation_results.append([float(x) for x in portfolio_values])
    
    # Convert results to numpy array
    simulation_array = np.array(simulation_results)
    
    # Calculate percentiles
    percentiles = {
        '5th': [float(x) for x in np.percentile(simulation_array, 5, axis=0)],
        '25th': [float(x) for x in np.percentile(simulation_array, 25, axis=0)],
        '50th': [float(x) for x in np.percentile(simulation_array, 50, axis=0)],
        '75th': [float(x) for x in np.percentile(simulation_array, 75, axis=0)],
        '95th': [float(x) for x in np.percentile(simulation_array, 95, axis=0)]
    }
    
    return {
        'percentiles': percentiles,
        'final_values': [float(x) for x in simulation_array[:, -1]]
    }

def analyze_convergence(df: pd.DataFrame, max_simulations: int, time_horizon: int) -> Dict:
    """Analyze convergence of Monte Carlo simulations."""
    simulation_counts = [100, 500, 1000, 2000, max_simulations]
    convergence_results = []
    
    for num_sims in simulation_counts:
        results = run_monte_carlo(df, num_sims, time_horizon)
        final_values = results['final_values']
        convergence_results.append({
            'num_simulations': num_sims,
            'mean': float(np.mean(final_values)),
            'std': float(np.std(final_values))
        })
    
    return {'convergence': convergence_results}

def calculate_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame) -> Dict:
    """Calculate various portfolio metrics."""
    annual_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12
    
    portfolio_return = np.sum(annual_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate Sortino ratio (using downside deviation)
    risk_free_rate = 0.02
    
    # Calculate portfolio returns for downside deviation
    portfolio_returns = returns.dot(weights)
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = np.sqrt(np.sum(negative_returns**2) / len(returns)) * np.sqrt(12)
    
    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe_ratio': (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
        'sortino_ratio': (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    }

def calculate_risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Calculate risk contribution of each asset."""
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk_contribution = np.dot(cov_matrix, weights)
    risk_contribution = np.multiply(marginal_risk_contribution, weights) / portfolio_risk
    return risk_contribution

def risk_parity_objective(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Objective function for risk parity optimization."""
    risk_contrib = calculate_risk_contribution(weights, cov_matrix)
    target_risk = 1.0 / len(weights)  # Equal risk contribution
    return np.sum((risk_contrib - target_risk) ** 2)

def calculate_historical_rebalance(
    df: pd.DataFrame,
    weights: np.ndarray,
    rebalance_frequency: str = 'M',  # 'M' for monthly, 'Q' for quarterly, 'Y' for yearly
    transaction_cost: float = 0.001  # 0.1% transaction cost
) -> Dict:
    """Calculate historical performance with periodic rebalancing."""
    returns = df.pct_change().dropna()
    
    # Initialize portfolio
    portfolio_value = 1.0
    current_weights = weights.copy()
    portfolio_values = []
    turnover_history = []
    rebalance_dates = []
    
    # Resample dates for rebalancing
    rebalance_points = returns.resample(rebalance_frequency).last().index
    
    for date in returns.index:
        # Apply daily returns
        daily_return = np.sum(returns.loc[date] * current_weights)
        portfolio_value *= (1 + daily_return)
        
        # Update weights due to price changes
        current_weights *= (1 + returns.loc[date])
        current_weights /= current_weights.sum()
        
        portfolio_values.append(portfolio_value)
        
        # Rebalance if needed
        if date in rebalance_points:
            old_weights = current_weights.copy()
            current_weights = weights.copy()
            
            # Calculate turnover
            turnover = np.sum(np.abs(current_weights - old_weights))
            transaction_cost_impact = turnover * transaction_cost
            
            # Apply transaction costs
            portfolio_value *= (1 - transaction_cost_impact)
            
            turnover_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'turnover': float(turnover),
                'cost': float(transaction_cost_impact)
            })
            rebalance_dates.append(date.strftime('%Y-%m-%d'))
    
    return {
        'portfolio_values': portfolio_values,
        'dates': [d.strftime('%Y-%m-%d') for d in returns.index],
        'turnover_history': turnover_history,
        'rebalance_dates': rebalance_dates,
        'final_value': float(portfolio_value),
        'total_turnover': float(np.sum([t['turnover'] for t in turnover_history])),
        'total_cost': float(np.sum([t['cost'] for t in turnover_history]))
    }

def find_optimal_portfolio(
    df: pd.DataFrame,
    objective: OptimizationObjective,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    n_portfolios: int = 10000
) -> Dict:
    """Find optimal portfolio weights using various optimization methods."""
    returns = df.pct_change().dropna()
    n_assets = len(df.columns)
    cov_matrix = returns.cov() * 12
    annual_returns = returns.mean() * 12
    
    if objective == OptimizationObjective.RISK_PARITY:
        # Risk Parity Optimization (existing code)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        )
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        result = minimize(
            risk_parity_objective,
            np.array([1/n_assets] * n_assets),
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
    else:
        # Generate random portfolios for other optimization objectives
        weights = np.random.uniform(min_weight, max_weight, (n_portfolios, n_assets))
        weights = weights / weights.sum(axis=1)[:, np.newaxis]  # Normalize weights
        
        # Calculate portfolio metrics for each random portfolio
        portfolio_metrics = []
        for w in weights:
            metrics = calculate_portfolio_metrics(w, returns)
            portfolio_metrics.append({
                'weights': w,
                'return': metrics['return'],
                'risk': metrics['risk'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio']
            })
        
        # Find optimal portfolio based on objective
        if objective == OptimizationObjective.SHARPE_RATIO:
            optimal_idx = max(range(len(portfolio_metrics)), 
                            key=lambda i: portfolio_metrics[i]['sharpe_ratio'])
        elif objective == OptimizationObjective.MIN_VOLATILITY:
            optimal_idx = min(range(len(portfolio_metrics)), 
                            key=lambda i: portfolio_metrics[i]['risk'])
        elif objective == OptimizationObjective.MAX_RETURN:
            optimal_idx = max(range(len(portfolio_metrics)), 
                            key=lambda i: portfolio_metrics[i]['return'])
        elif objective == OptimizationObjective.MAX_SORTINO:
            optimal_idx = max(range(len(portfolio_metrics)), 
                            key=lambda i: portfolio_metrics[i]['sortino_ratio'])
        
        optimal_weights = portfolio_metrics[optimal_idx]['weights']
    
    # Calculate final metrics for optimal portfolio
    metrics = calculate_portfolio_metrics(optimal_weights, returns)
    
    # Calculate risk contributions if needed
    risk_contrib = calculate_risk_contribution(optimal_weights, cov_matrix)
    
    return {
        'optimal_weights': [float(w) for w in optimal_weights],
        'tickers': list(df.columns),
        'metrics': {
            'expected_annual_return': float(metrics['return']),
            'annual_volatility': float(metrics['risk']),
            'sharpe_ratio': float(metrics['sharpe_ratio']),
            'sortino_ratio': float(metrics['sortino_ratio']),
            'risk_contributions': [float(r) for r in risk_contrib]
        },
        'rebalance_analysis': calculate_historical_rebalance(df, optimal_weights)
    }

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    """Main endpoint for stock analysis and simulation."""
    results = {}
    
    for portfolio in request.portfolio_groups:
        # Fetch historical data
        data = await fetch_historical_data(portfolio.tickers, request.start_year)
        df = data['prices']
        dividend_data = data['dividends']
        
        # Calculate statistics
        stats = calculate_statistics(df)
        
        # Run Monte Carlo simulation
        simulation_results = run_monte_carlo(df, request.num_simulations, request.time_horizon)
        
        # Analyze convergence
        convergence_results = analyze_convergence(df, request.num_simulations, request.time_horizon)
        
        # Find optimal portfolio
        optimization_results = find_optimal_portfolio(
            df,
            request.optimization_objective,
            request.min_weight,
            request.max_weight
        )
        
        # Project future dividend income
        initial_portfolio_value = 1.0
        projected_value = simulation_results['percentiles']['50th'][-1]
        projected_annual_income = dividend_data['portfolio_yield'] * projected_value
        projected_income_growth = (1 + dividend_data['weighted_growth']) ** request.time_horizon
        final_projected_income = projected_annual_income * projected_income_growth
        
        dividend_projections = {
            'current_yield': dividend_data['portfolio_yield'],
            'weighted_growth_rate': dividend_data['weighted_growth'],
            'current_annual_income': dividend_data['annual_income'],
            'projected_annual_income': float(final_projected_income),
            'individual_metrics': dividend_data['individual_dividends']
        }
        
        # Add metadata
        metadata = {
            "data_start": df.index[0].strftime("%Y-%m-%d"),
            "data_end": df.index[-1].strftime("%Y-%m-%d"),
            "frequency": "monthly",
            "number_of_months": len(df),
            "tickers_analyzed": list(df.columns),
            "portfolio_description": f"Optimized portfolio of {len(df.columns)} stocks"
        }
        
        results[portfolio.name] = {
            "metadata": metadata,
            "statistics": stats,
            "simulation_results": simulation_results,
            "convergence_analysis": convergence_results,
            "dividend_analysis": dividend_projections,
            "optimization_results": optimization_results
        }
    
    return results

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request}) 