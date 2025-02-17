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

def calculate_historical_rebalance(df: pd.DataFrame, optimal_weights: np.ndarray) -> Dict:
    """Calculate historical performance with periodic rebalancing."""
    returns = df.pct_change().dropna()
    
    # Initialize portfolio with $1
    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    current_weights = optimal_weights.copy()
    
    # Track rebalancing points and turnover
    rebalance_dates = []
    turnover_history = []
    total_turnover = 0.0
    transaction_cost = 0.001  # 0.1% transaction cost
    
    # Monthly rebalancing
    rebalance_points = returns.resample('ME').last().index
    
    for date in returns.index:
        # Update portfolio value
        daily_return = np.sum(current_weights * returns.loc[date])
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(float(portfolio_value))
        
        # Update weights due to price changes
        current_weights = current_weights * (1 + returns.loc[date])
        current_weights = current_weights / np.sum(current_weights)
        
        if date in rebalance_points:
            # Calculate turnover
            weight_diff = np.abs(current_weights - optimal_weights)
            turnover = np.sum(weight_diff) / 2
            
            # Record weight changes
            weight_changes = []
            for i, (old_w, new_w) in enumerate(zip(current_weights, optimal_weights)):
                abs_change = float(new_w - old_w)
                pct_change = float(abs_change / old_w * 100) if old_w != 0 else float('inf')
                direction = 'up' if abs_change > 0 else 'down' if abs_change < 0 else 'none'
                
                weight_changes.append({
                    'ticker': df.columns[i],
                    'old_weight': float(old_w),
                    'new_weight': float(new_w),
                    'absolute_change': abs_change,
                    'percent_change': pct_change,
                    'direction': direction
                })
            
            turnover_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'turnover': float(turnover),
                'weight_changes': weight_changes
            })
            
            # Apply transaction costs
            portfolio_value *= (1 - turnover * transaction_cost)
            total_turnover += float(turnover)
            
            # Rebalance to optimal weights
            current_weights = optimal_weights.copy()
            rebalance_dates.append(date.strftime('%Y-%m-%d'))
    
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in returns.index],
        'portfolio_values': [float(v) for v in portfolio_values[:-1]],  # Exclude last duplicate value
        'rebalance_dates': rebalance_dates,
        'turnover_history': turnover_history,
        'total_turnover': float(total_turnover),
        'total_cost': float(total_turnover * transaction_cost),
        'final_value': float(portfolio_values[-2])  # Use second to last value
    }

def minimize_volatility(weights: np.ndarray, returns: pd.DataFrame, cov_matrix: np.ndarray) -> float:
    """Objective function for minimizing portfolio volatility."""
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_std

def maximize_sharpe(weights: np.ndarray, returns: pd.DataFrame, cov_matrix: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Objective function for maximizing Sharpe ratio (negative for minimization)."""
    portfolio_return = np.sum(returns.mean() * weights) * 12
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -1 * (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0

def maximize_sortino(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Objective function for maximizing Sortino ratio (negative for minimization)."""
    portfolio_returns = returns.dot(weights)
    portfolio_return = portfolio_returns.mean() * 12
    
    # Calculate downside deviation
    negative_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = np.sqrt(np.sum(negative_returns**2) / len(returns)) * np.sqrt(12)
    
    return -1 * (portfolio_return - risk_free_rate) / downside_std if downside_std > 0 else 0

def maximize_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Objective function for maximizing portfolio return (negative for minimization)."""
    portfolio_return = np.sum(returns.mean() * weights) * 12
    return -1 * portfolio_return

def find_optimal_portfolio(
    df: pd.DataFrame,
    objective: OptimizationObjective,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    risk_free_rate: float = 0.02
) -> Dict:
    """Find optimal portfolio weights using various optimization methods."""
    returns = df.pct_change().dropna()
    n_assets = len(df.columns)
    cov_matrix = returns.cov() * 12
    annual_returns = returns.mean() * 12
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    if objective == OptimizationObjective.RISK_PARITY:
        result = minimize(
            risk_parity_objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    else:
        # Choose objective function based on optimization goal
        objective_functions = {
            OptimizationObjective.MIN_VOLATILITY: (minimize_volatility, (returns, cov_matrix)),
            OptimizationObjective.SHARPE_RATIO: (maximize_sharpe, (returns, cov_matrix, risk_free_rate)),
            OptimizationObjective.MAX_SORTINO: (maximize_sortino, (returns, risk_free_rate)),
            OptimizationObjective.MAX_RETURN: (maximize_return, (returns,))
        }
        
        obj_function, args = objective_functions[objective]
        
        result = minimize(
            obj_function,
            initial_weights,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    optimal_weights = result.x
    
    # Calculate metrics for the optimal portfolio
    metrics = calculate_portfolio_metrics(optimal_weights, returns)
    risk_contrib = calculate_risk_contribution(optimal_weights, cov_matrix)
    
    # Calculate efficient frontier points for visualization
    n_points = 100
    efficient_frontier = calculate_efficient_frontier(returns, cov_matrix, n_points, min_weight, max_weight)
    
    # Convert NumPy types to Python native types
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
        'efficient_frontier': efficient_frontier,
        'optimization_details': {
            'objective': objective.value,
            'min_weight': float(min_weight),
            'max_weight': float(max_weight),
            'risk_free_rate': float(risk_free_rate),
            'convergence': bool(result.success),
            'message': str(result.message)
        },
        'rebalance_analysis': calculate_historical_rebalance(df, optimal_weights)
    }

def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility (standard deviation)."""
    # Input validation
    if len(weights) == 0 or len(cov_matrix) == 0:
        raise ValueError("Empty weights or covariance matrix")
    if len(weights) != len(cov_matrix):
        raise ValueError("Dimension mismatch between weights and covariance matrix")
    
    try:
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    except Exception as e:
        raise ValueError(f"Error calculating portfolio volatility: {str(e)}")

def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Calculate portfolio expected return."""
    # Input validation
    if len(weights) == 0 or returns.empty:
        raise ValueError("Empty weights or returns")
    if len(weights) != len(returns.columns):
        raise ValueError("Dimension mismatch between weights and returns")
    
    try:
        return np.sum(returns.mean() * weights) * 12
    except Exception as e:
        raise ValueError(f"Error calculating portfolio return: {str(e)}")

def portfolio_sharpe_ratio(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Calculate portfolio Sharpe ratio."""
    # Input validation
    if len(weights) == 0 or returns.empty:
        raise ValueError("Empty weights or returns")
    if not (0 <= risk_free_rate <= 1):
        raise ValueError("Risk-free rate must be between 0 and 1")
    
    try:
        ret = portfolio_return(weights, returns)
        vol = portfolio_volatility(weights, returns.cov() * 12)
        return (ret - risk_free_rate) / vol if vol > 0 else 0
    except Exception as e:
        raise ValueError(f"Error calculating Sharpe ratio: {str(e)}")

def calculate_efficient_frontier(returns: pd.DataFrame, cov_matrix: pd.DataFrame, n_points: int, min_weight: float, max_weight: float) -> List[Dict]:
    """Calculate points along the efficient frontier."""
    frontier_points = []
    
    # Get min and max returns from individual assets
    min_ret = returns.mean().min() * 12
    max_ret = returns.mean().max() * 12
    
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    for target_return in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return}  # target return constraint
        ]
        
        bounds = tuple((min_weight, max_weight) for _ in range(len(returns.columns)))
        
        result = minimize(
            portfolio_volatility,
            x0=np.array([1/len(returns.columns)] * len(returns.columns)),
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            risk = portfolio_volatility(weights, cov_matrix)
            sharpe = portfolio_sharpe_ratio(weights, returns, 0.02)  # Using 2% risk-free rate
            
            frontier_points.append({
                'risk': float(risk),
                'return': float(target_return),
                'sharpe': float(sharpe)
            })
    
    return frontier_points

@app.post("/api/analyze")
async def analyze_stocks(request: StockRequest):
    """Main endpoint for stock analysis and simulation."""
    results = {}
    
    # Validate request
    if not request.portfolio_groups:
        raise HTTPException(status_code=400, detail="At least one portfolio group is required")
    
    for portfolio in request.portfolio_groups:
        # Validate portfolio
        if not portfolio.tickers:
            raise HTTPException(
                status_code=400, 
                detail=f"Portfolio '{portfolio.name}' must contain at least one ticker"
            )
        
        if len(portfolio.tickers) < 2:
            raise HTTPException(
                status_code=400, 
                detail=f"Portfolio '{portfolio.name}' must contain at least 2 stocks for diversification"
            )
        
        # Validate simulation parameters
        if request.num_simulations < 100:
            raise HTTPException(
                status_code=400,
                detail="Number of simulations must be at least 100"
            )
        
        if request.time_horizon < 1:
            raise HTTPException(
                status_code=400,
                detail="Time horizon must be at least 1 month"
            )
        
        # Fetch historical data
        try:
            data = await fetch_historical_data(portfolio.tickers, request.start_year)
            df = data['prices']
            dividend_data = data['dividends']
            
            # Check if we have enough data
            if df.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"No price data available for portfolio '{portfolio.name}'"
                )
            
            if len(df) < 12:  # At least 12 months of data
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient historical data for portfolio '{portfolio.name}'. Need at least 12 months."
                )
            
            # Calculate statistics
            stats = calculate_statistics(df)
            
            # Run Monte Carlo simulation
            simulation_results = run_monte_carlo(df, request.num_simulations, request.time_horizon)
            
            # Analyze convergence
            convergence_results = analyze_convergence(df, request.num_simulations, request.time_horizon)
            
            try:
                # Find optimal portfolio
                optimization_results = find_optimal_portfolio(
                    df,
                    request.optimization_objective,
                    request.min_weight,
                    request.max_weight
                )
            except ValueError as e:
                # If optimization fails, return error in results rather than failing completely
                optimization_results = {
                    "error": str(e),
                    "status": "failed",
                    "optimal_weights": [1.0/len(df.columns)] * len(df.columns)  # Default to equal weights
                }
            
            # Project future dividend income with safety checks
            initial_portfolio_value = 1.0
            projected_value = simulation_results['percentiles']['50th'][-1]
            projected_annual_income = dividend_data['portfolio_yield'] * projected_value
            projected_income_growth = (1 + max(min(dividend_data['weighted_growth'], 0.5), -0.5)) ** request.time_horizon
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
            
        except Exception as e:
            # Log the error and return a more user-friendly message
            print(f"Error processing portfolio '{portfolio.name}': {str(e)}")
            results[portfolio.name] = {
                "error": f"Failed to process portfolio: {str(e)}",
                "status": "failed"
            }
    
    if not results:
        raise HTTPException(
            status_code=500,
            detail="Failed to process any portfolios"
        )
    
    return results

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request}) 