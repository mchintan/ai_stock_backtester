// Add portfolio group management
document.getElementById('addPortfolioBtn').addEventListener('click', () => {
    const portfolioGroups = document.getElementById('portfolioGroups');
    const newGroup = document.createElement('div');
    newGroup.className = 'portfolio-group mb-4';
    newGroup.innerHTML = `
        <div class="mb-3">
            <label class="form-label">Portfolio Name</label>
            <input type="text" class="form-control portfolio-name" placeholder="e.g., Tech Stocks">
        </div>
        <div class="mb-3">
            <label class="form-label">Stock Tickers (comma-separated)</label>
            <input type="text" class="form-control portfolio-tickers" placeholder="AAPL,MSFT,GOOGL">
        </div>
        <button type="button" class="btn btn-outline-danger btn-sm remove-portfolio">Remove Portfolio</button>
    `;
    portfolioGroups.appendChild(newGroup);
});

document.getElementById('portfolioGroups').addEventListener('click', (e) => {
    if (e.target.classList.contains('remove-portfolio')) {
        e.target.closest('.portfolio-group').remove();
    }
});

document.getElementById('stockForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const portfolioGroups = [];
    document.querySelectorAll('.portfolio-group').forEach(group => {
        const name = group.querySelector('.portfolio-name').value.trim();
        const tickers = group.querySelector('.portfolio-tickers').value.split(',').map(t => t.trim());
        
        if (name && tickers.length > 0 && tickers[0] !== '') {
            portfolioGroups.push({ name, tickers });
        }
    });
    
    if (portfolioGroups.length === 0) {
        alert('Please add at least one portfolio group with valid tickers');
        return;
    }
    
    // Get optimization parameters
    const optimizationObjective = document.getElementById('optimizationObjective').value;
    const minWeight = parseFloat(document.getElementById('minWeight').value) / 100;
    const maxWeight = parseFloat(document.getElementById('maxWeight').value) / 100;
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                portfolio_groups: portfolioGroups,
                num_simulations: parseInt(document.getElementById('numSimulations').value),
                time_horizon: parseInt(document.getElementById('timeHorizon').value),
                start_year: parseInt(document.getElementById('startYear').value),
                optimization_objective: optimizationObjective,
                min_weight: minWeight,
                max_weight: maxWeight
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing stocks. Please check the console for details.');
    }
});

function displayResults(data) {
    // Clear previous results
    clearPreviousResults();
    
    const portfolioCount = Object.keys(data).length;
    
    // Show/hide sections based on portfolio count
    const comparisonSections = document.querySelectorAll('.comparison-section');
    const singlePortfolioSections = document.querySelectorAll('.single-portfolio-section');
    
    if (portfolioCount > 1) {
        // Show comparison sections, hide single portfolio sections
        comparisonSections.forEach(section => section.style.display = 'block');
        singlePortfolioSections.forEach(section => section.style.display = 'none');
        
        // Display comparison sections
        displayPortfolioComparison(data);
        displayMonteCarloComparison(data);
    } else {
        // Hide comparison sections, show single portfolio sections
        comparisonSections.forEach(section => section.style.display = 'none');
        singlePortfolioSections.forEach(section => section.style.display = 'block');
        
        // Display single portfolio data
        const [portfolioName, portfolioData] = Object.entries(data)[0];
        
        // Update portfolio description with metadata
        const descContainer = document.getElementById('portfolioDescription');
        descContainer.innerHTML = `
            <p class="lead">${portfolioData.metadata.portfolio_description}</p>
            <p>Historical analysis from ${portfolioData.metadata.data_start} to ${portfolioData.metadata.data_end}</p>
            <p>Analysis frequency: ${portfolioData.metadata.frequency}</p>
            <p>Number of months analyzed: ${portfolioData.metadata.number_of_months}</p>
        `;
        
        displayStatistics(portfolioData.statistics, 'statistics');
        displayHistoricalPerformance(portfolioData.metadata, portfolioData.statistics.portfolio, 'historicalValuePlot');
        displayDividendAnalysis(portfolioData.dividend_analysis, portfolioData.simulation_results, 'dividendStats');
        displaySimulationPlot(portfolioData.simulation_results);
        displayConvergencePlot(portfolioData.convergence_analysis);
        
        // Display optimization results
        if (portfolioData.optimization_results) {
            displayOptimizationResults(portfolioData.optimization_results);
        }
    }
}

function clearPreviousResults() {
    // Clear any existing portfolio detail sections
    const existingDetails = document.querySelectorAll('.portfolio-details');
    existingDetails.forEach(detail => detail.remove());
    
    // Clear all content containers
    const containers = [
        'statistics',
        'portfolioDescription',
        'historicalValuePlot',
        'returnsHistogramPlot',
        'dividendStats',
        'individualDividends',
        'simulationPlot',
        'convergencePlot',
        'portfolioComparison',
        'simulationComparison'
    ];
    
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.innerHTML = '';
        }
    });
}

function displayPortfolioComparison(data) {
    const container = document.getElementById('portfolioComparison');
    let html = `
        <div class="table-responsive">
            <table class="table table-sm statistics-table">
                <thead>
                    <tr>
                        <th>Portfolio</th>
                        <th>Annual Return</th>
                        <th>Annual Std Dev</th>
                        <th>Sharpe Ratio</th>
                        <th>Total Return</th>
                        <th>Dividend Yield</th>
                        <th>Projected Value</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    Object.entries(data).forEach(([portfolioName, portfolioData]) => {
        const stats = portfolioData.statistics.portfolio;
        const projectedValue = portfolioData.simulation_results.percentiles['50th'][portfolioData.simulation_results.percentiles['50th'].length - 1];
        
        html += `
            <tr>
                <td>${portfolioName}</td>
                <td>${(stats.annual_return * 100).toFixed(2)}%</td>
                <td>${(stats.annual_std_dev * 100).toFixed(2)}%</td>
                <td>${stats.sharpe_ratio.toFixed(2)}</td>
                <td>${(stats.total_return * 100).toFixed(2)}%</td>
                <td>${(portfolioData.dividend_analysis.current_yield * 100).toFixed(2)}%</td>
                <td>$${projectedValue.toFixed(2)}</td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    container.innerHTML = html;
}

function displayMonteCarloComparison(data) {
    const traces = [];
    const colors = ['#2c3e50', '#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6'];
    
    Object.entries(data).forEach(([portfolioName, portfolioData], index) => {
        const color = colors[index % colors.length];
        const timePoints = Array.from({length: portfolioData.simulation_results.percentiles['50th'].length}, (_, i) => i);
        
        // Add median line
        traces.push({
            name: `${portfolioName} (Median)`,
            y: portfolioData.simulation_results.percentiles['50th'],
            x: timePoints,
            line: {color: color, width: 2},
            type: 'scatter'
        });
        
        // Add confidence interval
        traces.push({
            name: `${portfolioName} (25th-75th)`,
            y: portfolioData.simulation_results.percentiles['75th'],
            x: timePoints,
            line: {color: color, width: 0},
            fillcolor: color,
            opacity: 0.1,
            showlegend: false,
            type: 'scatter'
        });
        
        traces.push({
            name: `${portfolioName} (25th-75th)`,
            y: portfolioData.simulation_results.percentiles['25th'],
            x: timePoints,
            line: {color: color, width: 0},
            fillcolor: color,
            opacity: 0.1,
            fill: 'tonexty',
            showlegend: false,
            type: 'scatter'
        });
    });
    
    const layout = {
        title: 'Monte Carlo Simulation Comparison',
        xaxis: {
            title: 'Months',
            showgrid: true
        },
        yaxis: {
            title: 'Portfolio Value (Starting at $1)',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1
        }
    };
    
    Plotly.newPlot('simulationComparison', traces, layout);
}

function displayPortfolioDetails(portfolioName, data) {
    // Create a new section for this portfolio
    const container = document.createElement('div');
    container.className = 'row mb-4';
    container.innerHTML = `
        <div class="col-12">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">${portfolioName} Details</h5>
                    <div id="statistics-${portfolioName}" class="statistics-container"></div>
                    <div class="row mt-4">
                        <div class="col-md-6">
                            <h6 class="text-primary">Historical Performance</h6>
                            <div id="historicalPlot-${portfolioName}" class="plot-container"></div>
                        </div>
                        <div class="col-md-6">
                            <h6 class="text-primary">Dividend Analysis</h6>
                            <div id="dividends-${portfolioName}" class="statistics-section"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.querySelector('.container').appendChild(container);
    
    // Display statistics for this portfolio
    displayStatistics(data.statistics, `statistics-${portfolioName}`);
    
    // Display historical performance
    displayHistoricalPerformance(data.metadata, data.statistics.portfolio, `historicalPlot-${portfolioName}`);
    
    // Display dividend analysis
    displayDividendAnalysis(data.dividend_analysis, data.simulation_results, `dividends-${portfolioName}`);
}

function displayStatistics(stats, containerId) {
    const container = document.getElementById(containerId);
    let html = container.innerHTML;
    
    // Portfolio statistics section
    html += `
        <div class="statistics-section">
            <h6 class="text-primary">Portfolio Statistics</h6>
            <div class="table-responsive">
                <table class="table table-sm statistics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Annual Return</td>
                            <td>${(stats.portfolio.annual_return * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>Annual Std Dev</td>
                            <td>${(stats.portfolio.annual_std_dev * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>${stats.portfolio.sharpe_ratio.toFixed(2)}</td>
                        </tr>
                        <tr>
                            <td>Total Return</td>
                            <td>${(stats.portfolio.total_return * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>Monthly Return Range</td>
                            <td>${(stats.portfolio.monthly_return_range[0] * 100).toFixed(2)}% to ${(stats.portfolio.monthly_return_range[1] * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>Median Monthly Return</td>
                            <td>${(stats.portfolio.median_monthly_return * 100).toFixed(2)}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    // Individual stocks section
    html += `
        <div class="statistics-section">
            <h6 class="text-primary">Individual Stock Statistics</h6>
            <div class="table-responsive">
                <table class="table table-sm statistics-table">
                    <thead>
                        <tr>
                            <th>Stock</th>
                            <th>Annual Return</th>
                            <th>Annual Std Dev</th>
                            <th>Sharpe Ratio</th>
                            <th>Total Return</th>
                            <th>Monthly Return Range</th>
                            <th>Median Monthly Return</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(stats.individual).map(([ticker, stat]) => `
                            <tr>
                                <td>${ticker}</td>
                                <td>${(stat.annual_return * 100).toFixed(2)}%</td>
                                <td>${(stat.annual_std_dev * 100).toFixed(2)}%</td>
                                <td>${stat.sharpe_ratio.toFixed(2)}</td>
                                <td>${(stat.total_return * 100).toFixed(2)}%</td>
                                <td>${(stat.monthly_return_range[0] * 100).toFixed(2)}% to ${(stat.monthly_return_range[1] * 100).toFixed(2)}%</td>
                                <td>${(stat.median_monthly_return * 100).toFixed(2)}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function displayHistoricalPerformance(metadata, portfolio, containerId) {
    // Plot cumulative performance
    const historicalTrace = {
        x: portfolio.historical_performance.dates,
        y: portfolio.historical_performance.values,
        type: 'scatter',
        mode: 'lines',
        name: 'Portfolio Value',
        line: {
            color: '#2c3e50',
            width: 2
        }
    };

    const historicalLayout = {
        title: 'Historical Portfolio Value (Starting at $1)',
        xaxis: {
            title: 'Date',
            showgrid: true
        },
        yaxis: {
            title: 'Portfolio Value',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1
        }
    };

    Plotly.newPlot(containerId, [historicalTrace], historicalLayout);

    // Plot returns distribution
    const returnsTrace = {
        x: portfolio.historical_performance.returns,
        type: 'histogram',
        name: 'Returns Distribution',
        marker: {
            color: '#3498db'
        },
        nbinsx: 30
    };

    const returnsLayout = {
        title: 'Monthly Returns Distribution',
        xaxis: {
            title: 'Monthly Return',
            showgrid: true
        },
        yaxis: {
            title: 'Frequency',
            showgrid: true
        },
        showlegend: false
    };

    Plotly.newPlot('returnsHistogramPlot', [returnsTrace], returnsLayout);
}

function displaySimulationPlot(results) {
    const timePoints = Array.from({length: results.percentiles['50th'].length}, (_, i) => i);
    
    const traces = [
        {
            name: '95th Percentile',
            y: results.percentiles['95th'],
            x: timePoints,
            line: {color: '#2ecc71'},
            type: 'scatter'
        },
        {
            name: '75th Percentile',
            y: results.percentiles['75th'],
            x: timePoints,
            line: {color: '#3498db'},
            type: 'scatter'
        },
        {
            name: 'Median',
            y: results.percentiles['50th'],
            x: timePoints,
            line: {color: '#2c3e50', width: 2},
            type: 'scatter'
        },
        {
            name: '25th Percentile',
            y: results.percentiles['25th'],
            x: timePoints,
            line: {color: '#3498db'},
            type: 'scatter'
        },
        {
            name: '5th Percentile',
            y: results.percentiles['5th'],
            x: timePoints,
            line: {color: '#2ecc71'},
            type: 'scatter'
        }
    ];
    
    const layout = {
        title: 'Monte Carlo Simulation Results',
        xaxis: {
            title: 'Months',
            showgrid: true
        },
        yaxis: {
            title: 'Portfolio Value (Starting at $1)',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1
        }
    };
    
    Plotly.newPlot('simulationPlot', traces, layout);
}

function displayConvergencePlot(convergence) {
    const data = convergence.convergence;
    
    const traces = [
        {
            name: 'Mean',
            x: data.map(d => d.num_simulations),
            y: data.map(d => d.mean),
            type: 'scatter',
            mode: 'lines+markers',
            line: {color: '#2c3e50'}
        },
        {
            name: 'Standard Deviation',
            x: data.map(d => d.num_simulations),
            y: data.map(d => d.std),
            type: 'scatter',
            mode: 'lines+markers',
            line: {color: '#e74c3c'}
        }
    ];
    
    const layout = {
        title: 'Convergence Analysis',
        xaxis: {
            title: 'Number of Simulations',
            showgrid: true
        },
        yaxis: {
            title: 'Value',
            showgrid: true
        },
        showlegend: true,
        legend: {
            x: 0,
            y: 1
        }
    };
    
    Plotly.newPlot('convergencePlot', traces, layout);
}

function displayDividendAnalysis(dividendData, simulationResults, containerId) {
    // Display portfolio dividend metrics
    const statsContainer = document.getElementById('dividendStats');
    statsContainer.innerHTML = `
        <div class="table-responsive">
            <table class="table table-sm statistics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Current</th>
                        <th>Projected</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Portfolio Yield</td>
                        <td>${(dividendData.current_yield * 100).toFixed(2)}%</td>
                        <td>${(dividendData.current_yield * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Annual Income (per $1 invested)</td>
                        <td>$${dividendData.current_annual_income.toFixed(3)}</td>
                        <td>$${dividendData.projected_annual_income.toFixed(3)}</td>
                    </tr>
                    <tr>
                        <td>Dividend Growth Rate</td>
                        <td>${(dividendData.weighted_growth_rate * 100).toFixed(2)}%</td>
                        <td>Compound Annual</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // Display individual stock dividend metrics
    const individualContainer = document.getElementById('individualDividends');
    let individualHtml = `
        <div class="table-responsive">
            <table class="table table-sm statistics-table">
                <thead>
                    <tr>
                        <th>Stock</th>
                        <th>Dividend Yield</th>
                        <th>Growth Rate</th>
                        <th>Annual Income (per $1)</th>
                    </tr>
                </thead>
                <tbody>
    `;

    Object.entries(dividendData.individual_metrics).forEach(([ticker, metrics]) => {
        individualHtml += `
            <tr>
                <td>${ticker}</td>
                <td>${(metrics.dividend_yield * 100).toFixed(2)}%</td>
                <td>${(metrics.dividend_growth * 100).toFixed(2)}%</td>
                <td>$${(metrics.dividend_yield).toFixed(3)}</td>
            </tr>
        `;
    });

    individualHtml += `
                </tbody>
            </table>
        </div>
    `;
    individualContainer.innerHTML = individualHtml;
}

function displayOptimizationResults(optimization) {
    // Add pie chart for optimal weights
    const pieTrace = {
        values: optimization.optimal_weights,
        labels: optimization.tickers,
        type: 'pie',
        name: 'Portfolio Allocation',
        textinfo: 'label+percent',
        hoverinfo: 'label+percent+value',
        hole: 0.4
    };

    const pieLayout = {
        title: 'Optimal Portfolio Allocation',
        showlegend: true,
        legend: {
            x: 0,
            y: 1
        },
        height: 400
    };

    // Create a new div for the pie chart if it doesn't exist
    let pieChartDiv = document.getElementById('allocationPieChart');
    if (!pieChartDiv) {
        pieChartDiv = document.createElement('div');
        pieChartDiv.id = 'allocationPieChart';
        pieChartDiv.className = 'plot-container';
        document.getElementById('rebalanceStats').parentElement.insertBefore(pieChartDiv, document.getElementById('rebalanceStats'));
    }

    Plotly.newPlot('allocationPieChart', [pieTrace], pieLayout);

    // Display metrics table
    const metricsContainer = document.getElementById('rebalanceStats');
    metricsContainer.innerHTML = `
        <div class="table-responsive">
            <table class="table table-sm statistics-table">
                <tbody>
                    <tr>
                        <td>Expected Annual Return</td>
                        <td>${(optimization.metrics.expected_annual_return * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Annual Volatility</td>
                        <td>${(optimization.metrics.annual_volatility * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>${optimization.metrics.sharpe_ratio.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Sortino Ratio</td>
                        <td>${optimization.metrics.sortino_ratio.toFixed(2)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // Add risk contribution display for risk parity
    if (optimization.metrics.risk_contributions) {
        metricsContainer.innerHTML += `
            <div class="mt-3">
                <h6 class="text-primary">Risk Contributions</h6>
                <div class="table-responsive">
                    <table class="table table-sm statistics-table">
                        <thead>
                            <tr>
                                <th>Stock</th>
                                <th>Risk Contribution</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${optimization.tickers.map((ticker, i) => `
                                <tr>
                                    <td>${ticker}</td>
                                    <td>${(optimization.metrics.risk_contributions[i] * 100).toFixed(2)}%</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // Display rebalancing analysis
    displayRebalancingAnalysis(optimization.rebalance_analysis);
}

function displayRebalancingAnalysis(rebalanceData) {
    // Plot historical performance with rebalancing
    const performanceTrace = {
        x: rebalanceData.dates,
        y: rebalanceData.portfolio_values,
        type: 'scatter',
        mode: 'lines',
        name: 'Portfolio Value',
        line: {
            color: '#2563eb',
            width: 2
        }
    };

    // Add markers for rebalancing points
    const rebalancePoints = {
        x: rebalanceData.rebalance_dates,
        y: rebalanceData.portfolio_values.filter((_, i) => 
            rebalanceData.rebalance_dates.includes(rebalanceData.dates[i])),
        type: 'scatter',
        mode: 'markers',
        name: 'Rebalancing Points',
        marker: {
            color: '#e11d48',
            size: 8,
            symbol: 'diamond'
        }
    };

    const layout = {
        title: 'Portfolio Value with Rebalancing',
        xaxis: {
            title: 'Date',
            showgrid: true
        },
        yaxis: {
            title: 'Portfolio Value',
            showgrid: true
        },
        showlegend: true
    };

    Plotly.newPlot('rebalancePlot', [performanceTrace, rebalancePoints], layout);

    // Display rebalancing statistics and weight changes
    const statsContainer = document.getElementById('rebalanceStats');
    
    // Summary statistics
    let html = `
        <div class="table-responsive mb-4">
            <table class="table table-sm statistics-table">
                <tbody>
                    <tr>
                        <td>Final Portfolio Value</td>
                        <td>$${rebalanceData.final_value.toFixed(2)}</td>
                    </tr>
                    <tr>
                        <td>Total Turnover</td>
                        <td>${(rebalanceData.total_turnover * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Total Transaction Costs</td>
                        <td>${(rebalanceData.total_cost * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Number of Rebalances</td>
                        <td>${rebalanceData.rebalance_dates.length}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // Detailed rebalancing history
    html += `
        <h6 class="text-primary">Rebalancing History</h6>
        <div class="accordion" id="rebalancingAccordion">
    `;

    rebalanceData.turnover_history.forEach((rebalance, index) => {
        const accordionId = `rebalance-${index}`;
        
        // Calculate significant changes (more than 1% absolute change)
        const significantChanges = rebalance.weight_changes.filter(change => 
            Math.abs(change.absolute_change) >= 0.01
        );
        
        // Summary of major changes for the accordion header
        const summaryText = significantChanges.length > 0 
            ? `Major changes: ${significantChanges.slice(0, 2)
                .map(change => `${change.ticker} ${change.direction === 'up' ? '↑' : '↓'} ${Math.abs(change.percent_change).toFixed(1)}%`)
                .join(', ')}${significantChanges.length > 2 ? '...' : ''}`
            : 'Minor rebalancing adjustments';

        html += `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading-${accordionId}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse-${accordionId}">
                        <strong>${rebalance.date}</strong> &nbsp;|&nbsp; 
                        Turnover: ${(rebalance.turnover * 100).toFixed(2)}% &nbsp;|&nbsp; 
                        ${summaryText}
                    </button>
                </h2>
                <div id="collapse-${accordionId}" class="accordion-collapse collapse" 
                     data-bs-parent="#rebalancingAccordion">
                    <div class="accordion-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Stock</th>
                                        <th>Old Weight</th>
                                        <th>New Weight</th>
                                        <th>Change</th>
                                        <th>% Change</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${rebalance.weight_changes.map(change => `
                                        <tr class="${Math.abs(change.absolute_change) >= 0.01 ? 'table-active' : ''}">
                                            <td>${change.ticker}</td>
                                            <td>${(change.old_weight * 100).toFixed(2)}%</td>
                                            <td>${(change.new_weight * 100).toFixed(2)}%</td>
                                            <td>
                                                <span class="text-${change.direction === 'up' ? 'success' : change.direction === 'down' ? 'danger' : 'secondary'}">
                                                    ${change.direction === 'up' ? '↑' : change.direction === 'down' ? '↓' : '−'}
                                                    ${Math.abs(change.absolute_change * 100).toFixed(2)}%
                                                </span>
                                            </td>
                                            <td>
                                                <span class="text-${change.direction === 'up' ? 'success' : change.direction === 'down' ? 'danger' : 'secondary'}">
                                                    ${change.direction === 'up' ? '+' : change.direction === 'down' ? '-' : ''}
                                                    ${Math.abs(change.percent_change).toFixed(2)}%
                                                </span>
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    statsContainer.innerHTML = html;
} 