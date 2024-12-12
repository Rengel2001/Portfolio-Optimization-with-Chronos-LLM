import numpy as np
import pandas as pd
import torch
import random
import quantstats as qs
import warnings
import logging
import yfinance as yf

class EqualWeightStrategy:
    def __init__(self, df):
        self.full_data = df.copy()
        self.full_data['Year'] = self.full_data['Date'].dt.year
        self.full_data['Month'] = self.full_data['Date'].dt.month
        self.returns = []  # Will store {'Date': date, 'Return': float}
    
    def get_data(self, date):
        current_data = self.full_data[self.full_data['Date'] == date]
        # Filter out stocks that don't have a valid Daily Return
        current_data = current_data[~current_data['Daily Return'].isna()]
        return current_data
    
    def calculate_portfolio_returns(self, portfolio_ids, weights, start_date, end_date):
        # Calculate the return from start_date to end_date for each stock and apply weights
        returns = []
        for i, stock_id in enumerate(portfolio_ids):
            stock_data = self.full_data[
                (self.full_data['Id'] == stock_id) &
                (self.full_data['Date'] >= start_date) &
                (self.full_data['Date'] <= end_date)
            ].sort_values('Date')

            if len(stock_data) == 0:
                stock_return = 0.0
            else:
                daily_returns = stock_data['Daily Return'].dropna()
                if len(daily_returns) == 0:
                    stock_return = 0.0
                else:
                    stock_return = np.prod((1 + daily_returns.values)) - 1
                    if np.isnan(stock_return) or np.isinf(stock_return):
                        stock_return = 0.0
            returns.append(stock_return)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        portfolio_return = torch.dot(weights_tensor, returns_tensor).item()
        return portfolio_return

    def backtest(self):
        years = sorted(self.full_data['Year'].unique())

        for y in years:
            year_data = self.full_data[self.full_data['Year'] == y]
            if year_data.empty:
                continue
            
            BOM_dates = set(year_data.groupby(['Year', 'Month'])['Date'].min())
            EOM_dates = set(year_data.groupby(['Year', 'Month'])['Date'].max())
            trading_dates = sorted(year_data['Date'].unique())

            portfolio_ids = None
            weights = None
            start_date = None

            for day in trading_dates:
                if day in BOM_dates:
                    data = self.get_data(day)
                    portfolio_ids = data['Id'].tolist()
                    if len(portfolio_ids) > 0:
                        # Equal weights
                        weights = [1/len(portfolio_ids)] * len(portfolio_ids)
                        start_date = day

                elif day in EOM_dates and portfolio_ids is not None and weights is not None:
                    # Compute EOM returns
                    end_date = day
                    portfolio_return = self.calculate_portfolio_returns(portfolio_ids, weights, start_date, end_date)
                    self.returns.append({'Date': end_date, 'Return': portfolio_return})

        # Convert returns to Pandas Series
        results_df = pd.DataFrame(self.returns).sort_values('Date').set_index('Date')
        results_df.index = pd.to_datetime(results_df.index, utc=False)
        returns_series = results_df['Return'].fillna(0)
        returns_series.index = returns_series.index.tz_localize(None)
        return returns_series


if __name__ == '__main__':
    # Silence warnings/logging
    warnings.filterwarnings("ignore")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.disable(logging.CRITICAL)

    # Set seeds for reproducibility (optional)
    def set_random_seeds(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    print("Setting Random Seeds for Reproducibility...")
    set_random_seeds(11)

    # Load your factor data (even though we won't use factors, we still need Daily Return)
    df = pd.read_csv('../data/factors_v3.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Run equal-weight backtest
    strategy = EqualWeightStrategy(df)
    returns_series = strategy.backtest()
    print("Completed Equal-Weight Backtest.")

    # Download SPY data for comparison
    spy_data = yf.download("SPY", start="1999-01-01", end="2024-07-01", interval='1d')
    spy_data = spy_data['Close'].dropna()
    spy_data.index = spy_data.index.tz_localize(None)

    # Align dates and compute SPY returns
    common_dates = returns_series.index.intersection(spy_data.index)
    # Insert a date at the start if needed to align pct_change computation
    common_dates = common_dates.insert(0, pd.Timestamp('1999-12-31'))
    spy_data = spy_data.loc[common_dates].sort_index()
    spy_returns = spy_data.pct_change().dropna()
    spy_returns.name = "SPY"

    # Align strategy returns and SPY returns
    common_dates_after_pchange = returns_series.index.intersection(spy_returns.index)
    returns_series = returns_series.loc[common_dates_after_pchange]
    spy_returns = spy_returns.loc[common_dates_after_pchange]
    returns_series.name = "EWP"

    # Generate quantstats HTML report
    qs.reports.html(returns_series, benchmark=spy_returns, output='../results/equal_weight_vs_spy.html')

    # Save to CSV
    output_df = pd.DataFrame({'Equal_Weight_Returns': returns_series, 'SPY_Returns': spy_returns})
    output_df.index.name = 'Date'
    output_df.to_csv('../results/equal_weight_vs_spy.csv')
