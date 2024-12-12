import numpy as np
import pandas as pd
import torch
import random
import talib as ta
import copy
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig
import quantstats as qs  
import warnings
import logging
import yfinance as yf

# Define the portfolio optimization Agent class
class Agent(torch.nn.Module):
    def __init__(self, base_model, input_dim, hidden_dim=256):
        super().__init__()
        self.llm = base_model
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)  # Project indicators to model hidden size
        self.output_layer = torch.nn.Linear(hidden_dim, 1)  # Final layer to output a single score per stock

    def forward(self, inputs_embeds):
        # Project input embeddings
        x = self.input_layer(inputs_embeds)  # Shape: (batch_size, seq_length, hidden_dim)

        # Forward pass through the LLM encoder
        x = self.llm(inputs_embeds=x)

        # Use the hidden state of the last time step for each stock
        x = x.last_hidden_state[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Pass through the final output layer
        x = self.output_layer(x).squeeze(-1)  # Shape: (batch_size,)

        # Scale outputs to create a centered exponential function
        x_tanh = torch.tanh(x)

        # Softmax to compute portfolio weights
        x_softmax = torch.nn.functional.softmax(x, dim=-1).unsqueeze(0)

        # Compute Tanh Softmax outputs
        x = x_tanh * x_softmax

        # Normalize outputs so the absolute sum equals 1
        abs_sum = torch.sum(torch.abs(x), dim=-1, keepdim=True)  # Compute absolute sum per batch
        portfolio_weights = x / abs_sum  # Normalize each element

        return portfolio_weights



class Portfolio_Optimization_Strategy():
    def __init__(self):
    
        # Initialize price-based factor names
        self.factor_names = [
            'Daily Return', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'ret_exc', 'ret_exc_3l', 'mkt_exc_3l',
            'mktrf_ld1', 'mktrf_lg1', 'dollar_volume', 'rvol', 'rmax1', 'rmax5', 'rskew', 'beta', 'ivol_capm',
            'iskew_capm', 'coskew', 'ivol_ff3', 'iskew_ff3', 'ivol_hxz4', 'iskew_hxz4', 'beta_dimson', 'amihud',
            'cv_dollar_volume', 'price_to_high', 'rmax5_rvol_21d', 'corr_1260d', 'betabab_1260d',
            'ivol_capm_252d', 'ret_12_1', 'ret_12_7', 'ret_3_1', 'ret_6_1', 'ret_9_1', 'ret_1_0', 'seas_1yr'
        ]

        # Total input features
        self.input_dim = len(self.factor_names)

        # Initialize variables
        self.indicators_data = {}
        self.portfolio_weights = None
        self.current_BOM_date = None
        self.current_portfolio = []
        self.returns = []

        # Initialize Chronos-T5 LLM from cache
        model_path = "/home/ryengel/.cache/huggingface/hub/models--amazon--chronos-bolt-tiny/snapshots/f6ff2d2ba9168d498c015bc8dd07e3b395b31b3f"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).get_encoder()

        # Freeze the base model's weights
        for param in model.parameters():
            param.requires_grad = False

        # Initialize LoRA Config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["SelfAttention.q", "SelfAttention.k", "SelfAttention.v", "SelfAttention.o"],
            lora_dropout=0.3,
            bias="all"
        )
        self.lora_model = get_peft_model(model, lora_config)

        # Initialize data
        self.full_data = df.copy()


    def get_data(self, date, df):
        # Get all of the data for each stock on a given day
        current_data = df[df['Date'] == date]

        # Filter to only include stocks with non-NaN 'Daily Return'
        current_data = current_data[~current_data['Daily Return'].isna()]

        return current_data


    def backtest(self, dataframe):
        # Looping through all t years of the dataframe
        years = sorted(dataframe['Date'].dt.year.unique())
        evaluation_count = 0  # Counter for evaluation years

        for y in years[:-1]:
            # Reinitialize variables each year
            self.indicators_data = {}
            self.portfolio_weights = None
            self.current_BOM_date = None
            self.current_portfolio = []

            # Initialize a new Agent each year
            independent_lora_model = copy.deepcopy(self.lora_model)  
            self.agent = Agent(independent_lora_model, input_dim=self.input_dim).cuda()
            self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=1e-1)

            # Train up to year y (with either expanding window if <3 years, else rolling 3-year window)
            self.train(dataframe, epochs=8, current_year=y, start_year=years[0], window_length=2)

            # Evaluate on year y
            self.evaluate(y)
            evaluation_count += 1

            # Stop after n evaluation periods if needed
            n = 25
            if evaluation_count >= n:
                print(f"Stopping backtest after {n} years of evaluation.")
                break


    def train(self, df, epochs, current_year, start_year, window_length):
        # Determine training data window
        if (current_year - start_year + 1) < window_length:
            # Use expanding window from start_year to current_year
            training_data = df[df['Date'].dt.year <= current_year]
        else:
            # Use rolling window of last window_length years
            training_data = df[(df['Date'].dt.year > current_year - window_length) & (df['Date'].dt.year <= current_year)]

        for n in range(epochs):
            print(f"Training Year {current_year}, Epoch: {n}")

            training_data['Year'] = training_data['Date'].dt.year
            training_data['Month'] = training_data['Date'].dt.month
            BOM_dates = set(training_data.groupby(['Year', 'Month'])['Date'].min())
            EOM_dates = set(training_data.groupby(['Year', 'Month'])['Date'].max())

            training_dates = sorted(training_data['Date'].unique())
            training_dates = pd.to_datetime(training_dates)

            # Initialize variables to hold portfolio at BOM
            portfolio_ids = None
            weights = None
            start_date = None

            for day in training_dates:
                if day in BOM_dates:
                    # Get this day's data
                    data = self.get_data(day, training_data)
                    self.current_portfolio = data['Id'].tolist()

                    # Calculate indicators and generate weights
                    self.calculate_indicators(data)
                    self.generate_weights()

                    portfolio_ids = self.current_portfolio
                    weights = self.portfolio_weights
                    start_date = day

                elif day in EOM_dates:
                    # Calculate returns of the previous portfolio
                    if portfolio_ids is not None and weights is not None:
                        end_date = day
                        wts, rets = self.calculate_portfolio_returns(
                            portfolio_ids, weights, start_date, end_date
                        )
                        if wts is not None and rets is not None:
                            loss = self.loss_function(wts, rets)
                            print(f"loss: {loss}")
                            self.backpropagate_loss(loss)
                    else:
                        print(f"Warning: No portfolio to calculate returns for EOM date {day}")


    def evaluate(self, current_year):
        # Evaluate the model on current_year+1
        print(f"Evaluating year {current_year+1}")

        test_data = self.full_data[self.full_data['Date'].dt.year == current_year + 1]
        if test_data.empty:
            print(f"No test data available for year {current_year+1}. Skipping evaluation.")
            return

        test_data['Year'] = test_data['Date'].dt.year
        test_data['Month'] = test_data['Date'].dt.month
        BOM_dates = set(test_data.groupby(['Year', 'Month'])['Date'].min())
        EOM_dates = set(test_data.groupby(['Year', 'Month'])['Date'].max())

        test_dates = sorted(test_data['Date'].unique())
        test_dates = pd.to_datetime(test_dates)

        portfolio_ids = None
        weights = None
        start_date = None

        for day in test_dates:
            if day in BOM_dates:
                data = self.get_data(day, test_data)
                self.current_portfolio = data['Id'].tolist()

                self.calculate_indicators(data)
                self.generate_weights()

                portfolio_ids = self.current_portfolio
                weights = self.portfolio_weights
                start_date = day

            elif day in EOM_dates:
                if portfolio_ids is not None and weights is not None:
                    end_date = day
                    weights, returns = self.calculate_portfolio_returns(
                        portfolio_ids, weights, start_date, end_date
                    )
                    if weights is not None and returns is not None:
                        strategy_return = torch.dot(weights, returns).item()
                        self.returns.append({'Date': end_date, 'Return': strategy_return})
                else:
                    print(f"Warning: No portfolio to calculate returns for EOM date {day}")


    def calculate_indicators(self, current_data):
        self.indicators_data = {}
        stock_ids = current_data['Id'].unique()

        for stock_id in stock_ids:
            stock_data = self.full_data[self.full_data['Id'] == stock_id].sort_values('Date').reset_index(drop=True)
            current_date = current_data[current_data['Id'] == stock_id]['Date'].iloc[0]
            date_index = stock_data[stock_data['Date'] == current_date].index[0]

            past_21_index = date_index - 20
            if past_21_index >= 0:
                past_21_data = stock_data.iloc[past_21_index:date_index + 1]
            else:
                pad_length = 21 - (date_index + 1)
                pad_data = pd.DataFrame({col: [np.nan] * pad_length for col in stock_data.columns})
                past_21_data = pd.concat([pad_data, stock_data.iloc[0:date_index + 1]], ignore_index=True)

            factors = {}
            for factor_name in self.factor_names:
                if factor_name in past_21_data.columns:
                    factor_values = past_21_data[factor_name].values
                else:
                    factor_values = np.full(21, np.nan)
                factors[factor_name] = factor_values

            self.indicators_data[stock_id] = {'factors': factors}


    def generate_weights(self):
        if not self.current_portfolio:
            print("Error: Current portfolio is empty. Cannot generate weights.")
            return

        input_data = []

        for stock_id in self.current_portfolio:
            data = self.indicators_data.get(stock_id, {})
            factors = data.get('factors', {})

            factors_series = np.array(
                [factors.get(name, np.full(21, np.nan)) for name in self.factor_names]
            )
            if np.isnan(factors_series).any():
                factors_series = np.nan_to_num(factors_series, nan=0.0, posinf=1e6, neginf=-1e6)

            input_data.append(factors_series)

        if not input_data:
            print("Error: No valid data in input_data. Check indicators and portfolio setup.")
            return

        try:
            input_data = np.stack(input_data, axis=0)  # (num_stocks, num_factors, time_series_length)
        except ValueError as e:
            print(f"Error stacking input_data: {e}")
            print(f"Check individual shapes: {[np.shape(item) for item in input_data]}")
            raise

        input_data = np.transpose(input_data, (0, 2, 1))  # (num_stocks, time_series_length, num_factors)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).cuda()

        self.portfolio_weights = self.agent(inputs_embeds=input_tensor)
        print(f"Portfolio Weights: {self.portfolio_weights}") 


    def calculate_portfolio_returns(self, portfolio_ids, weights, start_date, end_date):
        if not portfolio_ids or weights is None:
            print("Warning: Portfolio is empty or weights are None. Skipping return calculation.")
            return None, None

        returns = []
        matching_weights = []

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
            stock_weight = weights[0, i]
            matching_weights.append(stock_weight)

        device = weights.device
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        weights_tensor = torch.stack(matching_weights).to(device)

        return weights_tensor, returns_tensor


    def loss_function(self, weights, returns):
        # Ensure loss is on same device as weights
        device = weights.device

        # Compute expected return: w^T * returns
        expected_return = torch.dot(weights, returns)

        # Compute entropy regularization term: η * w^T * w
        entropy = -torch.dot(weights, weights)

        # Set η (eta) for regularization strength
        eta = 1e-2

        # Compute total loss
        loss = (-expected_return + eta * entropy).to(device)

        return loss


    def backpropagate_loss(self, loss):
        self.optimizer.zero_grad()  
        loss.backward() 
        self.optimizer.step()


if __name__ == '__main__':
    # Silence Warnings
    warnings.filterwarnings("ignore", module="matplotlib.font_manager")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

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

    df = pd.read_csv('../data/factors_v3.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Run backtest
    strategy = Portfolio_Optimization_Strategy()
    strategy.backtest(df)
    print("Completed Backtest.")

    # Convert strategy returns to a pandas Series with datetime index
    results_df = pd.DataFrame(strategy.returns)
    results_df = results_df.sort_values('Date').set_index('Date')
    results_df.index = pd.to_datetime(results_df.index, utc=False)  # ensure timezone-naive
    returns_series = results_df['Return'].fillna(0)  # ensure no NaNs in strategy returns
    returns_series.index = returns_series.index.tz_localize(None)   # remove any timezone info

    # Download SPY data (daily)
    spy_data = yf.download("SPY", start="1999-01-01", end="2024-07-01", interval='1d')
    spy_data = spy_data['Close'].dropna()
    spy_data.index = spy_data.index.tz_localize(None)  # ensure timezone-naive

    # Get the intersection of dates
    common_dates = returns_series.index.intersection(spy_data.index)

    # Manually add this date: 1999-12-31 to common dates, this way we can calculate the percent change for later
    # If we don't do this then we don't get a pct change for the first date
    common_dates = common_dates.insert(0, pd.Timestamp('1999-12-31'))

    # Restrict SPY data to only those common dates
    spy_data = spy_data.loc[common_dates].sort_index()

    # Compute pct change of SPY only for the matching dates
    # (Note: pct_change will lose the first date as there's no previous day to compare)
    spy_returns = spy_data.pct_change().dropna()
    spy_returns.name = "SPY"

    # Align strategy returns to the dates that remain after pct_change
    common_dates_after_pchange = returns_series.index.intersection(spy_returns.index)
    returns_series = returns_series.loc[common_dates_after_pchange]

    # Generate quantstats HTML report with SPY as benchmark
    qs.reports.html(returns_series, benchmark=spy_returns, output='../results/long_short_backtest.html')

    # Align SPY returns to the strategy returns index
    aligned_spy_returns = spy_returns.reindex(returns_series.index)

    # Create the DataFrame by appending SPY returns as a new column
    output_df = pd.DataFrame({'Strategy_Returns': returns_series})
    output_df['SPY_Returns'] = aligned_spy_returns

    # Ensure the index is named "Date" for clarity
    output_df.index.name = 'Date'

    # Save to CSV
    output_csv_path = '../results/long_short_backtest.csv'
    output_df.to_csv(output_csv_path)
