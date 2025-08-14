# Optimal-Trade-Execution-using-Machine-Learning-Driven-Impact-Modeling
# Optimal Trade Execution via ML-Driven Impact Modeling

This project develops a sophisticated algorithmic trading strategy to minimize the market impact costs of large stock orders. The system uses a Gradient Boosting model, trained on high-frequency limit order book data, to power an adaptive execution algorithm that intelligently schedules trades based on real-time market conditions.

## Project Overview

Executing a large stock order in a single trade can significantly move the price, leading to high transaction costs (a phenomenon known as "temporary market impact"). The goal of an optimal execution algorithm is to break this large "parent" order into smaller "child" orders and execute them over time to minimize this impact.

This project tackles the problem in two main stages:

1. **Impact Modeling**: First, we build a highly accurate machine learning model to predict the price impact of a trade, given the current state of the market and the desired trade size.
2. **Execution Algorithm**: Second, we use the model's predictions to power an adaptive algorithm that decides how many shares to buy each minute, aiming to concentrate trades in periods of high liquidity (low cost).

The final system demonstrates a significant performance improvement over a standard Time-Weighted Average Price (TWAP) benchmark, achieving cost savings of over 27% for an illiquid security.

## Project Logic & Details

The project is implemented as a Python script that simulates the entire workflow, from model training to executing a 50,000-share order over a 390-minute trading day.

### Step 1: Feature Engineering & Dataset Creation

The foundation of the system is a robust dataset that teaches the model how market conditions relate to trade impact.

- **Data Source**: The model uses high-frequency CSV files containing 10 levels of limit order book data for three tickers with different liquidity profiles (`SOUN`, `FROG`, `CRWV`).
- **"Ground Truth" Calculation**: For each historical snapshot of the order book, we simulate executing trades of various sizes (e.g., 50, 100, 500... shares) by "walking the book." The resulting slippage from this simulation becomes the "true" impact and our model's target variable.
- **Feature Creation**: For each simulated trade, we create a rich feature set that describes the market at that moment, including:
  - `order_size`: The size of the proposed trade.
  - `spread`: The difference between the best bid and ask prices.
  - `liquidity_imbalance`: A measure of whether there is more buying or selling pressure in the book.
  - **Full Book Depth**: The prices and sizes of all 10 bid and ask levels.

### Step 2: Training the Gradient Boosting Impact Model

We use a **Gradient Boosting** model (specifically, LightGBM) to predict the temporary market impact. This model was chosen after analysis showed that simpler parametric models (like the Square Root Impact model) were unreliable across different stocks.

- **Training**: For each ticker, a separate model is trained on its feature-engineered dataset. The model learns the complex, non-linear relationships between the market features and the resulting price impact.
- **Performance**: The models achieve exceptionally high predictive accuracy on unseen test data, with R-squared values of **0.99 (SOUN), 0.95 (FROG), and 0.89 (CRWV)**.

### Step 3: The Adaptive Execution Algorithm

This is the core logic that determines how many shares, $x_i$, to buy in each minute, $i$.

1. **Establish a Baseline**: Before the simulation starts, the algorithm calculates the *average* predicted impact for a standard trade size across the entire day's data. This serves as its benchmark for "normal" market conditions.
2. **Calculate Target Rate**: At each minute, it knows the simple average number of shares it needs to trade to finish on time (e.g., `remaining_shares / minutes_left`).
3. **Assess Live Market**: It uses the trained model to predict the impact for a standard trade size based on the *current* market state.
4. **Adapt Aggressively**: It compares the current predicted impact to the daily average.
   - If the current impact is **lower** than average (a good time to trade), it increases its trading rate.
   - If the current impact is **higher** than average (a bad time to trade), it reduces its trading rate, saving shares for later.
5. **Safety Checks**: The algorithm includes risk limits, such as a cap on the maximum shares per minute and a "panic mode" to ensure completion near the end of the day.

## How to Use the Code

The project is contained in a single Python script.

1. **Setup**: Place the ticker CSV files (`SOUN_*.csv`, `FROG_*.csv`, `CRWV_*.csv`) in the same directory as the script.
2. **Run Question 1 Analysis**: The first part of the script (`if __name__ == '__main__':` block in the first code cell) will automatically run the model training and validation for all three tickers. This will output the performance metrics and feature importance plots for each.
3. **Run Question 2 Simulation**: The second part of the script (in the second code cell) will use the trained models to run the execution simulation. You can change the `ticker_to_simulate` variable to see how the algorithm performs for different stocks. The output will be a table of performance metrics and a bar chart showing the final execution schedule.
