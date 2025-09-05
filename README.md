# Optimal Trade Execution using Machine Learning-Driven Impact Modeling

This project develops an algorithmic trading strategy to minimize **market impact costs** for large stock orders.  
The system uses a **Gradient Boosting model**, trained on **high-frequency limit order book (LOB)** data, to power an **adaptive execution algorithm** that intelligently schedules trades based on real-time market conditions.

---

## 📌 Project Overview

Executing a large stock order in a single trade can significantly move the price, leading to high transaction costs (a phenomenon known as **temporary market impact**).  

The goal of an **optimal execution algorithm** is to break this large **parent order** into smaller **child orders** and execute them over time to minimize this impact.

This project tackles the problem in two main stages:

1. **Impact Modeling**  
   Build a machine learning model to predict the price impact of a trade, given the current state of the market and the desired trade size.

2. **Execution Algorithm**  
   Use the model's predictions to power an adaptive algorithm that decides how many shares to buy each minute, concentrating trades in periods of high liquidity.

---

## ⚙️ Project Logic & Details

The project is implemented as a **Python script** that simulates the workflow, from model training to executing a **50,000-share order** over a **390-minute trading day**.

---

### Step 1: Feature Engineering & Dataset Creation

The foundation of the system is a robust dataset that captures how market conditions relate to trade impact.

- **Data Source**:  
  High-frequency CSV files containing **10 levels of LOB data** for three tickers:
  - **SOUN** (high liquidity)  
  - **FROG** (medium liquidity)  
  - **CRWV** (illiquid)

- **Ground Truth Calculation**:  
  For each historical LOB snapshot, we simulate executing trades of various sizes (50, 100, 200, 500, 1,000, 2,000, 5,000 shares) by *walking the book* on the ask side.  
  - The resulting slippage, normalized by the mid-price, becomes the **true impact** and the model’s **target variable**.

- **Feature Creation**:
  - `order_size`: proposed trade size  
  - `spread`: best ask – best bid  
  - `liquidity_imbalance`:  
    \[
    \frac{\text{bid\_liquidity} - \text{ask\_liquidity}}{\text{bid\_liquidity} + \text{ask\_liquidity}}
    \]  
  - `full_book_depth`: prices and sizes of all 10 bid and ask levels

- **Time-Series Handling**:  
  The dataset uses **time-based splitting (80% train, 20% test)** to respect temporal order and prevent look-ahead bias.

---

### Step 2: Training the Gradient Boosting Impact Model

We use a **Gradient Boosting model (LightGBM)** to predict temporary market impact, chosen because simpler parametric models (e.g., the **square root model**) were unreliable across different stocks.

- **Training**:  
  Separate models are trained for each ticker using **time-series-aware splits**.  
  The model learns **non-linear relationships** between market features and price impact.

- **Performance**:

| Stock | Liquidity | R²     | MAE       |
|-------|-----------|--------|-----------|
| SOUN  | High      | 0.5043 | 0.000053  |
| FROG  | Medium    | 0.6022 | 0.000266  |
| CRWV  | Illiquid  | 0.4847 | 0.000573  |

➡️ These results reflect realistic performance across different liquidity environments.  
➡️ The model **outperforms** the square root model (R² ~0.21 for FROG, ~0.33 for CRWV, **negative for SOUN**).

---

### Step 3: The Adaptive Execution Algorithm

This is the **core logic** that determines how many shares \(x_i\) to buy in each minute \(i\) to minimize total cost:

\[
\min_{C(X)} = \sum_{i=1}^{N} x_i \cdot \text{GBM\_predict}(\text{features}_i, x_i)
\]

**Workflow:**
1. **Establish Baseline**: Calculate average predicted impact for a standard trade size across the day.  
2. **Calculate Target Rate**:  
   \[
   \text{shares/min} = \frac{\text{remaining\_shares}}{\text{minutes\_left}}
   \]  
3. **Assess Market**: Predict impact for a standard trade size using current LOB state.  
4. **Adapt Aggressively**:
   - If current impact < average → **increase trading rate** (high liquidity).  
   - If current impact > average → **reduce trading rate** (save shares for later).  

**Safety Checks:**
- Cap on maximum shares per minute.  
- **Panic mode** to ensure completion by day’s end.  

**Highlights:**
- Exploits **intraday liquidity patterns** (e.g., tighter spreads near close for FROG).  
- Preliminary results suggest **up to 27% cost savings** vs. TWAP for illiquid securities like CRWV.

---

## 📊 Conclusion

This project demonstrates a practical, **machine learning-driven approach** to optimal trade execution.

- **Gradient Boosting model** captures **non-linear market impact dynamics**, outperforming traditional models.  
- **Adaptive execution algorithm** leverages predictions to make **real-time decisions** across liquidity profiles.  

  
