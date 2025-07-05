import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Load combined data
folder = 'stock_data/'
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

dataframes = []
for file in csv_files:
    try:
        df = pd.read_csv(os.path.join(folder, file))
        df['Ticker'] = file.replace('.csv', '')
        df['Date'] = pd.to_datetime(df['Date'])
        dataframes.append(df)
    except Exception as e:
        st.error(f"Error loading {file}: {e}")

if not dataframes:
    st.error("No data loaded.")
    st.stop()

combined_df = pd.concat(dataframes)

# Precompute SMAs for user convenience
combined_df['SMA20'] = combined_df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
combined_df['SMA50'] = combined_df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())

# Sidebar - User Inputs
st.sidebar.title("Stock Dashboard with Python Strategy Testing")

tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=combined_df['Ticker'].unique().tolist(),
    default=[combined_df['Ticker'].unique()[0]]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

normalize = st.sidebar.checkbox("Normalize Prices", value=False)
show_volume = st.sidebar.checkbox("Show Volume Plot", value=False)

# Filter data based on user inputs
filtered_data = combined_df[
    (combined_df['Ticker'].isin(tickers)) &
    (combined_df['Date'] >= pd.to_datetime(start_date)) &
    (combined_df['Date'] <= pd.to_datetime(end_date))
].copy()

# Normalize prices if requested
if normalize:
    for ticker in tickers:
        mask = filtered_data['Ticker'] == ticker
        initial = filtered_data.loc[mask, 'Close'].iloc[0]
        filtered_data.loc[mask, 'Close'] = filtered_data.loc[mask, 'Close'] / initial

# --- Custom Python Strategy Code ---
st.subheader("🧠 Write Your Own Python Strategy")
def_strategy = '''
# Define your strategy here.
# df is the DataFrame containing your selected historical data.
# Add a 'Signal' column: 1 = Buy, -1 = Sell, 0 = Hold

import numpy as np

df['Signal'] = 0
# Example: SMA Crossover
buy_condition = df['Close'] > df['SMA20']
sell_condition = df['Close'] < df['SMA20']
df.loc[buy_condition, 'Signal'] = 1
df.loc[sell_condition, 'Signal'] = -1
'''

user_code = st.text_area("✍️ Paste or write your strategy code below:", def_strategy, height=300)
run_button = st.button("🚀 Run Strategy")

if run_button and user_code:
    try:
        # Controlled execution environment
        local_env = {"df": filtered_data.copy(), "np": np, "pd": pd}
        exec(user_code, {}, local_env)
        filtered_data = local_env['df']
        st.success("Strategy executed successfully.")
    except Exception as e:
        st.error(f"Error running strategy: {e}")
        st.stop()

# Plotting
st.title("Stock Price Chart with Strategy Signals")
fig, ax = plt.subplots(figsize=(12, 6))

for ticker in tickers:
    df_plot = filtered_data[filtered_data['Ticker'] == ticker]
    ax.plot(df_plot['Date'], df_plot['Close'], label=f"{ticker} Close")

    buys = df_plot[df_plot['Signal'] == 1]
    sells = df_plot[df_plot['Signal'] == -1]

    ax.plot(buys['Date'], buys['Close'], '^', color='green', label=f"{ticker} Buy", markersize=8)
    ax.plot(sells['Date'], sells['Close'], 'v', color='red', label=f"{ticker} Sell", markersize=8)

ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price" if normalize else "Close Price")
ax.set_title(f"Prices and Signals for {', '.join(tickers)}")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Optional volume plot
if show_volume:
    st.subheader("Volume Chart")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    for ticker in tickers:
        df_vol = filtered_data[filtered_data['Ticker'] == ticker]
        ax2.plot(df_vol['Date'], df_vol['Volume'], label=f"{ticker} Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# Display data and signals
st.subheader("Filtered Data")
st.dataframe(filtered_data)

st.subheader("Buy/Sell Signals")
signals_table = filtered_data[filtered_data['Signal'] != 0][['Date', 'Ticker', 'Close', 'Signal']]
signals_table['Action'] = signals_table['Signal'].map({1: 'Buy', -1: 'Sell'})
st.dataframe(signals_table)

# Download buttons
csv_filtered = filtered_data.to_csv(index=False).encode('utf-8')
csv_signals = signals_table.to_csv(index=False).encode('utf-8')

st.download_button("Download Filtered Data CSV", csv_filtered, "filtered_stock_data.csv", "text/csv")
st.download_button("Download Signals CSV", csv_signals, "buy_sell_signals.csv", "text/csv")
