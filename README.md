# Adi_portfolio
Data science portfolio


## Real-Time Cryptocurrency Price Analysis

I used a cryptocurrency API that provides real-time price data. The CoinGecko API.

### Import Necessary Libraries

```ruby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime
from matplotlib.animation import FuncAnimation
```

### Set Up Real-Time Data Retrieval
Function to fetch real-time cryptocurrency price data
```ruby
def get_crypto_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '1',
        'interval': '1m'
    }
    response = requests.get(url, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
```

Function to calculate simple moving average (SMA)
```ruby
def calculate_sma(data, window=20):
    data['SMA'] = data['price'].rolling(window=window).mean()
    return data
```

Function to check for price crossing SMA and generate alerts
```ruby
def check_alert(data):
    if data['price'].iloc[-1] > data['SMA'].iloc[-1]:
        print(f"Alert: Bitcoin price crossed above SMA at {data['timestamp'].iloc[-1]}!")
    elif data['price'].iloc[-1] < data['SMA'].iloc[-1]:
        print(f"Alert: Bitcoin price crossed below SMA at {data['timestamp'].iloc[-1]}!")
```

### Real-Time Visualization with Matplotlib Animation
Initialize the plot
```ruby
fig, ax = plt.subplots(figsize=(10, 6))
plt.title('Real-Time Bitcoin Price with SMA')

# Function to update the plot with real-time data, SMA, and alerts
def update_plot(frame):
    data = get_crypto_data()
    data = calculate_sma(data, window=20)
    
    ax.clear()
    ax.plot(data['timestamp'], data['price'], label='Bitcoin Price', color='b')
    ax.plot(data['timestamp'], data['SMA'], label='SMA (20)', color='r', linestyle='dashed')
    
    check_alert(data)  # Check for alerts
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()

# Use FuncAnimation to update the plot every minute
ani = FuncAnimation(fig, update_plot, interval=60000)

plt.show()
```

- I added a calculate_sma function to calculate the simple moving average (SMA) of the Bitcoin price.
- The check_alert function checks if the current price has crossed above or below the SMA and prints an alert message accordingly.
- The update_plot function now includes plotting both the Bitcoin price and the SMA on the same plot.

### Conclusions:
1) Real-Time Bitcoin Price - The plot displays the real-time price of Bitcoin updated every minute.
2) Trends and Patterns - The plot analyze trends and patterns in the Bitcoin price over time.
3) Integration with Dashboard - The plot helps to extend the project by integrating the real-time plot into a dashboard using tools like Dash or Streamlit.



