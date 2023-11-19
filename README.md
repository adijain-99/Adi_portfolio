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
### Function to fetch real-time cryptocurrency price data
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

### Real-Time Visualization with Matplotlib Animation
### Initialize the plot
```ruby
fig, ax = plt.subplots(figsize=(10, 6))
plt.title('Real-Time Bitcoin Price')

# Function to update the plot with real-time data
def update_plot(frame):
    data = get_crypto_data()
    ax.clear()
    ax.plot(data['timestamp'], data['price'], label='Bitcoin Price', color='b')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (USD)')
    ax.legend()

# Use FuncAnimation to update the plot every minute
ani = FuncAnimation(fig, update_plot, interval=60000)

plt.show()
```



