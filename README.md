# Adi_portfolio
Data Analysis Portfolio


==Real-Time Cryptocurrency Price Analysis==

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
1. Real-Time Bitcoin Price - The plot displays the real-time price of Bitcoin updated every minute.
2. Trends and Patterns - The plot analyze trends and patterns in the Bitcoin price over time.
3. Integration with Dashboard - The plot helps to extend the project by integrating the real-time plot into a dashboard using tools like Dash or Streamlit.





## Real-Time Predictive Modeling on the Titanic Dataset

Let's adapt the Titanic predictive modeling project to make it suitable for a real-time scenario. For a more practical real-time project, I used a continuous stream of data and update the model dynamically. However, it's important to note that deploying machine learning models in a real-time environment involves more complex considerations, such as model deployment frameworks, monitoring, and scalability. Below is a simplified example using a streaming approach with simulated real-time updates:

I downloaded the Titanic dataset from the following link: [Titanic Dataset]([https://www.example.com](https://www.kaggle.com/c/titanic/data)https://www.kaggle.com/c/titanic/data)

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

titanic_data = pd.read_csv('path_to_your_file/train.csv')
print(titanic_data.head())
```   
Data Preprocessing and Feature Engineering
```
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = titanic_data[features]
y = titanic_data['Survived']
```
Initialize and Train the Initial Model
```
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
```
### Model Persistence:
Save the trained model to disk after each update, allowing to reload the latest model if the application restarts.
```
import joblib
joblib.dump(model, 'path_to_your_model/model.joblib')
```
### Logging:
Implement logging to keep track of model updates, accuracy, and any potential issues.
```
import logging
# Configure logging
logging.basicConfig(filename='real_time_model.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# ...

# Log the real-time model accuracy
logging.info(f"Real-Time Model Accuracy: {accuracy_val:.2f}")
```
### Visualization:
Visualize the real-time predictions or other relevant metrics in a graphical interface.
```
import matplotlib.pyplot as plt
from IPython.display import clear_output

# ...

# Visualize real-time predictions (for a Jupyter notebook)
def visualize_real_time(predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, marker='o', linestyle='-', color='b')
    plt.title('Real-Time Model Predictions')
    plt.xlabel('Updates')
    plt.ylabel('Survival Predictions')
    plt.grid(True)
    plt.show()
    clear_output(wait=True)

# ...

# In the loop, add predictions to a list and call the visualization function
predictions_list = []

while True:
    # ...

    # Predict on the validation set
    y_pred_val = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    predictions_list.append(accuracy_val)

    # Visualize real-time predictions
    visualize_real_time(predictions_list)

    # Pause for a short duration
    time.sleep(60)
```


Simulate Real-Time Updates (Continuously Updating Model)
```
while True:
    # Simulate real-time data updates (for demonstration purposes)
    # In a real-world scenario, you would receive new data from a streaming source
    new_data = titanic_data.sample(10)  # Simulate new data
    
    # Perform data preprocessing on the new data
    # (Same as in the initial preprocessing steps)
    # ...
    
    # Update the model with the new data
    X_new = new_data[features]
    y_new = new_data['Survived']
    model.fit(X_new, y_new)
    
    # Evaluate the updated model on a validation set (assuming you have a validation set)
    X_val, y_val = titanic_data.sample(20)[features], titanic_data.sample(20)['Survived']
    y_pred_val = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    
    # Display the real-time model accuracy
    print(f"Real-Time Model Accuracy: {accuracy_val:.2f}")
    
    # Pause for a short duration (simulating real-time intervals)
    time.sleep(60)  # Sleep for 1 minute before the next update
```

### Flask API:
Deploy the real-time predictive model as a RESTful API using Flask.
```
from flask import Flask, request, jsonify

app = Flask(__name__)

# API endpoint for real-time predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']  # Make sure to send the required features for prediction
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```

### Conclusions:
1. Real-Time Model Updates - The model is continuously updated with simulated real-time data.
2. Real-Time Model Accuracy - The real-time accuracy of the model is continuously displayed.
3. This model provide better persistence, logging, visualization, and even the option to deploy the real-time predictive model as an API through Flask API.




## 
