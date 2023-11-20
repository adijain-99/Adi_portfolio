# Adi_portfolio
Data Analysis Portfolio


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




## Twitter Sentiment Analysis

In this project, we'll use Natural Language Processing (NLP) techniques and a sentiment analysis algorithm to analyze the sentiment of tweets. We'll implement this using Python, Tweepy for Twitter data retrieval, and the popular NLP library, NLTK, for sentiment analysis.

```
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Setting up Twitter API credentials
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define a function to retrieve tweets
def get_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search, q=query, lang="en").items(count)
    tweet_list = [tweet.text for tweet in tweets]
    return tweet_list

# Get tweets related to a specific query (e.g., "Python")
tweets = get_tweets("Python", count=200)

# Preprocess tweets
stop_words = set(stopwords.words('english'))

def preprocess_tweet(tweet):
    # Remove special characters, links, and stopwords
    tweet = ' '.join([word.lower() for word in tweet.split() if word.isalpha() and word not in stop_words])
    return tweet

tweets = [preprocess_tweet(tweet) for tweet in tweets]

# Analyze sentiment using TextBlob
sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
```

Also, I used an advanced NLP Model (VADER Sentiment Analysis). I introduced VADER Sentiment Analysis from NLTK, which is specifically designed for social media text.
```
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiment using VADER
vader_sentiments = [sid.polarity_scores(tweet)['compound'] for tweet in tweets]
```

Also, I made a real-Time Dashboard with Dash. I used Dash, a Python web framework, to create a real-time dashboard for sentiment analysis. The dashboard updates every few seconds to display the sentiment histogram and word clouds for positive and negative sentiments.
```
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of the dashboard
app.layout = html.Div(children=[
    dcc.Graph(id='sentiment-histogram'),
    dcc.Graph(id='word-cloud-positive'),
    dcc.Graph(id='word-cloud-negative')
])

# Define callback for real-time updates
@app.callback(
    [Output('sentiment-histogram', 'figure'),
     Output('word-cloud-positive', 'figure'),
     Output('word-cloud-negative', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n_intervals):
    # Get real-time tweets
    real_time_tweets = get_tweets("AI", count=100)
    
    # Preprocess and analyze sentiment using TextBlob
    real_time_tweets = [preprocess_tweet(tweet) for tweet in real_time_tweets]
    real_time_sentiments = [TextBlob(tweet).sentiment.polarity for tweet in real_time_tweets]

    # Update Sentiment Histogram
    fig_sentiment = sns.histplot(real_time_sentiments, bins=20, kde=True)
    fig_sentiment = plt.gcf()

    # Update Word Clouds
    real_time_positive_tweets = ' '.join([tweet for tweet, sentiment in zip(real_time_tweets, real_time_sentiments) if sentiment > 0])
    real_time_negative_tweets = ' '.join([tweet for tweet, sentiment in zip(real_time_tweets, real_time_sentiments) if sentiment < 0])

    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(real_time_positive_tweets)
    fig_wordcloud_positive = plt.gcf()

    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(real_time_negative_tweets)
    fig_wordcloud_negative = plt.gcf()

    return fig_sentiment, fig_wordcloud_positive, fig_wordcloud_negative

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
```


Visualize Sentiment Analysis
```
sns.histplot(sentiments, bins=20, kde=True)
plt.title('Sentiment Distribution in Tweets')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.show()

# Display Word Cloud of positive and negative words
positive_tweets = ' '.join([tweet for tweet, sentiment in zip(tweets, sentiments) if sentiment > 0])
negative_tweets = ' '.join([tweet for tweet, sentiment in zip(tweets, sentiments) if sentiment < 0])

# Word Cloud for Positive Tweets
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud for Positive Tweets')
plt.axis('off')
plt.show()

# Word Cloud for Negative Tweets
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud for Negative Tweets')
plt.axis('off')
plt.show()
```

This project offers a more sophisticated sentiment analysis approach, considers multiple queries, and presents the results in a real-time dashboard for continuous monitoring. 

### Conclusions:
1. Sentiment Distribution - Most tweets are neutral, but the distribution of sentiment polarity provides insights into the overall sentiment.
2. Positive and Negative Words - Word clouds help identify the most frequent words associated with positive and negative sentiments in the analyzed tweets.
3. Further Analysis - Explore sentiment patterns over time, sentiment correlations with specific hashtags, or sentiment differences across user demographics.










