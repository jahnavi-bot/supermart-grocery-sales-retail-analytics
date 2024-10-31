from flask import Flask, render_template, request
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
from prophet import Prophet

app = Flask(__name__)

# Load the dataset
data_path = "C:\\Users\\bella\\OneDrive\\Dokumen\\janu2\\janu2\\climate_nasa.csv"
data = pd.read_csv(data_path)

# Ensure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Function to calculate sentiment
def get_sentiment(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        return blob.sentiment.polarity
    return None

# Keywords
keywords = ['atmosphere', 'co2', 'global warming', 'planet', 'climate change', 'world', 'climate', 'people']

# Function to count keywords
def count_keywords(text):
    if isinstance(text, str):
        return {keyword: text.lower().count(keyword) for keyword in keywords}
    return {keyword: 0 for keyword in keywords}

# Calculate sentiments for the dataset
data['sentiment'] = data['text'].apply(get_sentiment)
data.dropna(subset=['sentiment'], inplace=True)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.dropna(subset=['date'], inplace=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    avg_sentiment = None
    filename = None
    selected_timeframe = None

    if request.method == 'POST':
        selected_value = request.form.get('value')
        selected_value_type = request.form.get('value_type')
        
        # Filter data based on selected timeframe
        if selected_value_type == 'day':
            daily_data = data[data['date'].dt.day == int(selected_value)]
            selected_timeframe = f"Day {selected_value}"
            filename = 'sentiment_day.png'
        elif selected_value_type == 'month':
            daily_data = data[data['date'].dt.month == int(selected_value)]
            selected_timeframe = f"Month {selected_value}"
            filename = 'sentiment_month.png'
        elif selected_value_type == 'year':
            daily_data = data[data['date'].dt.year == int(selected_value)]
            selected_timeframe = f"Year {selected_value}"
            filename = 'sentiment_year.png'
        
        if not daily_data.empty:
            avg_sentiment = daily_data['sentiment'].mean()
            plt.figure(figsize=(8, 4))
            daily_data['sentiment'].plot(kind='bar', color='skyblue')
            plt.title(f'Sentiment Analysis for {selected_timeframe}')
            plt.xlabel('Record Index')
            plt.ylabel('Sentiment Score')
            plt.tight_layout()
            plt.savefig(f'static/{filename}')
            plt.close()
        else:
            filename = None

    return render_template('sentiment_analysis.html', avg_sentiment=avg_sentiment, timeframe=selected_timeframe, filename=filename)

@app.route('/keyword_frequency', methods=['GET', 'POST'])
def keyword_frequency():
    filename = None
    selected_timeframe = None

    if request.method == 'POST':
        selected_value = request.form.get('value')
        selected_value_type = request.form.get('value_type')
        
        if selected_value_type == 'day':
            filtered_data = data[data['date'].dt.day == int(selected_value)]
            selected_timeframe = f"Day {selected_value}"
            filename = 'keyword_counts_day.png'
        elif selected_value_type == 'month':
            filtered_data = data[data['date'].dt.month == int(selected_value)]
            selected_timeframe = f"Month {selected_value}"
            filename = 'keyword_counts_month.png'
        elif selected_value_type == 'year':
            filtered_data = data[data['date'].dt.year == int(selected_value)]
            selected_timeframe = f"Year {selected_value}"
            filename = 'keyword_counts_year.png'
        
        if not filtered_data.empty:
            keyword_counts = filtered_data['text'].apply(count_keywords)
            total_keyword_counts = pd.DataFrame(keyword_counts.tolist()).sum()
            plt.figure(figsize=(10, 5))
            total_keyword_counts.plot(kind='bar', color='coral')
            plt.title(f'Keyword Frequency for {selected_timeframe}')
            plt.xlabel('Keywords')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'static/{filename}')
            plt.close()
        else:
            filename = None

    return render_template('keyword_frequency.html', timeframe=selected_timeframe, filename=filename)

@app.route('/future_prediction', methods=['GET', 'POST'])
def future_prediction():
    filename = None
    prediction = None

    if request.method == 'POST':
        timeframe = request.form.get('timeframe')
        
        forecast_data = data[['date', 'sentiment']].rename(columns={'date': 'ds', 'sentiment': 'y'})
        forecast_data['ds'] = forecast_data['ds'].dt.tz_localize(None)  # Remove timezone

        model = Prophet()
        model.fit(forecast_data)

        if timeframe == 'day':
            periods = 10
            freq = 'D'
        elif timeframe == 'month':
            periods = 12
            freq = 'M'
        elif timeframe == 'year':
            periods = 5
            freq = 'Y'

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        plt.figure(figsize=(10, 5))
        plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sentiment', color='blue')
        plt.plot(forecast_data['ds'], forecast_data['y'], label='Historical Sentiment', color='black')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.title(f'Future Sentiment Predictions ({timeframe.capitalize()})')
        plt.legend()
        filename = f'future_sentiment_prediction_{timeframe}.png'
        plt.savefig(f'static/{filename}')
        plt.close()

        forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
        prediction = forecast[['ds', 'yhat']].tail(periods).to_dict(orient='records')

    return render_template('future_prediction.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
