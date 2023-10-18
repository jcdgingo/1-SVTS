import requests
import pandas as pd
from pandas import json_normalize
import pytz

def extract_sentiment(ticker='FOREX:EUR', date_from='20210404T0130', sort='EARLIEST', apikey='151J1CO4W4YLKP87'):
    # Construct the API URL
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={date_from}&limit=1000&sort={sort}&apikey={apikey}'
    r = requests.get(url)
    eur_news_data = r.json()

    # Convert json to dataframe
    df = pd.DataFrame(eur_news_data)

    # Normalize the feed column where sentiment is located
    nested_df = json_normalize(df['feed'])

    # Filter rows where the 'ticker' key in the dictionary is the specified ticker
    filtered_data = nested_df['ticker_sentiment'].apply(lambda x: any(item.get('ticker') == ticker for item in x))

    # Keep only the rows that match the condition
    nested_df = nested_df[filtered_data]

    # Extract the specific dictionary for the given ticker for each row
    nested_df['ticker_sentiment'] = nested_df['ticker_sentiment'].apply(lambda x: [item for item in x if item.get('ticker') == ticker][0])

    # Normalize json and convert to dataframe
    eur_sentiment = pd.json_normalize(nested_df['ticker_sentiment'])

    # Add time published to the dataframe
    df_time = pd.json_normalize(df['feed'])

    # Convert the time data to pandas datetime format
    df_time['time_published'] = pd.to_datetime(df_time['time_published'], format='%Y%m%dT%H%M%S')

    # Convert from UTC to NY local time
    df_time['time_published'] = df_time['time_published'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

    # Append this as a new column to the main DataFrame
    eur_sentiment['time_published(est)'] = df_time['time_published']

    return eur_sentiment

# You can call the function like this:
# sentiment_data = extract_sentiment(ticker='FOREX:EUR', date_from='20210404T0130', sort='EARLIEST')
# print(sentiment_data)
