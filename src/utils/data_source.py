import pandas as pd
import requests
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class FinancialAnalytics:
    def __init__(self, file_path, ticker, date_from, sort):
        self.file_path = file_path
        self.ticker = ticker
        self.date_from = date_from      
        self.sort = sort

    def fetch_market_price_data(self):
        df_tsv = pd.read_csv(self.file_path, sep='\t') # raw data gives tab separated values
        df_tsv.drop('<VOL>', axis=1, inplace=True) #forex data has no real volume 
        df_tsv['DateTime'] = pd.to_datetime(df_tsv['<DATE>'] + ' ' + df_tsv['<TIME>'])
        df_tsv.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
        df_tsv = df_tsv[['DateTime', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<SPREAD>']]
        return df_tsv

    def fetch_news_sentiment_data(self):
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&time_from={self.date_from}&limit=1000&sort={self.sort}&apikey=151J1CO4W4YLKP87'
        r = requests.get(url)
        eur_news_data = r.json()
        df = pd.DataFrame(eur_news_data)
        nested_df = pd.json_normalize(df['feed'])
        filtered_data = nested_df['ticker_sentiment'].apply(lambda x: any(item.get('ticker') == self.ticker for item in x))
        nested_df = nested_df[filtered_data]
        nested_df['ticker_sentiment'] = nested_df['ticker_sentiment'].apply(lambda x: [item for item in x if item.get('ticker') == self.ticker][0])
        eur_sentiment = pd.json_normalize(nested_df['ticker_sentiment'])
        df_time = pd.json_normalize(df['feed'])
        df_time['time_published'] = pd.to_datetime(df_time['time_published'], format='%Y%m%dT%H%M%S')
        df_time['time_published'] = df_time['time_published'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        df_time['time_published'] = df_time['time_published'].dt.strftime('%Y-%m-%d %H:%M')
        eur_sentiment['time_published(est)'] = df_time['time_published']
        return eur_sentiment

    def cluster_data(self): 
        df_cluster = self.fetch_market_price_data()
        df1 = df_cluster.drop('DateTime', axis=1)
        df1 = df1.dropna()
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df1)
        wcss = []
        for i in range(1, 6):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(scaled_df)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 6), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        best_k = 6  # Assuming 6 is used based on domain knowledge
        kmeans = KMeans(n_clusters=best_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(scaled_df)
        df_cluster['Cluster'] = pred_y
        return df_cluster

    def all_features_combined(self):
        #news_sentiment = self.fetch_news_sentiment_data()
        #news_sentiment = news_sentiment.drop('ticker', axis=1)
        news_sentiment = pd.read_csv(r'C:\Users\jessi\OneDrive\1 Projects\1-svts\data\external\news-analysis\usd_sentiment_combined_20231023T1516.csv')
        clustered_data = self.cluster_data()
        clustered_data['DateTime'] = pd.to_datetime(clustered_data['DateTime'])
        news_sentiment['time_published(est)'] = pd.to_datetime(news_sentiment['time_published(est)']) 
        all_features_df = pd.merge(clustered_data, news_sentiment, left_on='DateTime', right_on='time_published(est)', how='outer')
        return all_features_df