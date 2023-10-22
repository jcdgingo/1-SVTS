import pandas as pd
import requests
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class FinancialAnalytics:

    def __init__(self, ticker='FOREX:EUR', date_from='20210404T0130', sort='EARLIEST', file_path=r'C:\Users\jessi\OneDrive\1 Projects\1-svts\data\raw\EURUSD_M1_201101030000_202310201036.csv'):
        self.ticker = ticker
        self.date_from = date_from
        self.file_path = file_path
        self.sort = sort

    def fetch_market_price_data(self):
        df_tsv = pd.read_csv(self.file_path, sep='\t')
        df_tsv.drop('<VOL>', axis=1, inplace=True)
        df_tsv['DateTime'] = pd.to_datetime(df_tsv['<DATE>'] + ' ' + df_tsv['<TIME>'])
        df_tsv.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
        df_tsv = df_tsv[['DateTime', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<SPREAD>']]
        output_csv_file_path = r'C:\Users\jessi\OneDrive\1 Projects\1-svts\data\interim\EURUSD_M1.csv'
        df_tsv.to_csv(output_csv_file_path, index=False)
        return df_tsv

    def fetch_news_sentiment_data(self, apikey='151J1CO4W4YLKP87'):
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&time_from={self.date_from}&limit=1000&sort={self.sort}&apikey={apikey}'
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
        eur_sentiment['time_published(est)'] = df_time['time_published']
        return eur_sentiment

    def cluster_data(self): 
        df_cluster = self.fetch_market_price_data()
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df_cluster)
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
        best_k = 6  # Assuming 6 is used based on your domain knowledge
        kmeans = KMeans(n_clusters=best_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(scaled_df)
        df_cluster['Cluster'] = pred_y
        return df_cluster

    # Additional methods to append data can be added here
