import bs4 as bs
import _pickle as pickle
import requests
import os
import datetime as dt
import pandas as pd
import pandas_datareader.data as web 
import matplotlib.pyplot as plt
import matplotlib.style as style 
import numpy as np

style.use('seaborn')

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table',  {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers


def get_data(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2018, 12, 31)

    for i, ticker in enumerate(tickers):
        print('{}. {}'.format(i, ticker))
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            except Exception:
                print('\nxxxx {} presented and error. Skipping...\n'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        except FileNotFoundError:
            print('No data for {}.csv. Skipping...'.format(ticker))
            continue
        except Exception as e:
            raise e

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        
        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


def visualize_data(n):
    df = pd.read_csv('sp500_joined_closes.csv')
    df.drop(df.columns[len(df.columns)-400:], axis=1, inplace=True)
    df_corr = df.corr()
    data = df_corr.values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    save_sp500_tickers()
    get_data()
    compile_data()
    visualize_data()
    