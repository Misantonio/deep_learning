import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

def create_csv():
    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2018, 12, 31)

    df = web.DataReader('TSLA', 'yahoo', start, end)
    df.to_csv('tesla.csv')

# create_csv()

df = pd.read_csv('tesla.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
df.dropna(inplace=True)
ax1 = plt.subplot2grid((7,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((7,1), (5,0), rowspan=3, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()
