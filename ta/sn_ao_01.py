#  https://medium.com/codex/bitcoin-trade-automation-with-awesome-oscillator-in-python-51f2c52c5b25
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
from poptimizer.shared import adapters, col
import time as tm


plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

def get_crypto_price(symbol, exchange, start_date = None):
    api_key = open(r'api_key.txt')
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    df = df.rename(columns = {'1a. open (USD)': 'Open', '2a. high (USD)': 'High', '3a. low (USD)': 'Low', '4a. close (USD)': 'Close', '5. volume': 'Volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    return df

def LoadFromTV(ticker: str, TVformula: str, mult: int):
    import pandas as pd

    sn_ticker = ticker
    sn_df_quotes = pd.DataFrame(columns = [col.DATE, col.OPEN, col.CLOSE, col.HIGH, col.LOW, col.TURNOVER])
    sn_df_quotes = sn_df_quotes.set_index(col.DATE)

    from poptimizer.tvDatafeed import TvDatafeed,Interval
    tv = TvDatafeed()
    tv.clear_cache()
#    t_GLDRUB = tv.get_hist(TVformula, interval=Interval.in_daily,n_bars=100000)
    t_GLDRUB = tv.get_hist(TVformula, interval=Interval.in_4_hour,n_bars=100000)

    sn_df_quotes[col.OPEN] = t_GLDRUB['open']
    sn_df_quotes[col.CLOSE] = t_GLDRUB['close']
    sn_df_quotes[col.HIGH] = t_GLDRUB['high']
    sn_df_quotes[col.LOW] = t_GLDRUB['low']
    sn_df_quotes[col.TURNOVER] = t_GLDRUB['volume'].mul(t_GLDRUB['close']) * mult
#    sn_df_quotes.index = sn_df_quotes.index.normalize()         # Откинем часы
    return sn_df_quotes




btc = LoadFromTV('BTCRUB', "BINANCE:BTCUSD", 1)
#btc = get_crypto_price(symbol = 'BTC', exchange = 'USD', start_date = '2020-01-01')
print(btc)



def sma(price, period):
    sma = price.rolling(period).mean()
    return sma

def ao(price, period1, period2):
    median = price.rolling(2).median()
    short = sma(median, period1)
    long = sma(median, period2)
    ao = short - long
    ao_df = pd.DataFrame(ao).rename(columns = {'Close':'ao'})
    return ao_df

btc['ao'] = ao(btc['CLOSE'], 5, 34)
btc = btc.dropna()
print(btc.tail())

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((10,1), (6,0), rowspan = 4, colspan = 1)
ax1.plot(btc['CLOSE'])
ax1.set_title('BITCOIN CLOSING PRICE')
for i in range(len(btc)):
    if btc['ao'][i-1] > btc['ao'][i]:
        ax2.bar(btc.index[i], btc['ao'][i], color = '#f44336')
    else:
        ax2.bar(btc.index[i], btc['ao'][i], color = '#26a69a')
ax2.set_title('BITCOIN AWESOME OSCILLATOR 5,34')
plt.show()
plt.savefig('/home/sn/sn/poptimizer-master/ta/pos_' + tm.strftime("%Y%m%d_%H%M%S", tm.gmtime()) + '01.png')


def implement_ao_crossover(price, ao):
    buy_price = []
    sell_price = []
    ao_signal = []
    signal = 0
    
    for i in range(len(ao)):
        if ao[i] > 0 and ao[i-1] < 0:
            if signal != 1:
                buy_price.append(price[i])
                sell_price.append(np.nan)
                signal = 1
                ao_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                ao_signal.append(0)
        elif ao[i] < 0 and ao[i-1] > 0:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(price[i])
                signal = -1
                ao_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                ao_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            ao_signal.append(0)
    return buy_price, sell_price, ao_signal

buy_price, sell_price, ao_signal = implement_ao_crossover(btc['CLOSE'], btc['ao'])

ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((10,1), (6,0), rowspan = 4, colspan = 1)
ax1.plot(btc['CLOSE'], label = 'BTC', color = 'skyblue')
ax1.plot(btc.index, buy_price, marker = '^', markersize = 12, color = '#26a69a', linewidth = 0, label = 'BUY SIGNAL')
ax1.plot(btc.index, sell_price, marker = 'v', markersize = 12, color = '#f44336', linewidth = 0, label = 'SELL SIGNAL')
ax1.legend()
ax1.set_title('BITCOIN CLOSING PRICE')
for i in range(len(btc)):
    if btc['ao'][i-1] > btc['ao'][i]:
        ax2.bar(btc.index[i], btc['ao'][i], color = '#f44336')
    else:
        ax2.bar(btc.index[i], btc['ao'][i], color = '#26a69a')
ax2.set_title('BITCOIN AWESOME OSCILLATOR 5,34')
plt.show()
plt.savefig('/home/sn/sn/poptimizer-master/ta/pos_' + tm.strftime("%Y%m%d_%H%M%S", tm.gmtime()) + '02.png')


position = []
for i in range(len(ao_signal)):
    if ao_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(btc['CLOSE'])):
    if ao_signal[i] == 1:
        position[i] = 1
    elif ao_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
ao = btc['ao']
close_price = btc['CLOSE']
ao_signal = pd.DataFrame(ao_signal).rename(columns = {0:'ao_signal'}).set_index(btc.index)
position = pd.DataFrame(position).rename(columns = {0:'ao_position'}).set_index(btc.index)

frames = [close_price, ao, ao_signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

print(strategy)
print(strategy[20:25])

strategy.to_excel('/home/sn/sn/poptimizer-master/ta/pos_' + tm.strftime("%Y%m%d_%H%M%S", tm.gmtime()) + '.xlsx', index=True)



btc_ret = pd.DataFrame(np.diff(btc['CLOSE'])).rename(columns = {0:'returns'})
ao_strategy_ret = []

for i in range(len(btc_ret)):
    returns = btc_ret['returns'][i]*strategy['ao_position'][i]
    ao_strategy_ret.append(returns)
    
ao_strategy_ret_df = pd.DataFrame(ao_strategy_ret).rename(columns = {0:'ao_returns'})
investment_value = 200000
number_of_stocks = floor(investment_value/btc['CLOSE'][-1])
ao_investment_ret = []

for i in range(len(ao_strategy_ret_df['ao_returns'])):
    returns = number_of_stocks*ao_strategy_ret_df['ao_returns'][i]
    ao_investment_ret.append(returns)

ao_investment_ret_df = pd.DataFrame(ao_investment_ret).rename(columns = {0:'investment_returns'})
total_investment_ret = round(sum(ao_investment_ret_df['investment_returns']), 2)
profit_percentage = round((total_investment_ret/investment_value)*100, 2)
print(cl('Profit gained from the AO strategy by investing $200k in BTC : {}'.format(total_investment_ret), attrs = ['bold']))
print(cl('Profit percentage of the AO strategy : {}%'.format(profit_percentage), attrs = ['bold']))
