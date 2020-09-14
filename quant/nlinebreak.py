# %%
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
import matplotlib.pyplot as plt
# %%
lg = bs.login()

print('login respond error_code: ' + lg.error_code)
print('login respond error_msg: ' + lg.error_msg)

rs = bs.query_all_stock(day='2020-09-11')
print('query_all_stock respond error_code: ' + rs.error_code)
print('query_all_stock respond error_msg: ' + rs.error_msg)
# %%
data_list = []
while (rs.error_code == '0') and rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
# %%
result.head()
# %%
data_list = []
for code in tqdm(result['code']):
    rs = bs.query_stock_basic(code=code)
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())
result2 = pd.DataFrame(data_list, columns=rs.fields)

# %%
result2.head()
# %%
all_stocks = result2.loc[(result2['type'] == '1') & (result2['status'] == '1')]
# %%
all_stocks.head()
# %%
bs.logout()
# %%


def download_data(start_date, end_date, stocks_list):
    """
    start_date: start date
    end_date: end date, must be workday
    stocks_list: stocks code lists
    """
    bs.login()
    data_df = pd.DataFrame()
    for code in tqdm(stocks_list):
        k_rs = bs.query_history_k_data_plus(code,
                                            'date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST',
                                            start_date, end_date)
        data_df = data_df.append(k_rs.get_data())
    bs.logout()
    data_df['open'] = data_df['open'].astype('float')
    data_df['high'] = data_df['high'].astype('float')
    data_df['low'] = data_df['low'].astype('float')
    data_df['close'] = data_df['close'].astype('float')
    return data_df


# %%
data = download_data('2020-01-01', '2020-09-14', all_stocks['code'])
# %%
data.head()
# %%
data = data.merge(all_stocks[['code', 'code_name']],
                  how='left', left_on='code', right_on='code')
# %%


def nlinebreak(date, close, n):
    height = []
    top = []
    bottom = []
    color = []
    datetime = []
    for i in range(1, len(close)):
        if not height:
            if close[i] > close[0]:
                top.append(close[i])
                bottom.append(close[0])
                height.append(top[-1] - bottom[-1])
                color.append('red')
                datetime.append(date[i])
            else:
                top.append(close[0])
                bottom.append(close[i])
                height.append(top[-1] - bottom[-1])
                color.append('green')
                datetime.append(date[i])
        else:
            if close[i] > max(top[-n:]):
                bottom.append(top[-1])
                top.append(close[i])
                height.append(top[-1] - bottom[-1])
                color.append('red')
                datetime.append(date[i])
            elif close[i] < min(bottom[-n:]):
                top.append(bottom[-1])
                bottom.append(close[i])
                height.append(top[-1] - bottom[-1])
                color.append('green')
                datetime.append(date[i])
    return datetime, height, top, bottom, color


# %%
qzmt = data.loc[data['code_name'] == '钱江摩托']
# %%
datetime, height, top, bottom, color = nlinebreak(
    qzmt['date'].tolist(), qzmt['close'].tolist(), 3)
# %%
plt.figure(figsize=(16, 9))
plt.bar(datetime, height=height, width=1, bottom=bottom, color=color)
plt.xticks(rotation=90)
plt.show()
# %%


def find_PositiveStar_NegativeStar(df, stockcode='sh.600000'):
    """找出启明星和黄昏之星"""
    # 获取历史K线数据
    hisdata = df.loc[df['code'] == stockcode]
    hisdata = hisdata.reset_index()
    highlist = hisdata['high'].astype('float')
    lowlist = hisdata['low'].astype('float')
    openlist = hisdata['open'].astype('float')
    closelist = hisdata['close'].astype('float')
    zdflist = hisdata['pctChg'].astype('float')
    datelist = hisdata['date']
    for i in range(len(datelist)):
        if i > 2 and i < len(datelist) - 2:
            # 启明星 前一日是中阳线，或者连续两天阳线
            if zdflist[i-1] < -5 or (zdflist[i-1] < -4 and zdflist[i-1] + zdflist[i-2] < -6):
                # 当日高开或者平开，但是有上下影线,且上影线长度超过了K线实体的长度
                # or zdflist[i] < 0
                if (min(openlist[i], closelist[i]) - lowlist[i] > (abs(openlist[i] - closelist[i]))):
                    # 次日下跌，且跌到之前上涨阳线的一半以下
                    if zdflist[i + 1] > 0 and closelist[i + 1] > 0.5*(openlist[i - 1] + closelist[i - 1]):
                        print("at %s，%s出现了启明星，前日涨幅%f，当日涨幅%f，次日涨幅 %f，形态出现后涨幅%f" % (
                            datelist[i], stockcode, zdflist[i-1], zdflist[i], zdflist[i+1], zdflist[i+2]))
                        return stockcode, datelist[i], zdflist[i+2]
    return None, None, None


# %%
find_PositiveStar_NegativeStar(data, 'sz.000913')
# %%
codelist, datelist, zdf = [], [], []
for code in tqdm(all_stocks['code']):
    c, d, z = find_PositiveStar_NegativeStar(data, code)
    if c and d and z:
        codelist.append(c)
        datelist.append(d)
        zdf.append(z)
# %%
res = pd.DataFrame({'stockcode': codelist, 'date': datelist, 'zdf': zdf})
# %%
res['pos'] = res['zdf'].apply(lambda x: 1 if x > 0 else 0)
# %%
sum(res['pos']) / res.shape[0]
# %%
