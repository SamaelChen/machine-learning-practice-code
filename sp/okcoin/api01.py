import pandas as pd
import requests
import time

l = []
for i in range(30):
    r = requests.get(
        'https://www.okcoin.cn/api/v1/ticker.do?symbol=eth_cny')
    l.append(r.json()['ticker'])
    time.sleep(2)

pd.DataFrame(l)
