# %%
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
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
