import pandas as pd
import numpy as np
import tushare as ts
from pymongo import MongoClient
import matplotlib.pyplot as plt
from get_data_func import get_portfolios

pro = ts.pro_api()

client = MongoClient(host="10.23.0.3", port=27017)
client.admin.authenticate("admin", "11031103", mechanism='SCRAM-SHA-1')
db = client.fundsdb

def calc_D(returns_df, starttime, endtime, code, eta):
    return_cluster = returns_df.loc[starttime:endtime, code]  # 获取所有基金时间范围内收益
    D = return_cluster.mean()-(eta*return_cluster.std())  # 收益减方差
    return D

# 计算置信上界
def upper_bound_probs(weight, D, x_new):
    D = np.mat(D)
    A_a = np.mat(np.dot(D.T, D)+np.eye(D.shape[1]))
    upper_bound_probs = np.dot(np.mat(x_new), weight) + 0.25*np.sqrt(np.dot(np.dot(x_new.values, A_a.I), x_new.values.T))
    return upper_bound_probs

s_date = '20160101'
e_date = '20201231'

HS300 = pro.index_daily(ts_code='399300.SZ', start_date=s_date, end_date=e_date, fields='trade_date,close')
HS300['trade_date'] = pd.to_datetime(HS300['trade_date'])
HS300.set_index('trade_date', inplace=True)
HS300.sort_index(inplace=True)

# 根据基金code获取基金数据
def get_fund_nav(code_list, s_time, e_time):
    acc_nav_df = pd.DataFrame()
    nav_list = list(db.stockfundNAV.aggregate(
        [{'$match': {'code': {'$in': code_list}}},
         {'$project': {'code': 1,
                       'nav/d': {'$filter': {'input': '$nav/d',
                                             'as': 'item',
                                             'cond': {'$and': [{'$gte': ['$$item.end_date', s_time]},
                                                               {'$lte': ['$$item.end_date', e_time]}
                                                               ]
                                                      }
                                             }
                                 }
                       }
          }
        ]
    ))

    for i in range(len(nav_list)):
        for slice in nav_list[i]['nav/d']:
            acc_nav_df.loc[slice['end_date'], nav_list[i]['code']] = slice['accum_nav']

    acc_nav_df.index = pd.to_datetime(acc_nav_df.index)
    acc_nav_df.sort_index(inplace=True)

    # 查看持有期delta天数中是否有基金的数据缺失过多
    # a = pd.DataFrame(acc_nav_df.loc[change_date:].isna().sum())
    # lst = []
    # for i in range(len(a)):
    #     if int(a.iloc[i]) > int(len(a)*0.2):
    #         lst.append(str(a.index[i]))
    # acc_nav_df.drop(lst, axis=1, inplace=True)
    acc_nav_df.fillna(method='ffill', inplace=True)
    return acc_nav_df

moneyPrt = 1000000
moneyHS300 = 1000000
backtesting = pd.DataFrame(columns=['date','Portfolio-capital','HS300-capital'])
cum_regret = pd.DataFrame(columns=['date','cum_regret'])
calc_cluster = {}

t = 100         # date_all[100] = '2016-06-01'
delta = 10      # m=delta+n
n = 40
numround = 1218
regret = 0

date_all = list(HS300.index)

while True:
    # 设定m和n期限
    timerange_n = date_all[t-n:t]
    timerange_m = date_all[t-n-delta:t]

    # 根据时间获取cluster。首先将时间格式替换为‘YearMonthDay’
    s_time = str(timerange_m[0])[:10].replace('-','')
    e_time = str(date_all[t - n - delta:t + delta][-1])[:10].replace('-', '')
    c_time = str(timerange_m[-1])[:10].replace('-','')
    print('调仓日期:', c_time)

#!!!!!  cluster需要进行调整！！！应该是在一段时间内cluster保持不变。
    # 同时要记录在cluster保持不变的时候每个portfolio被推荐的次数
    cluster = get_portfolios(s_time, e_time, c_time, 10)

    # 初始化权重——每个portfolio中的股票同权
    weight_list = [1] * len(cluster)
    for k in range(len(cluster)):
        w = np.mat([1 / len(cluster[k])] * len(cluster[k])).T
        weight_list[k] = w

    # 获取cluster里面基金的净值数据。时间范围要考虑后期持有的delta天数，即时间范围应该是date_all[t-n-delta:t+delta]
    code_list = list(set([code for portfolio in cluster for code in portfolio]))
    nav_df = get_fund_nav(code_list, s_time, e_time)
    returns_df = (nav_df - nav_df.shift(1))/nav_df.shift(1)

    bound_dict = {}
    for k in range(len(cluster)):
        x = calc_D(returns_df.loc[:timerange_m[-1]], starttime=timerange_n[0], endtime=timerange_n[-1], code=cluster[k], eta=0.02)
        D = calc_D(returns_df.loc[:timerange_m[-1]], starttime=timerange_m[0], endtime=timerange_m[-1], code=cluster[k], eta=0.02)
        bound_dict[k] = upper_bound_probs(weight_list[k], D, x)

    key_name = max(bound_dict, key=bound_dict.get)
    print('选择的臂:',key_name)
    print('-'*20)

    # 得到所选臂的实际收益，更新特征及
    backtesting = backtesting.append(
        [{'date': timerange_m[-1], 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300}], ignore_index=True)
    code = cluster[key_name]
    amount = []
    # 获取最优portfolio的每个股票权重
    weight = weight_list[key_name]
    # 将资金按权重分给对应股票
    a = moneyPrt * weight
    # 计算资金可购买的股票数量
    for j in range(len(code)):
        amount.append(int(a[j][0, 0] / nav_df[code[j]][timerange_m[-1]]))
    # 买股票后的剩余资金
    cash_funds = moneyPrt - (np.mat(amount) * (np.mat(nav_df.loc[timerange_m[-1], code]).T))[0, 0]

    # 可以购买指数的数量
    amount_index = int(moneyHS300 / HS300.loc[timerange_m[-1]])
    # 买指数后的剩余资金
    cash_index = moneyHS300 - (HS300.loc[timerange_m[-1]] * amount_index)

    # 计算持有基金delta天的每天收益（前提是这些基金在这delta天任然运营）
    holding_time = date_all[t:t+delta]
    for d in holding_time:
        moneyPrt = (cash_funds + (np.mat(amount) * (np.mat(nav_df.loc[d, code].values).T)))[0, 0]
        moneyHS300 = cash_index + (amount_index * HS300.loc[d])
        backtesting = backtesting.append([{'date': d, 'Portfolio-capital': moneyPrt, 'HS300-capital': moneyHS300[0]}],
                                         ignore_index=True)

        # 计算后悔度
        reward = []
        for j in range(len(cluster)):
            weight = weight_list[j]
            reward.append((np.dot(np.mat(weight).T, returns_df.loc[d, cluster[j]]))[0, 0])
        regret += np.max(reward) - reward[key_name]
        cum_regret = cum_regret.append([{'date': d, 'cum_regret': regret}], ignore_index=True)

    t = t + delta + 1
    # 循环结束条件 numround选1218对应2020-12-31
    if t >= numround:
        break

# 画收益率图
backtesting.set_index('date',inplace=True)
df_return_rate = backtesting/backtesting.iloc[0]
df_return_rate.columns = ['Portfolio Return Rate','HS300 Return Rate']
df_return_rate = df_return_rate.astype(float)
df_return_rate.plot(figsize=(16,7))

# 后悔度图
# cum_regret.set_index('date',inplace=True)
# cum_regret['log_cum_regret'] = np.log(cum_regret['cum_regret'])
# cum_regret['log_cum_regret'].plot(figsize=(16,7))
plt.show()