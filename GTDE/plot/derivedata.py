import pandas as pd

path1 = "./data/eval_winrate/tune2_run1_logs_eval_win_rate_eval_win_rate.csv"
path2 = "./data/eval_winrate/tune2_run3_logs_eval_win_rate_eval_win_rate.csv"
path3 = "./data/eval_winrate/tune2_run5_logs_eval_win_rate_eval_win_rate.csv"

data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)
new_data = pd.concat([data1.iloc[:, 1:], data2.iloc[:, 2], data3.iloc[:, 2]], axis=1)

pd.concat([new_data.iloc[:, 0], new_data.iloc[:, 1:].max(axis=1), new_data.iloc[:, 1:].min(axis=1),
           new_data.iloc[:, 1:].mean(axis=1)], axis=1).to_csv("./data/eval_winrate/5v5.csv")
