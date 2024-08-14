import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./data/eval_winrate/5v5.csv")

plt.plot(data.iloc[:, 1], data.iloc[:, 4], label="ippo", color="red")
plt.plot(data.iloc[:, 1], data.iloc[:, 2], alpha=0.2, color="red")
plt.plot(data.iloc[:, 1], data.iloc[:, 3], alpha=0.2, color="red")
plt.fill_between(data.iloc[:, 1], data.iloc[:, 2], data.iloc[:, 3], color='red', alpha=0.2)
plt.legend()
plt.show()
