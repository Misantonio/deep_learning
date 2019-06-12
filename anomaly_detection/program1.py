import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16.0, 6.0)

data = pd.DataFrame(np.random.randn(250))
# data.plot()
# data.hist()
# plt.show()

window = 10
sigma = 2

data['suelo'] = data[0].rolling(window=window).mean() - (sigma*data[0].rolling(window=window).std())
data['techo'] = data[0].rolling(window=window).mean() + (sigma*data[0].rolling(window=window).std())

# data.plot()
# plt.show()

data["anom"] = data.apply(
    lambda row: row[0] if (row[0] <= row["suelo"] or row[0] >= row["techo"]) else 0, axis=1
)

data.plot()
plt.show()