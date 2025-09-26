import pandas as pd
import numpy as np


def prepro(env_name, train_s):
    if env_name in ["sim_chosmm_5000_g2_2_0.2", "sim_chosmm_50_10000_g2_2_0.2",
                    "sim_chosmm_10_10000_g2_2_0.2"]:
        data = np.load("../data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1]
        data = data[0]
        targetdata = data

    elif env_name in ["exchange"]:
        HLlist = ["USD_CNY", "USD_EUR", "USD_JPY"]
        data_all = pd.read_csv("../data/exchange.csv")
        name = [x + "_close" for x in HLlist]
        # print(name)
        label = np.zeros((len(data_all), len(HLlist)))
        data = np.array(data_all[name])

    elif env_name in ["machine"]:
        data_all = pd.read_csv("../data/machine-1-1_testg.csv")
        data = np.array(data_all.iloc[:, 0:3])
        targetdata = np.array(data_all.iloc[:, 0:3])
        label = np.zeros_like(data)
        label[:, 0] = np.array(data_all.loc[:, "label"])
        label[:, 1] = np.array(data_all.loc[:, "label"])
        label[:, 2] = np.array(data_all.loc[:, "label"])

    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    datag = (data - min) / (max - min)

    return datag, max, min, label
