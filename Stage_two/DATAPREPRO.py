import pandas as pd
import numpy as np


def prepro(env_name, train_s):
    if env_name in ["sim_chosmm_5000_g2_2_0.2", "sim_chosmm_50_10000_g2_2_0.2", "sim_chosmm_10_10000_g2_2_0.2",
                    "sim_chosmm_3_5000_g2_2_0.2_SO_1","sim_chosmm_3_5000_g2_2_0.2_SO_2"]:
        data = np.load("../data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1]
        data = data[0].reshape(len(label), -1)
        targetdata = data
    elif env_name in ["exchange"]:
        HLlist = ["USD_CNY", "USD_EUR", "USD_JPY"]
        data_all = pd.read_csv("../data/exchange.csv")
        data = []
        targetdata = []
        max = []
        min = []
        for name in HLlist:
            lsdata = np.array(
                data_all[[name + "_close", name + "_open", name + "_high", name + "_low", name + "_zdf"]]).reshape(-1,
                                                                                                                   5)
            lsmax = np.max(lsdata[:, 0:4])
            lsmin = np.min(lsdata[:, 0:4])
            lsdata[:, 0:4] = (lsdata[:, 0:4] - lsmin) / (lsmax - lsmin)
            lstarget = np.array(data_all[name + "_close"])
            lstarget = (lstarget - lsmin) / (lsmax - lsmin)
            max.append(lsmax)
            min.append(lsmin)
            data.append(lsdata.reshape(-1, 1, 5))
            targetdata.append(lstarget.reshape(-1, 1))

        datag = np.concatenate(data, axis=1)
        targetdatag = np.concatenate(targetdata, axis=1)
        label = np.zeros_like(targetdatag)
    elif env_name in ["machine"]:
        data_all = pd.read_csv("../data/machine-1-1_testg.csv")
        data = np.array(data_all.iloc[:, :-1])
        targetdata = np.array(data_all.iloc[:, :-1])
        label = np.zeros_like(data)
        for i in range(data.shape[1]):
            label[:, i] = np.array(data_all.loc[:, "label"])
    if env_name not in ["exchange"]:
        max = np.max(data, axis=0)
        min = np.min(data, axis=0)
        datag = (data - min) / (max - min)
        max = np.max(targetdata, axis=0)
        min = np.min(targetdata, axis=0)
        targetdatag = (targetdata - min) / (max - min)

    return datag, targetdatag, max, min, label
