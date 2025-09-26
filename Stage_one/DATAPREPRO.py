import pandas as pd
import numpy as np

def prepro(env_name, train_s, nn=0):
    if env_name in ["sim_chosmm_5000_g2_2_0.2", "sim_chosmm_50_10000_g2_2_0.2", "sim_chosmm_10_10000_g2_2_0.2"]:
        data = np.load("../data/" + env_name + ".pkl", allow_pickle=True)
        label = list(data[1][:, nn])
        data = data[0][:, nn]
        targetdata = data
    elif env_name in ["exchange"]:
        HLlist = ["USD_CNY", "USD_EUR", "USD_JPY"]
        data_all = pd.read_csv("../data/exchange.csv")
        name = HLlist[nn]
        label = np.zeros(len(data_all))
        data = np.array(data_all[[name + "_close", name + "_open", name + "_high", name + "_low", name + "_zdf"]])
        targetdata = np.array(data_all[name + "_close"])
    elif env_name in ["machine"]:
        data_all = pd.read_csv("../data/machine-1-1_testg.csv")
        data = np.array(data_all.iloc[:, nn])
        targetdata = np.array(data_all.iloc[:, nn])
        label = np.array(data_all.loc[:, "label"])

    if env_name not in ["HL2"]:
        max = np.max(data, axis=0)
        min = np.min(data, axis=0)
        datag = (data - min) / (max - min)
        max = np.max(targetdata, axis=0)
        min = np.min(targetdata, axis=0)
        targetdatag = (targetdata - min) / (max - min)
    else:
        max = np.max(data[:, 0:4])
        min = np.min(data[:, 0:4])
        datag = data
        datag[:, 0:4] = (data[:, 0:4] - min) / (max - min)
        targetdatag = (targetdata - min) / (max - min)

    return datag, targetdatag, max, min, label