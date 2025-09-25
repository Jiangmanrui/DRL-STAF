import pandas as pd
import numpy as np


def prepro(env_name, train_s):
    if env_name in ["sim_arima", "sim_arima4"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".npy")
        targetdata = data
        label = np.hstack([np.hstack([np.zeros(500), np.ones(500)])] * 5)
    elif env_name in ["sim_arima_markov", "sim_arima_markov2", "sim_sin_markov", "sim_arima_markov3",
                      "sim_arima_markov3g"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1]
        data = data[0]
        targetdata = data
    elif env_name in ["sim_network_arima_markov_5000_g3_0.2", "sim_network_arima_markov_5000_mm_g3_0.2",
                      "sim_chosmm_5000_g2_2_0.2", "sim_chosmm_50_10000_g2_2_0.2", "sim_chosmm_10_10000_g2_2_0.2"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1]
        data = data[0]
        targetdata = data

    elif env_name in ["HL"]:
        HLlist = ["USD_CNY", "USD_EUR", "USD_JPY", "USD_TRY"]
        data_all = pd.read_csv("G:/mypro/predict_and_states/data/all_HL.csv")
        name = [x + "_zdf" for x in HLlist]
        # print(name)
        label = np.zeros((len(data_all), len(HLlist)))
        data = np.array(data_all[name])

    elif env_name in ["HL2"]:
        HLlist = ["USD_CNY", "USD_EUR", "USD_JPY", "USD_TRY"]
        data_all = pd.read_csv("G:/mypro/predict_and_states/data/all_HL2.csv")
        name = [x + "_close" for x in HLlist]
        # print(name)
        label = np.zeros((len(data_all), len(HLlist)))
        data = np.array(data_all[name])

    elif env_name in ["ECG_data", "PeMS07g", "PeMS07gg"]:
        data_all = pd.read_csv("G:/mypro/predict_and_states/data/" + env_name + ".csv", header=None)
        data = np.array(data_all)
        label = np.zeros_like(data)
    elif env_name in ["machine"]:
        data_all = pd.read_csv("G:/mypro/predict_and_states/data/machine-1-1_testg.csv")
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


def prepro2(env_name, train_s, nn=0):
    if env_name in ["sim_arima", "sim_arima4"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".npy")
        targetdata = data
        label = np.hstack([np.hstack([np.zeros(500), np.ones(500)])] * 5)
    elif env_name in ["sim_arima_markov", "sim_arima_markov2", "sim_arima_markov2g", "sim_arima_markov2g2",
                      "sim_sin_markov",
                      "sim_arima_markov3", "sim_arima_markov3g"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1]
        data = data[0]
        targetdata = data
    elif env_name in ["sim_network_arima_markov_5000_g3_0.1", "sim_network_arima_markov_5000_g3_0.2",
                      "sim_network_arima_markov_5000_g2_0.2",
                      "sim_network_arima_markov_5000_mm_g3_0.1", "sim_network_arima_markov_5000_mm_g3_0.2",
                      "sim_chosmm_5000_g2_2_0.2", "sim_chosmm_50_10000_g2_2_0.2", "sim_chosmm_10_10000_g2_2_0.2"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".pkl", allow_pickle=True)
        label = list(data[1][:, nn])
        data = data[0][:, nn]
        targetdata = data

    # train_size = int(len(data) * train_s)
    # max = np.max(data[0:train_size], axis=0)
    # min = np.min(data[0:train_size], axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    datag = (data - min) / (max - min)
    # max = np.max(targetdata[0:train_size], axis=0)
    # min = np.min(targetdata[0:train_size], axis=0)
    max = np.max(targetdata, axis=0)
    min = np.min(targetdata, axis=0)
    targetdatag = (targetdata - min) / (max - min)

    datag = datag.reshape(datag.shape[0], datag.shape[-1])
    # print(label)
    return datag, max, min, label


def prepro3(env_name, train_s, nn=0):
    if env_name in ["sim_arima", "sim_arima4"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".npy")
        targetdata = data
        label = np.hstack([np.hstack([np.zeros(500), np.ones(500)])] * 5)
    elif env_name in ["sim_network_arima_markov_0", "sim_network_arima_markov_0.2", "sim_network_arima_markov_0.4",
                      "sim_network_arima_markov_0_2", "sim_network_arima_markov_0.2_2",
                      "sim_network_arima_markov_g_0.2", "sim_network_arima_markov_g2_0.2",
                      "sim_network_arima_markov_g3_0.2",
                      "sim_network_arima_markov_gg",
                      "sim_network_arima_markov2_0_0.2", "sim_network_arima_markov2_0.1_0.2",
                      "sim_network_arima_markov2_g_0_0.2",
                      "sim_network_arima_markov2_g_0.2_0.2", "sim_network_arima_markov2_g_0.2_0.4"]:
        data = np.load("G:/mypro/predict_and_states/data/" + env_name + ".pkl", allow_pickle=True)
        label = data[1][:, nn]
        data = data[0]
        targetdata = data[:, nn]

    train_size = int(len(data) * train_s)
    # max = np.max(data[0:train_size], axis=0)
    # min = np.min(data[0:train_size], axis=0)
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    # min = -10
    # max = 10
    datag = (data - min) / (max - min)
    # max = np.max(targetdata[0:train_size], axis=0)
    # min = np.min(targetdata[0:train_size], axis=0)
    max = np.max(targetdata, axis=0)
    min = np.min(targetdata, axis=0)
    # min = -10
    # max = 10
    targetdatag = (targetdata - min) / (max - min)
    # print(min,max)
    # print(data.shape)
    # print(label.shape)

    return datag, targetdatag, max, min, label
