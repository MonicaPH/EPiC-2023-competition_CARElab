import os

if __name__ == '__main__':
    # test all models
    for sensor in ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap', 'all']:
        cmd = f'python3 test_LSTM.py {sensor} szn3_fold1_testForLSTM.csv'
        print(cmd)
        os.system(cmd)

        # create plot
        cmd = f'python3 plot_performance.py performance_{sensor}.csv'
        print(cmd)
        os.system(cmd)