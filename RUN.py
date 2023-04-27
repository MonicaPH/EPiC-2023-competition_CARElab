import os

if __name__ == '__main__':
    # extract features
    os.system('cd features && python3 features.py -s 1')
    os.system('cd features && python3 features.py -s 2')
    os.system('cd features && python3 features.py -s 3')
    os.system('cd features && python3 features.py -s 4')

    # prepare train/test datasets
    os.system('cd io_data && python3 s1_train.py')
    os.system('cd io_data && python3 s1_test.py')
    os.system('cd io_data && python3 s2_train_v2.py')
    os.system('cd io_data && python3 s2_test.py')
    os.system('cd io_data && python3 s3_train.py')
    os.system('cd io_data && python3 s3_test.py')
    os.system('cd io_data && python3 s4_train.py')
    os.system('cd io_data && python3 s4_test.py')

    # train models
    os.system('cd models && python3 train_s1_121.py')
    os.system('cd models && python3 train_s1_123.py')
    os.system('cd models && python3 train_s1_129.py')
    os.system('cd models && python3 train_s1_130.py')
    os.system('cd models && python3 train_s1_131.py')
    os.system('cd models && python3 train_s1_171.py')
    os.system('cd models && python3 train_s1_250.py')
    os.system('cd models && python3 train_s2_121.py')
    os.system('cd models && python3 train_s2_123.py')
    os.system('cd models && python3 train_s2_129.py')
    os.system('cd models && python3 train_s2_130.py')
    os.system('cd models && python3 train_s2_250.py')
    os.system('cd models && python3 train_s3_123.py')
    os.system('cd models && python3 train_s3_130.py')
    os.system('cd models && python3 train_s4_129.py')
    os.system('cd models && python3 train_s4_250.py')
    
    # create results
    os.system('cd results && python3 test_s1.py')
    os.system('cd results && python3 test_s2.py')
    os.system('cd results && python3 test_s3.py')
    os.system('cd results && python3 test_s4.py')
    os.system('cd results && python3 filter_results.py')


    ## impact of shift on prediction
    os.system('cd CARElab && python3 train_LSTM_lag_models.py')
    os.system('cd CARElab && python3 test_LSTM_lag_models.py')