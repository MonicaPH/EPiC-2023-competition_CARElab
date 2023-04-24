import sys, os 
import pandas as pd
import numpy as np
from pathlib import Path


sys.path.append(os.path.relpath("../src/"))
from dataloader import S1
from feature_extractor import feature_extractor

import logging, datetime

import warnings
warnings.filterwarnings("ignore")

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format, 
                    force=True,
                    handlers=[
                        logging.FileHandler(f"../log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

root_path = Path('../').parents[1]

def check_dir(*dirs):
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True)

processed_data_path = root_path / 'processed_data'
train_data_path = processed_data_path / 'scenario_1' / 'train'
test_data_path = processed_data_path / 'scenario_1' / 'test'

feature_path = root_path / 'features'
train_feature_path = feature_path / 'scenario_1' / 'train'
test_feature_path = feature_path / 'scenario_1' / 'test'

check_dir(processed_data_path,
          train_data_path,
          test_data_path,
          feature_path,
          train_feature_path,
          test_feature_path)

s1 = S1()
for sub, vid in s1.train_test_indices['train']:
    X, y = s1.train_data(sub, vid)

    if (feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
        processed_data, features = feature_extractor(X, y, is_extract_features=False)
        processed_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
    else:
        processed_data, features = feature_extractor(X, y)
        processed_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        features.to_csv(train_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

    logging.info(f'scenario 1: extracted features for training data (sub = {sub} vid = {vid}).')

for sub, vid in s1.train_test_indices['test']:
    X, y = s1.test_data(sub, vid)
    if (feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
        processed_data, features = feature_extractor(X, y, is_extract_features=False)
        processed_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
    else:
        processed_data, features = feature_extractor(X, y)
        processed_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        features.to_csv(test_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
    logging.info(f'scenario 1: extracted features for test data (sub = {sub} vid = {vid}).')
