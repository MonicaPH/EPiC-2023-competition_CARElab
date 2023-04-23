import sys, os 
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


sys.path.append(os.path.relpath("../src/"))
from dataloader import S1
from feature_extractor import feature_extractor, emg_feature_extractor

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


parser = argparse.ArgumentParser(description='S1 feature extractor')
parser.add_argument('--sub_group', type=int, help='1, 2, 3, default=None (all groups)')
parser.add_argument('--emg_only', type=int, help='0(no), 1(yes), default=0', default=0)
args = parser.parse_args()

root_path = Path(__file__).parents[1]
feature_path = root_path / 'features'
scenario_path = feature_path / 'scenario_1'
train_path = scenario_path / 'train'
test_path = scenario_path / 'test'

if not scenario_path.exists():
    scenario_path.mkdir(parents=True)
if not train_path.exists():
    train_path.mkdir(parents=True)
if not test_path.exists():
    test_path.mkdir(parents=True)

s1 = S1()
group = s1.train_subs if args.sub_group is None else np.array(s1.train_subs).reshape(3, -1)[args.sub_group - 1]

logging.info(f'Group {group}')

for sub, vid in s1.train_test_indices['train']:
    if sub not in group:
        continue

    X, y = s1.train_data(sub, vid)
    
    if sub == 32 and vid == 1:
        feature_extractor(X, y).set_index(y.index).to_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario 1: extracted features for training data (sub = {sub} vid = {vid}).')

    if args.emg_only == 1:
        features = pd.read_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        emg_zygo, emg_coru, emg_trap = emg_feature_extractor(X, y)
        features = pd.concat([features,
                              emg_zygo.set_index(features.index),
                              emg_coru.set_index(features.index),
                              emg_trap.set_index(features.index)], axis=1).to_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

        logging.info(f'scenario 1 (emg only): extracted features for training data (sub = {sub} vid = {vid}).')

    else:
        feature_extractor(X, y).set_index(y.index).to_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario 1: extracted features for training data (sub = {sub} vid = {vid}).')


for sub, vid in s1.train_test_indices['test']:
    if sub not in group:
        continue

    X, y = s1.test_data(sub, vid)

    if sub in [32, 33, 34, 35] and vid == 1:
        feature_extractor(X, y).set_index(y.index).to_csv(test_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario 1: extracted features for test data (sub = {sub} vid = {vid}).')

    if args.emg_only == 1:
        features = pd.read_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        emg_zygo, emg_coru, emg_trap = emg_feature_extractor(X, y)
        features = pd.concat([features,
                              emg_zygo.set_index(features.index),
                              emg_coru.set_index(features.index),
                              emg_trap.set_index(features.index)], axis=1).to_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

        logging.info(f'scenario 1 (emg only): extracted features for training data (sub = {sub} vid = {vid}).')
    else:
        feature_extractor(X, y).set_index(y.index).to_csv(test_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario 1: extracted features for test data (sub = {sub} vid = {vid}).')
