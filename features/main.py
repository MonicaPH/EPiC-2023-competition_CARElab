import sys, os 
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from dataloader import S2, S1, S3, S4
from utils import check_dir
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

parser = argparse.ArgumentParser(description='Process data, extract features')
parser.add_argument('--scenario', type=int, help='1, 2, 3, 4', required=True)
parser.add_argument('--fold', type=int)

args = parser.parse_args()

scenario = args.scenario

assert scenario in [1, 2, 3, 4], logging.error('scenario should be one of 1, 2, 3, 4')

root_path = Path('../')
feature_path = root_path / 'features'
clean_data_path = root_path / 'clean_data'

if scenario == 1:
    s = S1()
elif scenario == 2:
    s = S2()
elif scenario == 3:
    s = S3()
else: 
    s = S4()

logging.info(f'scenario {scenario}')

if scenario == 1:
    train_data_path = clean_data_path / f'scenario_{scenario}' / 'train'
    test_data_path = clean_data_path / f'scenario_{scenario}' / 'test'
    train_feature_path = feature_path / f'scenario_{scenario}' / 'train'
    test_feature_path = feature_path / f'scenario_{scenario}' / 'test'
    check_dir(clean_data_path,
              train_data_path,
              test_data_path,
              feature_path,
              train_feature_path,
              test_feature_path)

    for sub, vid in s.train_test_indices['train']:
        X, y = s.train_data(sub, vid)

        try:
            # skip when all exists
            if (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (train_data_path / f'sub_{sub}_vid_{vid}.csv').exists():
                logging.info(f'scenario {scenario}: clean data and features exists')
                continue

            # save clean data when features exist
            elif (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
                logging.info(f'scenario {scenario}: features exists')
                clean_data, features = feature_extractor(X, y, is_extract_features=False)
                clean_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            else:
                clean_data, features = feature_extractor(X, y)
                clean_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                features.to_csv(train_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

            logging.info(f'scenario 1: extracted features for training data (sub = {sub} vid = {vid}).')
        except Exception as e:
            logging.error(f'scenario {scenario} (sub = {sub} vid = {vid}): {e}')

    for sub, vid in s.train_test_indices['test']:
        X, y = s.test_data(sub, vid)

        try:
            if (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (train_data_path / f'sub_{sub}_vid_{vid}.csv').exists():
                continue

            if (test_feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
                clean_data, features = feature_extractor(X, y, is_extract_features=False)
                clean_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            else:
                clean_data, features = feature_extractor(X, y)
                clean_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                features.to_csv(test_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
            logging.info(f'scenario {scenario}: extracted features for test data (sub = {sub} vid = {vid}).')
        except Exception as e:
            logging.error(f'scenario {scenario} (sub = {sub} vid = {vid}): {e}')

else:
    folds = s.fold if args.fold == None else args.fold

    if type(folds) == int:
        folds = [folds]

    for fold in folds:
        logging.info(f'fold {fold}')
        train_data_path = clean_data_path / f'scenario_{scenario}' / f'fold_{fold}' / 'train'
        test_data_path = clean_data_path / f'scenario_{scenario}' / f'fold_{fold}' / 'test'
        train_feature_path = feature_path / f'scenario_{scenario}' / f'fold_{fold}' / 'train'
        test_feature_path = feature_path / f'scenario_{scenario}' / f'fold_{fold}' / 'test'
        check_dir(clean_data_path,
                  train_data_path,
                  test_data_path,
                  feature_path,
                  train_feature_path,
                  test_feature_path)

        for sub, vid in s.train_test_indices[fold]['train']:
            try:
                X, y = s.train_data(fold, sub, vid)
                # skip when all exists
                if (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (train_data_path / f'sub_{sub}_vid_{vid}.csv').exists():
                    continue

                # save clean data when features exist
                elif (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
                    clean_data, features = feature_extractor(X, y, is_extract_features=False)
                    clean_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                else:
                    clean_data, features = feature_extractor(X, y)
                    clean_data.to_csv(train_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                    features.to_csv(train_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')

                logging.info(f'scenario {scenario} fold {fold}: extracted features for training data (sub = {sub} vid = {vid}).')
            except Exception as e:
                logging.error(f'scenario {scenario} fold {fold}: (sub = {sub} vid = {vid}) --- {e}')

        for sub, vid in s.train_test_indices[fold]['test']:
            try:
                X, y = s.test_data(fold, sub, vid)
                if (train_feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (train_data_path / f'sub_{sub}_vid_{vid}.csv').exists():
                    continue

                if (test_feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
                    clean_data, features = feature_extractor(X, y, is_extract_features=False)
                    clean_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                else:
                    clean_data, features = feature_extractor(X, y)
                    clean_data.to_csv(test_data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                    features.to_csv(test_feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
                logging.info(f'scenario {scenario} fold {fold}: extracted features for test data (sub = {sub} vid = {vid}).')
            except Exception as e:
                logging.error(f'scenario {scenario} fold {fold}: (sub = {sub} vid = {vid}) --- {e}')
