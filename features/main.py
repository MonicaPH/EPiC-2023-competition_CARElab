import sys, os 
import multiprocessing
from functools import partial
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

def extractor(sub_vid, s, scenario, fold, train_test, data_path, feature_path):
    sub, vid = sub_vid
    if train_test == 'train':
        X, y = s.train_data(sub, vid) if fold == -1 else s.train_data(fold, sub, vid)
    else:
        X, y = s.test_data(sub, vid) if fold == -1 else s.test_data(fold, sub, vid)

    try:
        # # skip when all exists
        # if (feature_path / f'sub_{sub}_vid_{vid}.csv').exists() and (data_path / f'sub_{sub}_vid_{vid}.csv').exists():
        #     logging.info(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""}: clean data and features exists')
        # # save clean data when features exist
        # elif (feature_path / f'sub_{sub}_vid_{vid}.csv').exists():
        #     logging.info(f'scenario {scenario}: features exists')
        #     clean_data, features = feature_extractor(X, y, is_extract_features=False)
        #     clean_data.to_csv(data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        # else:
        clean_data, features = feature_extractor(X, y)
        clean_data.to_csv(data_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        features.to_csv(feature_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""}: extracted features for {train_test} data (sub = {sub} vid = {vid}).')
    except Exception as e:
        logging.error(f'scenario {scenario}{" fold " + str(fold) if fold != -1 else ""} (sub = {sub} vid = {vid}): {e}')

# without folds
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

    func = partial(extractor, s=s, scenario=1, fold=-1,
                   train_test='train', data_path=train_data_path, feature_path=train_feature_path)
    pool_obj = multiprocessing.Pool()
    pool_obj.map(func, s.train_test_indices['train'])
    pool_obj.close()

    func = partial(extractor, s=s, scenario=1, fold=-1,
                   train_test='test', data_path=test_data_path, feature_path=test_feature_path)
    pool_obj = multiprocessing.Pool()
    pool_obj.map(func, s.train_test_indices['test'])
    pool_obj.close()

# with folds
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

        func = partial(extractor, s=s, scenario=scenario, fold=fold,
                       train_test='train', data_path=train_data_path, feature_path=train_feature_path)
        pool_obj = multiprocessing.Pool()
        pool_obj.map(func, s.train_test_indices[fold]['train'])
        pool_obj.close()

        func = partial(extractor, s=s, scenario=scenario, fold=fold,
                       train_test='test', data_path=test_data_path, feature_path=test_feature_path)
        pool_obj = multiprocessing.Pool()
        pool_obj.map(func, s.train_test_indices[fold]['test'])
        pool_obj.close()
