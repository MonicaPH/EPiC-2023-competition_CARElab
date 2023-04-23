import sys, os 
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.path.relpath("../src/"))
from dataloader import S3
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

root_path = Path(__file__).parents[1]
feature_path = root_path / 'features'


scenario = 3
s = S3()

logging.info(f'scenario {scenario}')
scenario_path = feature_path / f'scenario_{scenario}'
if not scenario_path.exists():
    scenario_path.mkdir(parents=True)

for fold, train_test_indices in enumerate(s.train_test_indices):
    fold_path = scenario_path / f'fold_{fold}'
    if not fold_path.exists():
        fold_path.mkdir(parents=True)

    train_path = fold_path / 'train'
    test_path = fold_path / 'test'
    if not train_path.exists():
        train_path.mkdir(parents=True)
    if not test_path.exists():
        test_path.mkdir(parents=True)

    for sub, vid in train_test_indices['train']:
        X, y = s.train_data(fold, sub, vid)
        feature_extractor(X, y).set_index(y.index).to_csv(train_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario {scenario} fold {fold}: extracted features for training data (sub = {sub} vid = {vid}).')

    for sub, vid in train_test_indices['test']:
        X, y = s.test_data(fold, sub, vid)
        feature_extractor(X, y).set_index(y.index).to_csv(test_path / f'sub_{sub}_vid_{vid}.csv', index_label='time')
        logging.info(f'scenario {scenario} fold {fold}: extracted features for test data (sub = {sub} vid = {vid}).')
