import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

from src.dataloader import S1
from src.utils import data_splitter, check_dir
from autogluon.tabular import TabularDataset, TabularPredictor

import warnings
warnings.filterwarnings("ignore")

import logging, datetime

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format, 
                    force=True,
                    handlers=[
                        logging.FileHandler(f"log/{log_filename}.log"),
                        logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )

# --------------------------------
# 1. data splitter
# --------------------------------
load_dir = Path('processed_data/scenario_1/train')
save_dir = Path('validation_data/processed_data/scenario_1/')

model_dir = Path('autogluon/scenario_1/processed_data_sub_vid_dependent/models')
output_dir = Path('autogluon/scenario_1/processed_data_sub_vid_dependent/output')
target = 'arousal'

check_dir(model_dir)

s1 = S1()
sub_vid_pairs = []
for sub in s1.train_subs[::5]:
    for vid in s1.train_vids[::2]:
        sub_vid_pairs.append((sub, vid))

data_splitter(load_dir, save_dir, s1, sub_vid_pairs)
# --------------------------------
# 2. train
# --------------------------------
def train(sub_vid_pairs, target:str):
    for sub, vid in sub_vid_pairs:
        train_data = TabularDataset(save_dir / 'train' / f'sub_{sub}_vid_{vid}.csv')
        train_data = train_data.fillna(0)
        train_data = train_data.drop(columns=['valence' if target == 'arousal' else 'arousal'])
        predictor = TabularPredictor(label=target, problem_type='regression', path= model_dir / f'sub_{sub}_vid_{vid}_arousal', verbosity=0).fit(train_data, ag_args_fit={'num_gpus': 2})
        logging.info(f'trained model for sub {sub} vid {vid}')


def test(sub_vid_pairs, target:str):
    subs = []
    vids = []
    rmses = []
    list_y_df = []
    for sub, vid in sub_vid_pairs:
        dropped_col = ['arousal', 'valence']
        predictor = TabularPredictor.load(model_dir / f'sub_{sub}_vid_{vid}_arousal')
        test_data = TabularDataset(save_dir / f'test' / f'sub_{sub}_vid_{vid}.csv')
        y_test = test_data[target]
        test_data_nolab = test_data.drop(columns=dropped_col)
        test_data_nolab = test_data_nolab.fillna(0)
        
        y_pred = predictor.predict(test_data_nolab)
        
        y_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
        y_df['sub'] = sub
        y_df['vid'] = vid
        
        list_y_df.append(y_df)
            
        rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
        logging.info(f'Sub {sub} Vid {vid} RMSE: {rmse}')
        rmses.append(rmse)

        logging.info(f'tested model for sub {sub} vid {vid}')

    pd.DataFrame({'sub': subs, 'vid': vids, 'rmse': rmses}).to_csv(
            output_dir / 'rmses.csv'
            )
    pd.concat(list_y_df, axis=0).to_csv(output_dir / 'annotations.csv')

train(sub_vid_pairs, target)
test(sub_vid_pairs, target)
