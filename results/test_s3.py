import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S3
from test_model import test
import warnings
import logging, datetime

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

prefix = '../'

scenario = 3
s3 = S3()

model_path = Path(prefix) / f'models/scenario_{scenario}'
for fold in s3.fold:
    save_path = Path(prefix) / f'results/scenario_3/fold_{fold}/test/annotations'
    check_dir(save_path)
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'fold_{fold}' / 'test' / 'physiology'
    
    if fold == 0:
        arousal_model_path = model_path / 'arousal_03421'
        valence_model_path = model_path  / 'valence_4102122'
    elif fold == 1:
        arousal_model_path = model_path / 'arousal_10162022'
        valence_model_path = model_path  / 'valence_4102122'
    elif fold == 2:
        arousal_model_path = model_path / 'arousal_03421'
        valence_model_path = model_path / 'valence_031620'
    else:
        arousal_model_path = model_path / 'arousal_10162022'
        valence_model_path = model_path / 'valence_031620'
        
    for sub in s3.test_subs[fold]:
        for vid in s3.test_vids[fold]:
            logging.info(f'start predict fold {fold} sub {sub} ...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X, arousal_model_path, valence_model_path, save_path / f'sub_{sub}_vid_{vid}.csv')
            logging.info(f'finish predicting fold {fold} sub {sub} vid {vid}.')
