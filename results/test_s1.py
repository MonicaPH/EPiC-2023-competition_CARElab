import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S1
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

def test(X, model_path, save_path):
    test_data = TabularDataset(X)

    predictor_arousal = TabularPredictor.load(str(model_path) + '_arousal')
    pred_arousal = predictor_arousal.predict(test_data)
    logging.info('arousal predicted')

    predictor_valence = TabularPredictor.load(str(model_path) + '_valence')
    pred_valence = predictor_valence.predict(test_data)
    logging.info('valence predicted')

    predictions = pd.DataFrame({'valence': pred_valence, 'arousal': pred_arousal})
    predictions.to_csv(save_path)

prefix = '../'

scenario = 1
num_gpus = 1

input_path = Path(prefix) / f'io_data/scenario_{scenario}' / 'test' / 'physiology'
model_path = Path(prefix) / f'models/scenario_{scenario}'
save_path = Path(prefix) / f'results/scenario_{scenario}/test/annotations'
check_dir(model_path, save_path)

s1 = S1()
for sub in s1.test_subs:
    for vid in s1.test_vids:
        logging.info(f'start sub {sub} vid {vid}...')
        X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        test(X, model_path / f'sub_{sub}_vid_{vid}', save_path / f'sub_{sub}_vid_{vid}.csv')
        logging.info(f'finish sub {sub} vid {vid}.')
