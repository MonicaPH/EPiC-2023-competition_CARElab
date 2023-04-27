import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S2
import warnings
import logging, datetime
import argparse

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--fold', type=int, help='0, 1, 2, 3, 4')
args = parser.parse_args()

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
    arousal = predictor_arousal.predict(test_data)
    logging.info('arousal predicted')

    predictor_valence = TabularPredictor.load(str(model_path) + '_valence')
    valence = predictor_valence.predict(test_data)
    logging.info('valence predicted')

    predictions = pd.DataFrame({'valence': valence, 'arousal': arousal})
    predictions.to_csv(save_path)

prefix = '../'

scenario = 2
s2 = S2()

model_path = Path(prefix) / f'models/scenario_{scenario}'

if args.fold is not None:
    folds = [args.fold]
else:
    folds = s2.fold

for fold in folds:
    save_path = Path(prefix) / f'results/scenario_{scenario}/fold_{fold}/test/annotations'
    check_dir(save_path)
    input_path = Path(prefix) / f'io_data/scenario_{scenario}/fold_{fold}/test/physiology'
    for sub in s2.test_subs[fold]:
        for vid in s2.test_vids[fold]:
            logging.info(f'start predicting fold {fold} sub {sub} vid {vid} ...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X, model_path / f'fold_{fold}_vid_{vid}', save_path / f'sub_{sub}_vid_{vid}.csv')
            logging.info(f'fold {fold} sub {sub} vid {vid} finished.')
