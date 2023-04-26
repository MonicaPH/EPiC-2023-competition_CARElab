import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S4
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

def test(X, model_list, save_path):
    test_data = TabularDataset(X)

    arousals = []
    for model_path in model_list:
        logging.info(str(model_path) + '_arousal')
        predictor_arousal = TabularPredictor.load(str(model_path) + '_arousal', require_py_version_match=False)
        pred_arousal = predictor_arousal.predict(test_data)
        arousals.append(pred_arousal)
        logging.info('arousal predicted')

    valences = []
    for model_path in model_list:
        predictor_valence = TabularPredictor.load(str(model_path) + '_valence', require_py_version_match=False)
        pred_valence = predictor_valence.predict(test_data)
        valences.append(pred_valence)
        logging.info('valence predicted')

    predictions = pd.DataFrame({'valence': pd.concat(valences, axis=1).mean(axis=1), 'arousal': pd.concat(arousals, axis=1).mean(axis=1)})
    # predictions = pd.concat([pd.concat(valences, axis=1).mean(axis=1), pd.concat(arousals, axis=1).mean(axis=1)], axis=1)
    predictions.to_csv(save_path)

prefix = '../'

scenario = 4
s4 = S4()

model_path = Path(prefix) / f'models/scenario_{scenario}'
for fold in s4.fold:
    save_path = Path(prefix) / f'results/scenario_4/fold_{fold}/test/annotations'
    check_dir(save_path)
    model_list = [model_path / f'vid_{vid}' for vid in s4.train_vids[fold]]
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'fold_{fold}' / 'test' / 'physiology'
    logging.info(f'use models {model_list}')
    for sub in s4.test_subs[fold]:
        for vid in s4.test_vids[fold]:
            logging.info(f'start predict fold {fold} sub {sub} ...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X, model_list, save_path / f'sub_{sub}_vid_{vid}.csv')
            logging.info(f'finish predicting fold {fold} sub {sub} vid {vid}.')

            exit()
