import pandas as pd
from pathlib import Path

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S4
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

scenario = 4
s4 = S4()

model_path = Path(prefix) / f'models/scenario_{scenario}'
for fold in s4.fold:
    save_path = Path(prefix) / f'results/scenario_4/fold_{fold}/test/annotations'
    check_dir(save_path)
    a_model_list = [model_path / f'vid_{vid}_arousal' for vid in s4.train_vids[fold]]
    v_model_list = [model_path / f'vid_{vid}_valence' for vid in s4.train_vids[fold]]
    input_path = Path(prefix) / f'io_data/scenario_{scenario}' / f'fold_{fold}' / 'test' / 'physiology'
    logging.info(f'use models {a_model_list} and {v_model_list}')
    for sub in s4.test_subs[fold]:
        for vid in s4.test_vids[fold]:
            logging.info(f'start predict fold {fold} sub {sub} ...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X, a_model_list, v_model_list, save_path / f'sub_{sub}_vid_{vid}.csv', late_fusion=True)
            logging.info(f'finish predicting fold {fold} sub {sub} vid {vid}.')
