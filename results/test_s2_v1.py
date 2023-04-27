import pandas as pd
from pathlib import Path

import sys, os
sys.path.append(os.path.relpath("../src/"))
from utils import check_dir
from dataloader import S2
from test import test
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

scenario = 2
s2 = S2()

model_path = Path(f'../models/scenario_{scenario}')

for fold in s2.fold:
    save_path = Path(f'../results/scenario_2/fold_{fold}/test/annotations')
    check_dir(save_path)
    model_list = [model_path / f'sub_{sub}' for sub in s2.train_subs[fold]]
    input_path = Path(f'../io_data/scenario_{scenario}/fold_{fold}/test/physiology')
    logging.info(f'use models {model_list}')
    for sub in s2.test_subs[fold]:
        for vid in s2.test_vids[fold]:
            logging.info(f'start predict fold {fold} sub {sub} ...')
            X = pd.read_csv(input_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            test(X, model_list, save_path / f'sub_{sub}_vid_{vid}.csv', late_fusion=True)
            logging.info(f'finish predicting fold {fold} sub {sub} vid {vid}.')
