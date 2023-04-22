import pandas as pd
import numpy as np
import os

from autogluon.tabular import TabularDataset, TabularPredictor

from src.dataloader import S1

import logging, datetime

log_format = '%(asctime)s [%(levelname)s] %(message)s'
log_filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(format=log_format, 
                    force=True,
                    handlers=[
                        logging.FileHandler(f"log/{log_filename}.log"),
#                         logging.StreamHandler()
                        ],
                    level=logging.INFO
                    )


label = 'arousal'
path_prefix = 'data'
save_prefix = 'splitted_data'
s1 = S1()
subjectID = []
videoID = []
root_mean_squared_error = []

leader_board_dataframe = None

train_pairs = s1.train_test_indices['train']
for sub, vid in train_pairs:
    train_data = TabularDataset(os.path.join(save_prefix, f'scenario_1/train', f'sub_{sub}_vid_{vid}.csv'))
    train_data = train_data.drop(columns=['valence'])
    predictor = TabularPredictor(label=label, problem_type='regression', path=f'AutogluonModels/scenario_1/sub_{sub}_vid_{vid}_arousal', verbosity=0).fit(train_data, ag_args_fit={'num_gpus': 2})

    subjectID.append(sub)
    videoID.append(vid)

    test_data = TabularDataset(os.path.join(save_prefix, f'scenario_1/test', f'sub_{sub}_vid_{vid}.csv'))
    y_test = test_data[label]
    test_data_nolab = test_data.drop(columns=[label, 'valence'])

    predictor = TabularPredictor.load(f'AutogluonModels/scenario_1/sub_{sub}_vid_{vid}_arousal')
    
    y_pred = predictor.predict(test_data_nolab)
    rmse = np.sqrt((y_pred - y_test) ** 2).mean()
    root_mean_squared_error.append(rmse)
    logging.info(f'Sub {sub} Vid {vid} RMSE: {rmse}')
    
evaluation_dataframe = pd.DataFrame({'subjectID': subjectID, 'videoID': videoID, 'rmse': root_mean_squared_error})
evaluation_dataframe.to_csv(f'AutogluonModels/scenario_1/evaluation_arousal.csv')
