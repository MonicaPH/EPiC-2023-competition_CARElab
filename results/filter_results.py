import sys
import pandas as pd
from pathlib import Path

sys.path.append('../src')
from dataloader import S1, S2, S3, S4
from scipy.ndimage import uniform_filter1d
from utils import check_dir

import argparse

parser = argparse.ArgumentParser(description='Apply average filter to the predictions')
parser.add_argument('--size', type=int, default=10, help='average sample number')
args = parser.parse_args()

size = args.size

def avg_filter(df, size=10):
    df.valence = uniform_filter1d(df.valence, size=size)
    df.arousal = uniform_filter1d(df.arousal, size=size)
    return df


s1 = S1()
load_path = Path(f'../results/scenario_1/test/annotations')
save_path = Path(f'../filtered_results/scenario_1/test/annotations')
check_dir(save_path)

for sub in s1.test_subs:
    for vid in s1.test_vids:
        df = pd.read_csv(load_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
        avg_filter(df, size).to_csv(save_path / f'sub_{sub}_vid_{vid}.csv')


s2 = S2()
for fold in s2.fold:
    load_path = Path(f'../results/scenario_2/fold_{fold}/test/annotations')
    save_path = Path(f'../filtered_results/scenario_2/fold_{fold}/test/annotations')
    check_dir(save_path)
    for sub in s2.test_subs[fold]:
        for vid in s2.test_vids[fold]:
            df = pd.read_csv(load_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            avg_filter(df, size).to_csv(save_path / f'sub_{sub}_vid_{vid}.csv')

s3 = S3()
for fold in s3.fold:
    load_path = Path(f'../results/scenario_3/fold_{fold}/test/annotations')
    save_path = Path(f'../filtered_results/scenario_3/fold_{fold}/test/annotations')
    check_dir(save_path)
    for sub in s3.test_subs[fold]:
        for vid in s3.test_vids[fold]:
            df = pd.read_csv(load_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            avg_filter(df, size).to_csv(save_path / f'sub_{sub}_vid_{vid}.csv')


s4 = S4()
for fold in s4.fold:
    load_path = Path(f'../results/scenario_4/fold_{fold}/test/annotations')
    save_path = Path(f'../filtered_results/scenario_4/fold_{fold}/test/annotations')
    check_dir(save_path)
    for sub in s4.test_subs[fold]:
        for vid in s4.test_vids[fold]:
            df = pd.read_csv(load_path / f'sub_{sub}_vid_{vid}.csv', index_col='time')
            avg_filter(df, size).to_csv(save_path / f'sub_{sub}_vid_{vid}.csv')
