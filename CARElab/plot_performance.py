import argparse
from matplotlib import pylab as plt
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Performance CSV file')
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    sensor = ''.join(args.file.split('_')[1:]).split('.')[0]

    plt.plot(df.offset/1000, df.rmse, '-o')
    plt.ylabel('RMSE')
    plt.xlabel('Delay [s]')
    plt.grid()
    plt.title(f'{sensor.upper()} Signals')
    plt.savefig(f'performance_{sensor}.png')
