import os

if __name__ == '__main__':
    # extract features
    os.system('cd features && python3 features.py -s 1')
    os.system('cd features && python3 features.py -s 2')
    os.system('cd features && python3 features.py -s 3')
    os.system('cd features && python3 features.py -s 4')
