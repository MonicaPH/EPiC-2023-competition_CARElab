from enum import Enum
import pandas as pd
import neurokit2 as nk
from emg import emg_process

class Signal(Enum):
    ECG = 1
    PPG = 2
    EDA = 3
    RSP = 4
    EMG = 5

def processor(signal, type, sampling_rate=1000):
    if type == Signal.ECG:
        processed_signal, _ = nk.ecg_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.PPG:
        processed_signal, _ = nk.ppg_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.EDA:
        processed_signal, _ = nk.eda_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.RSP:
        processed_signal, _ = nk.rsp_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.EMG:
        processed_signal, _ = emg_process(signal, sampling_rate=sampling_rate)

    return processed_signal

def analyzer(signal, keypoints, type, window_size_past, window_size_future, sampling_rate=1000):
    df = None
    maximum = keypoints.max()
    for idx in keypoints:
        if idx - window_size_past < 0:
            onset = 0
            end = window_size_future / sampling_rate + idx / sampling_rate
        elif idx + window_size_future > maximum:
            onset = idx - window_size_past
            end = (maximum - idx) / sampling_rate + window_size_past / sampling_rate
        else:
            onset = idx - window_size_past
            end = (window_size_past + window_size_future + 1) / sampling_rate
        epoch = nk.epochs_create(signal, events=[onset], epochs_end=[end])
        
        if type == Signal.ECG:
            features = nk.ecg_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.PPG:
            features = nk.ppg_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.EDA:
            features = nk.eda_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.RSP:
            features = nk.rsp_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.EMG:
            try:
                features = nk.emg_analyze(epoch, sampling_rate=sampling_rate, method='event-related')
            except:
                epoch['1'].EMG_Onsets = 0
                features = nk.emg_analyze(epoch, sampling_rate=sampling_rate, method='event-related')
        
        df = features if df is None else pd.concat([df, features], axis=0)
    
    return df

def feature_extractor(X, y):
    signal = X['bvp']
    type = Signal.PPG
    processed_signal = processor(signal, type)
    bvp = analyzer(processed_signal, y.index, type, 5000, 5000)
    
    signal = X['ecg']
    type = Signal.ECG
    processed_signal = processor(signal, type)
    ecg = analyzer(processed_signal, y.index, type, 10000, 10000)
    
    signal = X['rsp']
    type = Signal.RSP
    processed_signal = processor(signal, type)
    rsp = analyzer(processed_signal, y.index, type, 4000, 4000)
    
    signal = X['gsr']
    type = Signal.EDA
    processed_signal = processor(signal, type)
    gsr = analyzer(processed_signal, y.index, type, 2500, 2500)

    emg_zygo, emg_coru, emg_trap = emg_feature_extractor(X, y)

    dropped_cols = ['Label', 'Event_Onset']
    
    return pd.concat([bvp.drop(dropped_cols, axis=1),
                      ecg.drop(dropped_cols, axis=1),
                      rsp.drop(dropped_cols, axis=1),
                      gsr.drop(dropped_cols, axis=1),
                      emg_zygo, emg_coru, emg_trap], axis=1)

def emg_feature_extractor(X, y):
    type = Signal.EMG
    signal = X['emg_zygo']
    processed_signal = processor(signal, type)
    emg_zygo = analyzer(processed_signal, y.index, type, 200, 200)
    signal = X['emg_coru']
    processed_signal = processor(signal, type)
    emg_coru = analyzer(processed_signal, y.index, type, 200, 200)
    signal = X['emg_trap']
    processed_signal = processor(signal, type)
    emg_trap = analyzer(processed_signal, y.index, type, 200, 200)


    dropped_cols = ['Label', 'Event_Onset']
    emg_zygo = emg_zygo.drop(dropped_cols, axis=1),
    emg_coru = emg_coru.drop(dropped_cols, axis=1),
    emg_trap = emg_trap.drop(dropped_cols, axis=1)

    emg_zygo.columns = list(map(lambda s: s + '_zygo', list(emg_zygo.columns)))
    emg_coru.columns = list(map(lambda s: s + '_coru', list(emg_coru.columns)))
    emg_trap.columns = list(map(lambda s: s + '_trap', list(emg_trap.columns)))

    return emg_zygo, emg_coru, emg_trap
