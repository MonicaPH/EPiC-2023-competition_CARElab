import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings

warnings.filterwarnings("ignore")

def test(X, a_model_path, v_model_path, saved_path, late_fusion=False):
    """
        if late_fusion is True, model_path should be a list.
    """
    test_data = TabularDataset(X)

    if late_fusion == True:
        arousal = []
        valence = []
        for one_model_path in a_model_path:
            predictor_arousal = TabularPredictor.load(str(one_model_path))
            arousal.append(predictor_arousal.predict(test_data))
        for one_model_path in v_model_path:
            predictor_valence = TabularPredictor.load(str(one_model_path))
            valence.append(predictor_valence.predict(test_data))
        arousal = pd.concat(arousal, axis=1).mean(axis=1)
        valence = pd.concat(valence, axis=1).mean(axis=1)

    else:
        predictor_arousal = TabularPredictor.load(str(a_model_path))
        arousal = predictor_arousal.predict(test_data)

        predictor_valence = TabularPredictor.load(str(v_model_path))
        valence = predictor_valence.predict(test_data)

    predictions = pd.DataFrame({'valence': valence, 'arousal': arousal})
    predictions.to_csv(saved_path)
