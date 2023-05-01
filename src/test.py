import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings

warnings.filterwarnings("ignore")

def test(X, model_path, saved_path, late_fusion=False):
    """
        if late_fusion is True, model_path should be a list.
    """
    test_data = TabularDataset(X)

    if late_fusion:
        arousal = []
        valence = []
        for one_model_path in model_path:
            predictor_arousal = TabularPredictor.load(str(one_model_path) + '_arousal')
            arousal.append(predictor_arousal.predict(test_data))

            predictor_valence = TabularPredictor.load(str(one_model_path) + '_valence')
            valence.append(predictor_valence.predict(test_data))
        arousal = pd.concat(arousal, axis=1).mean(axis=1)
        valence = pd.concat(valence, axis=1).mean(axis=1)

    else:
        predictor_arousal = TabularPredictor.load(str(model_path) + '_arousal')
        arousal = predictor_arousal.predict(test_data)

        predictor_valence = TabularPredictor.load(str(model_path) + '_valence')
        valence = predictor_valence.predict(test_data)

    predictions = pd.DataFrame({'valence': valence, 'arousal': arousal})
    predictions.to_csv(saved_path)
