# introducing the team

Our team is formed by two information science master students, and three professors at different levels. Our specialties are on machine learning, signal processing, and affective research. We are a multinational team, our institutions are located in Japan and The Netherlands. We come from five different countries. Finally, we are keen to understand better the relationship between bodily changes and subjective experience.

## Wei Xin

## Huakun Liu

## Felix Dollack

## Chirag Raman

## Hideaki Uchiyama

## Kiyoshi Kiyokawa

## Monica Perusquia-Hernandez


# explaining your approach

We used a theoretical assumption approach to train multiple weak classifiers, together with late fusion. This approach was done independently for each scenario. General signal preprocessing was done in advance. As an output layer in the algorithm pipeline, we applied a low-pass filter to remove sudden annotation variations that are unlikely to happen.

# describing the repository content
The repository is organized in folders as follows:
- **CARElab** : exploratory analysis notebooks
- **data** : raw data provided for the challenge
- **features** : data cleaning and feature extraction
- **io_data** : input and output data generation
- **models** : model training
- **results** : solution data files for the challenge submission
- **src** : util scripts

# how to run the code

## Challenge
To reproduce the submitted challenge results:
1. features/features.py -> for generating clean signals and features
3. io_data/*.py -> for generating the specific input for each scenario
4. models/*.py -> for training the models
5. results/*.py -> for predicting the results

## Delay analysis
For reporduction of the data used in the delay plots execute the following code in order:
```
cd CARElab
python3 train_LSTM_lag_models.py
python3 test_LSTM_lag_models.py
```
This wil produce CSV files in the form of `performance_XXX.csv`, where XXX is either a short for the physiological signal used (e.g. ecg) or `all` if all signals were used as input.
