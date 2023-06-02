import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_path = 'C:/Users/Wasim/Documents/Universita/Magistrale/Tesi/workspace/ppg_feature_extraction/dataset_extracted'
# data_path = './workspace/ppg_feature_extraction/dataset_extracted'

# Loading the dataset
dataset1 = pd.read_csv(f'{data_path}/dataset_part{1}.csv')
dataset2 = pd.read_csv(f'{data_path}/dataset_part{2}_rest.csv')
dataset3 = pd.read_csv(f'{data_path}/dataset_part{3}.csv')
dataset4 = pd.read_csv(f'{data_path}/dataset_part{4}.csv')

dataset = pd.concat([dataset2])
print(f'dataset Data type: {type(dataset)}')
print(f'dataset shape/dimensions: {dataset.shape}')

features_to_exclude = []
features_to_exclude = ['st10', 'st25', 'st33', 'st50', 'st66', 'st75']
# features_to_exclude.extend(['st10_p_dt10'])
# features_to_exclude.extend(['st25_p_dt25'])
# features_to_exclude.extend(['st33_p_dt33'])
# features_to_exclude.extend(['st50_p_dt50'])
# features_to_exclude.extend(['st66_p_dt66'])
# features_to_exclude.extend(['st75_p_dt75'])
dataset = dataset.loc[:, ~dataset.columns.isin(features_to_exclude)]

X = dataset.iloc[0:, 4:].to_numpy()
y = dataset.iloc[0:, 0:4].to_numpy()

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True)

print(f'Train dataset : {y_train.shape[0]}')
print(f'Validation dataset : {y_val.shape[0]}')
print(f'Test dataset : {y_test.shape[0]}')

y_train_ids = y_train[0:, 0]
y_val_ids = y_val[0:, 0]
y_test_ids = y_test[0:, 0]

y_train = y_train[0:, 2:]
y_val = y_val[0:, 2:]
y_test = y_test[0:, 2:]

np.savetxt('./y_train_ids.csv', y_train_ids, delimiter=',')
np.savetxt('./y_val_ids.csv', y_val_ids, delimiter=',')
np.savetxt('./y_test_ids.csv', y_test_ids, delimiter=',')

np.savetxt('./y_train.csv', y_train, delimiter=',')
np.savetxt('./X_train.csv', X_train, delimiter=',')
np.savetxt('./y_val.csv', y_val, delimiter=',')
np.savetxt('./X_val.csv', X_val, delimiter=',')
np.savetxt('./y_test.csv', y_test, delimiter=',')
np.savetxt('./X_test.csv', X_test, delimiter=',')
