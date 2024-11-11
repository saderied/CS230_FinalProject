import pandas as pd
import numpy as np
import h5py

pd.set_option('future.no_silent_downcasting', True)


data_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/itineraries.parquet"
data = pd.read_parquet(data_path)
data = data.head(20_000_000)
data = data.dropna()
data.replace({False: 0, True: 1}, inplace=True)

data = data.sort_values(by=['legId', 'searchDate'])

data['day_of_week'] = data['searchDate'].dt.dayofweek
data['week_of_year'] = data['searchDate'].dt.isocalendar().week


def create_sequences(data, group_col, feature_cols, target_col, window_size):
    X, y = [], []
    for _, group in data.groupby(group_col):
        group_values = group[feature_cols + [target_col]].values
        for i in range(len(group_values) - window_size):
            X.append(group_values[i:i+window_size, :-1])
            y.append(group_values[i+window_size, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

feature_cols = data.columns.to_list()
feature_cols.remove('baseFare')
feature_cols.remove('legId')
feature_cols.remove('flightDate')
feature_cols.remove('searchDate')
target_col = 'baseFare'
window_size = 14
X, y = create_sequences(data, group_col='legId', feature_cols=feature_cols, target_col=target_col, window_size=window_size)
with h5py.File('/Users/connorjanowiak/Documents/Stanford/CS230/data/temporal_data.h5', 'w') as hf:
    hf.create_dataset('X', data=X, compression='gzip')
    hf.create_dataset('y', data=y, compression='gzip')
