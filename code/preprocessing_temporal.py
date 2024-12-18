import pandas as pd
import numpy as np
import h5py

pd.set_option('future.no_silent_downcasting', True)

chunksize = 10_000_000
window_size = 14

data_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/processed_itineraries.csv"
output_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/temporal_data.h5"

# This sample data is only used to identify the features that we care about
# We only load 10 rows since the entire dataset is very large
sample_data = pd.read_csv(data_path, nrows=10)
feature_cols = sample_data.columns.to_list()
feature_cols.remove('baseFare')
feature_cols.remove('legId')
feature_cols.remove('flightDate')
feature_cols.remove('searchDate')
target_col = 'baseFare'


def create_sequences(data, group_col, feature_cols, target_col, window_size):
    x, y = [], []
    for _, group in data.groupby(group_col):
        group_values = group[feature_cols + [target_col]].values
        for i in range(len(group_values) - window_size):
            x.append(group_values[i:i + window_size, :-1])
            y.append(group_values[i + window_size, -1])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


with h5py.File(output_path, 'w') as hf:
    hf.create_dataset('x', shape=(0, window_size, len(feature_cols)), maxshape=(None, window_size, len(feature_cols)), dtype=np.float32, compression='gzip')
    hf.create_dataset('y', shape=(0,), maxshape=(None,), dtype=np.float32, compression='gzip')

for chunk_index, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
    print(f"Processing chunk {chunk_index + 1}", end='\r')
    
    # This is needed to account for the last chunk being smaller
    if chunk_index == 7:
        break
    
    chunk.dropna(inplace=True)
    chunk.replace({False: 0, True: 1}, inplace=True)
    chunk['searchDate'] = pd.to_datetime(chunk['searchDate'])
    chunk = chunk.sort_values(by=['legId', 'searchDate'])
    chunk['day_of_week'] = chunk['searchDate'].dt.dayofweek
    chunk['week_of_year'] = chunk['searchDate'].dt.isocalendar().week

    x_chunk, y_chunk = create_sequences(chunk, group_col='legId', feature_cols=feature_cols, target_col=target_col, window_size=window_size)
    with h5py.File(output_path, 'a') as hf:
        hf['x'].resize(hf['x'].shape[0] + x_chunk.shape[0], axis=0)
        hf['x'][-x_chunk.shape[0]:] = x_chunk
        hf['y'].resize(hf['y'].shape[0] + y_chunk.shape[0], axis=0)
        hf['y'][-y_chunk.shape[0]:] = y_chunk