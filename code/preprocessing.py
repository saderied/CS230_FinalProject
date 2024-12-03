import pandas as pd
import numpy as np

csv_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/itineraries.csv"  
output_csv_path = "/Users/connorjanowiak/Documents/Stanford/CS230/data/processed_itineraries.csv"

def preprocess_flight_data(csv_path, output_parquet_path, chunksize=100000):
    columns_to_keep = [
        "legId", "searchDate", "flightDate", "startingAirport", 
        "destinationAirport", "travelDuration", "isBasicEconomy", 
        "isRefundable", "isNonStop", "baseFare", "seatsRemaining", 
        "totalTravelDistance", "segmentsAirlineName", "segmentsCabinCode",
        "segmentsDepartureAirportCode"
    ]
    
    processed_data = []
    cur_chunk = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        cur_chunk += 1
        print("Processing chunk: " + str(cur_chunk), end='\r')
        chunk = chunk[columns_to_keep]
        chunk['travelDuration'] = chunk['travelDuration'].str.extract(r'PT(\d+)H(\d+)M').fillna(0).astype(int).apply(
            lambda x: x[0] * 60 + x[1], axis=1)
        
        chunk['searchDate'] = pd.to_datetime(chunk['searchDate'])
        chunk['flightDate'] = pd.to_datetime(chunk['flightDate'])
        chunk['daysElapsed'] = (chunk['flightDate'] - chunk['searchDate']).dt.days
        
        chunk['isWeekend'] = chunk['flightDate'].dt.dayofweek.isin([5, 6])
        chunk['totalStops'] = chunk['segmentsDepartureAirportCode'].str.count(r'\|\|')
        chunk['airline'] = chunk['segmentsAirlineName'].str.split('||').str[0]
        chunk['cabinCode'] = chunk['segmentsCabinCode'].str.split('||').str[0]
        categorical_cols = ['startingAirport', 'destinationAirport', 'airline', 'cabinCode']
        chunk = pd.get_dummies(chunk, columns=categorical_cols, drop_first=True)
        chunk.drop(["segmentsAirlineName", "segmentsCabinCode", "segmentsDepartureAirportCode"], axis=1, inplace=True)
        processed_data.append(chunk)
    
    final_df = pd.concat(processed_data, ignore_index=True)
    final_df.to_csv(output_parquet_path)

preprocess_flight_data(csv_path, output_csv_path)
