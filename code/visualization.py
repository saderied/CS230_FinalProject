import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

#data = pd.read_parquet("/Users/connorjanowiak/Documents/Stanford/CS230/data/itineraries.parquet", engine="auto")
data = pd.read_csv("/Users/saderied/Library/Mobile Documents/com~apple~CloudDocs/School/Fall 2024/CS 230/processed_itineraries.csv" )
# temporal_data = pd.read_hdf("/Users/saderied/Library/Mobile Documents/com~apple~CloudDocs/School/Fall 2024/CS 230/temporal_data.h5")
cols_to_corr = ['baseFare', 'isBasicEconomy', 'isRefundable', 'isNonStop', 'daysElapsed', 'isWeekend']

plt.figure(figsize=(12, 6))
plt.hist(data['baseFare'], bins=300, edgecolor='k', alpha=0.7)
plt.title("Distribution of Flight Prices")
plt.xlabel("Base Fare")
plt.ylabel("Frequency")
plt.xlim(0, 2000)
plt.show()
correlation_matrix = data[cols_to_corr].corr(numeric_only=True)

plt.figure(figsize=(12, 10))
plt.matshow(correlation_matrix, fignum=1, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='left')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation", fontsize=16, pad=20)
plt.show()

# Price trends over time
avg_prices_by_date = data.groupby('searchDate')['baseFare'].mean()
avg_prices_by_date.plot(figsize=(10, 6))
plt.title('Average Base Fare Over Time')
plt.xlabel('Search Date')
plt.ylabel('Average Base Fare')
plt.show()

# Geographic Analysis

# Reverse one-hot encoding for startingAirport
starting_airport_cols = [col for col in data.columns if col.startswith('startingAirport_')]
data['startingAirport'] = data[starting_airport_cols].idxmax(axis=1).str.replace('startingAirport_', '')

# Reverse one-hot encoding for destinationAirport
destination_airport_cols = [col for col in data.columns if col.startswith('destinationAirport_')]
data['destinationAirport'] = data[destination_airport_cols].idxmax(axis=1).str.replace('destinationAirport_', '')

avg_fares_by_airport = data.groupby('startingAirport')['baseFare'].mean().sort_values()
avg_fares_by_airport.plot(kind='bar', figsize=(12, 6))
plt.title('Average Base Fare by Starting Airport')
plt.xlabel('Starting Airport')
plt.ylabel('Average Base Fare')
plt.show()

# Pivot table for heatmap
fare_heatmap = data.pivot_table(index='startingAirport', columns='destinationAirport', values='baseFare', aggfunc='mean')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(fare_heatmap, annot=False, cmap='coolwarm', cbar=True)
plt.title('Average Base Fare Heatmap by Starting and Destination Airports')
plt.xlabel('Destination Airport')
plt.ylabel('Starting Airport')
plt.show()

# Single Leg Analysis
# sample_leg = temporal_data[temporal_data['legId'] == temporal_data['legId'].iloc[0]]
# plt.plot(sample_leg['searchDate'], sample_leg['baseFare'])
# plt.title('Base Fare Over Time for a Single Leg')
# plt.xlabel('Search Date')
# plt.ylabel('Base Fare')
# plt.show()

# Seat Availability
# sns.scatterplot(x=data['seatsRemaining'], y=data['baseFare'])
# plt.title('Base Fare vs. Seats Remaining')
# plt.xlabel('Seats Remaining')
# plt.ylabel('Base Fare')
# plt.show()

# Days of the week
data['flightDate'] = pd.to_datetime(data['flightDate'])
data['day_of_week'] = data['flightDate'].dt.day_name()
avg_fare_by_day = data.groupby('day_of_week')['baseFare'].mean().reset_index()

# Optional: Order days of the week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_fare_by_day['day_of_week'] = pd.Categorical(avg_fare_by_day['day_of_week'], categories=day_order, ordered=True)
avg_fare_by_day = avg_fare_by_day.sort_values('day_of_week')

# Bar plot of average base fare by day of the week
sns.barplot(x=avg_fare_by_day['day_of_week'], y=avg_fare_by_day['baseFare'])
plt.title('Average Base Fare by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Base Fare')
plt.xticks(rotation=45) 
plt.show()

# Days until departure
avg_fare_by_days = data.groupby('daysElapsed')['baseFare'].mean().reset_index()
sns.lineplot(x=avg_fare_by_days['daysElapsed'], y=avg_fare_by_days['baseFare'], marker='o')
plt.title('Average Base Fare vs. Days Elapsed')
plt.xlabel('Days Elapsed')
plt.ylabel('Average Base Fare')
plt.show()
