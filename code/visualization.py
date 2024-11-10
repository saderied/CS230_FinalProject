import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_parquet("/Users/connorjanowiak/Documents/Stanford/CS230/data/itineraries.parquet", engine="auto")

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
