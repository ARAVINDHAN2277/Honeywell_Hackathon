import pandas as pd

# Load the Excel file (first sheet by default)
df = pd.read_excel('flight.xlsx')

print('Columns:', df.columns.tolist())
print('\nFirst 5 rows:')
print(df.head())
print('\nInfo:')
df.info()
print('\nMissing values per column:')
print(df.isnull().sum())
