import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('raw_data.csv')
print("Raw Data:")
print(df)

imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])  

scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nCleaned & Transformed Data:")
print(df)

df.to_csv('cleaned_data.csv', index=False)
print("\nETL completed successfully. File 'cleaned_data.csv' created.")
