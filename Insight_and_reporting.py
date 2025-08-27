from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('clean_data.csv')

# Log transfrom skewed feature
df['log_feature'] = df['feature1'].apply(lambda x: np.log1p(x))

# One-hot encode categorical feature
df_encoded = pd.get_dummies(df, columns=['category'])

# Select numeric columns only for scaling (excluding the target 'score')
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('score') # Remove target colum from features to scale

# Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded[numeric_cols])

print(df_scaled[:5]) # Show sample scaled features