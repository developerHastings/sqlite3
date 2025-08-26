import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load cleaned data
df = pd.read_csv('clean_data.csv')

# Statistical summaries
print(df.describe())

# Histogram of a feature
sns.histplot(df['feature1'], kde=True)
plt.show()

# Boxplot for outlier detection
sns.boxplot(x=df['feature2'])
plt.show()

# Scatter plot and correlation heatmap
sns.scatterplot(data=df, x='feature1', y='feature2')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()