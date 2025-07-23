import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Load iris dataset
df = pd.read_csv("iris.csv")

# Scatter plot of two features colored by species

sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=df)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# Distribution of petal length

sns.histplot(df["petal_length"], bins=30)
plt.title("Distribution of Petal Length")
plt.show()

# Box plot to compare sepal length across species

sns.boxplot(x="species", y="sepal_length", data=df)
plt.title("Sepal Length by Species")
plt.show()

# Correlation heatmap

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()



