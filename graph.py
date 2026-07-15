import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("iris.csv")

# Style
sns.set(style="whitegrid")

# 1. Pair Plot
sns.pairplot(data, hue="Species")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_data = data.drop(columns=["Species"])
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(data=data.drop(columns=["Id","Species"]))
plt.title("Box Plot of Features")
plt.show()

# 4. Histogram
data.drop(columns=["Id","Species"]).hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()