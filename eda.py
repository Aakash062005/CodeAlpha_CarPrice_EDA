import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("car_prices.csv")

# Dataset information
print(df.head())
print(df.info())
print(df.describe())

# -------------------------------
# 1. Histogram
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=20, kde=True)
plt.title("Car Price Distribution")
plt.show()

# -------------------------------
# 2. Scatter Plot
# -------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="EngineSize", y="Price", data=df)
plt.title("Engine Size vs Price")
plt.show()

# -------------------------------
# 3. Box Plot
# -------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Type", y="Price", data=df)
plt.title("Price by Car Type")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 4. Count Plot
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(x="Type", data=df)
plt.title("Car Type Count")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 6. Pair Plot
# -------------------------------
sns.pairplot(df[["Price","MPG.city","Horsepower","EngineSize"]])
plt.show()

print("EDA Completed Successfully!")