import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/fifa_wc_knockout.csv")

# Show Dataset
print("=== FIFA World Cup Finals Dataset ===")
print(df)

# Dataset Info
print("\n=== Dataset Info ===")
print(df.info())

# Statistics
print("\n=== Statistics ===")
print(df.describe())

# Missing Values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# EDA
print("\n=== Winners Count ===")
winner_count = df["winner"].value_counts()
print(winner_count)

# Chart 1 - Finals Goals by Year
plt.figure(figsize=(8,5))
plt.plot(df["year"], df["total_goals"], marker='o')
plt.xlabel("Year")
plt.ylabel("Total Goals")
plt.title("Goals in FIFA World Cup Finals")
plt.savefig("output/chart1_goals.png")
plt.show()

# Chart 2 - Winner Count
winner_count.plot(kind="bar")
plt.xlabel("Country")
plt.ylabel("Titles")
plt.title("World Cup Titles (1970-2022)")
plt.savefig("output/chart2_winners.png")
plt.show()

# Chart 3 - Penalty Shootout Count

penalty_count = df["penalty_shootout"].value_counts()
penalty_count.plot(kind="bar")
plt.title("Penalty Shootouts in Finals")
plt.savefig("output/chart3_penalty.png")
plt.show()

# Chart 4 - Heatmap
plt.figure(figsize=(5,4))
corr = df[["year", "total_goals"]].corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("output/chart4_heatmap.png")
plt.show()

# Chart 5 - Pie Chart
plt.figure(figsize=(7,7))
winner_count.plot(kind="pie", autopct='%1.1f%%')
plt.ylabel("")
plt.title("World Cup Winners Distribution")
plt.savefig("output/chart5_pie.png")
plt.show()

# Machine Learning - Predict total goals from year
X = df[["year"]]
y = df["total_goals"]

model = LinearRegression()
model.fit(X, y)

print("\n=== Machine Learning Model ===")
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

# Prediction
prediction = model.predict([[2026]])
print("\nPredicted total goals in 2026 final:")
print(round(prediction[0], 2))

