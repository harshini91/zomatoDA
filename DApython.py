# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🗂 Load Dataset
df = pd.read_csv(r"F:\archive (4)\zomato.csv", encoding='latin1')

# Rename columns FIRST
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# 🧹 Clean Data
df.drop(columns=["unnamed_0", "unnamed_0.1"], inplace=True, errors='ignore')
df.dropna(subset=["rate_out_of_5", "avg_cost_two_people"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Rename columns
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# 🎨 Set Visualization Style
sns.set(style="whitegrid")

# 📊 Plot 1: Rating Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['rate_out_of_5'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 📊 Plot 2: Cost vs Rating
plt.figure(figsize=(10, 5))
sns.scatterplot(x='avg_cost_two_people', y='rate_out_of_5', data=df, alpha=0.6)
plt.title("Cost for Two vs Rating")
plt.xlabel("Average Cost for Two")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# 📊 Plot 3: Online Order Count
plt.figure(figsize=(6, 4))
sns.countplot(x='online_order', data=df, palette='pastel')
plt.title("Restaurants Offering Online Order")
plt.xlabel("Online Order")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 📊 Plot 4: Top Cuisines
top_cuisines = df['cuisines_type'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette="viridis")
plt.title("Top 10 Cuisine Types")
plt.xlabel("Number of Restaurants")
plt.ylabel("Cuisine Type")
plt.tight_layout()
plt.show()

# 📊 Plot 5: Top Restaurant Areas
top_areas = df['area'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y=top_areas.index, x=top_areas.values, palette="magma")
plt.title("Top 10 Areas with Most Restaurants")
plt.xlabel("Number of Restaurants")
plt.ylabel("Area")
plt.tight_layout()
plt.show()

# 📦 Import Required Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 🧹 Preprocess Data
df_model = df.copy()

# Encode categorical columns
categorical_cols = ['restaurant_type', 'online_order', 'table_booking', 'area']
le = LabelEncoder()
for col in categorical_cols:
    df_model[col] = le.fit_transform(df_model[col])

# 🧮 Define Features and Target
X = df_model[['restaurant_type', 'num_of_ratings', 'avg_cost_two_people', 'online_order', 'table_booking', 'area']]
y = df_model['rate_out_of_5']

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔍 Predict
y_pred = model.predict(X_test)

# 📈 Evaluate the Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")


# 📦 Import Required Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 🧹 Prepare Data for Clustering
df_cluster = df[['rate_out_of_5', 'num_of_ratings', 'avg_cost_two_people', 'online_order']].copy()

# Convert online_order to binary
df_cluster['online_order'] = df_cluster['online_order'].map({'Yes': 1, 'No': 0})

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# 📊 Elbow Method to Determine Optimal K
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('SSE (Inertia)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 💡 Apply KMeans with Chosen K (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)

# 🎨 Visualize Segments (2D)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_cost_two_people', y='rate_out_of_5', hue='segment', palette='Set2')
plt.title("Customer Segments: Cost vs Rating")
plt.xlabel("Average Cost for Two")
plt.ylabel("Rating")
plt.legend(title='Segment')
plt.tight_layout()
plt.show()


