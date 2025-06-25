# zomato_dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load & Clean Data
df = pd.read_csv("zomato.csv", encoding='latin1')
df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
df.dropna(subset=["rate (out of 5)", "avg cost (two people)"], inplace=True)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df.reset_index(drop=True, inplace=True)

# ✅ Choose Filters
selected_areas = ['Indiranagar', 'Koramangala 5th Block', 'Bannerghatta Road']
selected_cuisines = ['North Indian', 'Chinese', 'South Indian']

# 🔹 Filter the Data
filtered_df = df[df['area'].isin(selected_areas) & df['cuisines_type'].isin(selected_cuisines)]

# 🎨 1. Rating Distribution
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df['rate_out_of_5'], bins=30, kde=True, color='skyblue')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("1_rating_distribution.png")
plt.show()

# 🎨 2. Online Order Bar Chart
plt.figure(figsize=(6, 4))
sns.countplot(data=filtered_df, x='online_order', palette='pastel')
plt.title("Online Order Availability")
plt.xlabel("Online Order")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("2_online_order.png")
plt.show()

# 🎨 3. Top Areas by Restaurant Count
plt.figure(figsize=(10, 5))
top_areas = filtered_df['area'].value_counts().head(10)
sns.barplot(x=top_areas.index, y=top_areas.values, palette='magma')
plt.xticks(rotation=45)
plt.title("Top Areas by Restaurant Count")
plt.xlabel("Area")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("3_top_areas.png")
plt.show()

# 🎨 4. Top Cuisines
plt.figure(figsize=(10, 5))
top_cuisines = filtered_df['cuisines_type'].value_counts().head(10)
sns.barplot(x=top_cuisines.index, y=top_cuisines.values, palette='viridis')
plt.xticks(rotation=45)
plt.title("Top 10 Cuisines")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("4_top_cuisines.png")
plt.show()

# 🎨 5. Cost vs Rating Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_df, x='avg_cost_two_people', y='rate_out_of_5', hue='online_order')
plt.title("Cost vs Rating by Online Order")
plt.xlabel("Average Cost for Two")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("5_cost_vs_rating.png")
plt.show()
