# 📦 Required Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📂 Load & Clean Data
df = pd.read_csv(r"F:\archive (4)\zomato.csv", encoding='latin1')
df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True, errors='ignore')
df.dropna(subset=["rate (out of 5)", "avg cost (two people)"], inplace=True)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Clean restaurant names for matching
df['restaurant_name'] = df['restaurant_name'].astype(str).str.strip().str.replace("'", "").str.lower()
df.reset_index(drop=True, inplace=True)

# 🧩 Combine Text Fields for Content Matching
df['combined_features'] = df['cuisines_type'].fillna('') + " " + df['restaurant_type'].fillna('')

# 🔍 TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# 🔗 Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🔁 Restaurant Name Index Mapping
indices = pd.Series(df.index, index=df['restaurant_name']).drop_duplicates()

# 🧠 Recommendation Function
def get_recommendations(name, cosine_sim=cosine_sim):
    # Clean input name for matching
    name_clean = name.strip().replace("'", "").lower()
    if name_clean not in indices:
        return "Restaurant not found."
    
    idx = indices[name_clean]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar
    
    restaurant_indices = [i[0] for i in sim_scores]
    # Show original names (with formatting) in output
    return df.loc[restaurant_indices, ['restaurant_name', 'cuisines_type', 'rate_out_of_5']]

# 🔍 Example Usage
print("🔎 Recommendations for '@ Biryani Central':")
print(get_recommendations("@ Biryani Central"))