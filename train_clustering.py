# train_clustering.py
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import pickle
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_FILE = os.path.join(OUT_DIR, "cases.csv")
MLB_FILE = os.path.join(BASE_DIR, "mlb_preprocessor.pkl")

# 1. Load your data
print(f"Loading data from {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print("Error: cases.csv not found. Please run app.py and log cases first.")
    exit()

df = pd.read_csv(CSV_FILE)

# 2. Preprocess the data (same as before)
def parse_list_string(s):
    if pd.isna(s) or s in ['[]', '']:
        return []
    return [item.strip().strip("'") for item in s.strip("[]").split(',')]

df['symptoms_list'] = df['symptoms'].apply(parse_list_string)
df['risks_list'] = df['risks'].apply(parse_list_string)
df['all_features'] = df['symptoms_list'] + df['risks_list']

# 3. Load the FITTED preprocessor
# We use the existing preprocessor so the feature columns are consistent
try:
    with open(MLB_FILE, 'rb') as f:
        mlb = pickle.load(f)
    print("Loaded existing preprocessor.")
    # Transform the new data
    X = mlb.transform(df['all_features'])
except Exception as e:
    print(f"Error loading {MLB_FILE}: {e}")
    print("Please run train_model.py first to create the preprocessor.")
    exit()

# 4. Train the Clustering Model
# We are NOT using 'level' or 'score'. This is unsupervised.
# Let's ask it to find 4 clusters.
k = 4
model = KMeans(n_clusters=k, random_state=42, n_init=10)
model.fit(X)

print(f"\n--- Clustering Model Trained (k={k}) ---")

# 5. Analyze and Display the Results
# This is the most important part. We look at what's in each cluster.

print("\n--- Top Features per Cluster ---")
# Get the "center" of each cluster
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# Get the feature names from our preprocessor
terms = mlb.classes_

for i in range(k):
    print(f"\nCluster {i}:")
    top_terms = [terms[ind] for ind in order_centroids[i, :5]] # Get top 5 terms
    print(f"  -> Top {len(top_terms)} terms: {', '.join(top_terms)}")

# 6. See how the clusters map to your *actual* levels
df['cluster'] = model.labels_
print("\n--- Crosstab: Clusters vs. Actual Triage Level ---")
crosstab = pd.crosstab(df['cluster'], df['level'])
print(crosstab)
print("\n--- END OF ANALYSIS ---")