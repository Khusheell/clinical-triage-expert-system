import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os, ast

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_FILE = os.path.join(OUT_DIR, "cases.csv")

# --- We MUST use the same preprocessor as the classification model ---
MLB_FILE = os.path.join(BASE_DIR, "mlb_preprocessor.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "regression_model.pkl")

# 1. Load data
print(f"Loading data from {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found.")
    print("Please run app.py and log at least 10-15 cases first.")
    exit()

df = pd.read_csv(CSV_FILE)

# --- Check for the 'score' column ---
if 'score' not in df.columns:
    print("Error: The 'score' column is missing from cases.csv.")
    print("This means your app.py or triage_kb.pl is not the latest version.")
    exit()

# 2. Preprocess data
# --- THIS IS THE BUG FIX ---
def parse_list_string(s):
    try:
        # ast.literal_eval is the safe and correct way to parse "['a','b']"
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
# --- END OF FIX ---

df['symptoms_list'] = df['symptoms'].apply(parse_list_string)
df['risks_list'] = df['risks'].apply(parse_list_string)
df['all_features'] = df['symptoms_list'] + df['risks_list']

# --- Handle NaN scores ---
df.dropna(subset=['score'], inplace=True)
if df.empty:
    print("Error: No data with valid scores to train on.")
    exit()

# 3. Load the FITTED preprocessor (mlb)
print(f"Loading preprocessor from {MLB_FILE}...")
if not os.path.exists(MLB_FILE):
    print(f"Error: {MLB_FILE} not found.")
    print("Please run train_model.py FIRST to create the preprocessor.")
    exit()

with open(MLB_FILE, 'rb') as f:
    mlb = pickle.load(f)

# Transform features using the *loaded* preprocessor
X = mlb.transform(df['all_features'])
y = df['score']

print(f"Found {X.shape[1]} total features.")

if len(df) < 10:
    print(f"Warning: Only {len(df)} cases found. Model may be inaccurate.")
    print("Please log more cases in the app.")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if len(y_test) == 0:
    print("Error: Not enough data to create a test set. Please log more cases.")
    exit()

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model trained with R-squared on test set: {r2:.2f} (closer to 1.0 is better)")
    
# 7. Save the new regression model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print(f"Regression model saved to {MODEL_FILE}")