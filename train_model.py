import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os, ast

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CSV_FILE = os.path.join(OUT_DIR, "cases.csv")
MODEL_FILE = os.path.join(BASE_DIR, "triage_model.pkl")
MLB_FILE = os.path.join(BASE_DIR, "mlb_preprocessor.pkl")

# 1. Load data
print(f"Loading data from {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print(f"Error: {CSV_FILE} not found.")
    print("Please run app.py and log at least 10-15 cases first.")
    exit()

df = pd.read_csv(CSV_FILE)

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

# 2. Preprocess data
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['all_features'])
y = df['level'] 

print(f"Found {X.shape[1]} total features (symptoms/risks).")

# 3. Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if len(y_test) == 0:
    print("Error: Not enough data to create a test set. Please log more cases.")
    exit()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"Model trained with accuracy on test set: {model.score(X_test, y_test):.2f}")

# 4. Save the model AND the preprocessor
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_FILE}")

with open(MLB_FILE, 'wb') as f:
    pickle.dump(mlb, f)
print(f"Preprocessor saved to {MLB_FILE}")