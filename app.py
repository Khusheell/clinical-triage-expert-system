import os, csv, uuid, json, ast, random
from datetime import datetime
import pickle
from flask import Flask, render_template, request, redirect, url_for
from pyswip import Prolog
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

try:
    from pyswip.prolog import PrologError
except Exception:
    class PrologError(Exception):
        pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUT_DIR, "cases.csv")

app = Flask(__name__, template_folder="templates", static_folder="static")
prolog = Prolog()
prolog.consult(os.path.join(BASE_DIR, "triage_kb.pl"))

# --- LOAD ALL ML MODELS ---
MODEL_FILE = os.path.join(BASE_DIR, "triage_model.pkl")
MLB_FILE = os.path.join(BASE_DIR, "mlb_preprocessor.pkl")
REGRESSION_MODEL_FILE = os.path.join(BASE_DIR, "regression_model.pkl") 

try:
    with open(MODEL_FILE, 'rb') as f:
        ml_model = pickle.load(f)
    print("Classification ML model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Classification model. Run train_model.py. {e}")
    ml_model = None

try:
    with open(MLB_FILE, 'rb') as f:
        mlb_preprocessor = pickle.load(f)
    print("Preprocessor (mlb_preprocessor.pkl) loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Preprocessor. Run a training script first. {e}")
    mlb_preprocessor = None

try:
    with open(REGRESSION_MODEL_FILE, 'rb') as f:
        ml_regression_model = pickle.load(f) 
    print("Regression ML model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Regression model. Run train_regression_model.py. {e}")
    ml_regression_model = None
# -----------------------------------------------

SYMPTOMS = [
    "chest_pain","shortness_of_breath","fever","cough","arm_pain",
    "nausea","sudden_weakness","confusion","vision_loss","loss_of_taste"
]
RISK_FACTORS = ["smoker","elderly","diabetes"]

def to_pl_list(py_list):
    if not py_list:
        return "[]"
    return "[" + ",".join("'" + str(x) + "'" for x in py_list) + "]"

def parse_symptoms_from_request(req):
    txt = req.form.get("symptoms_text", "").strip()
    if txt:
        items = [s.strip().replace(" ", "_") for s in txt.replace("\n", ",").split(",") if s.strip()]
        return items
    return [s for s in req.form.getlist("symptom") if s]

def parse_risks_from_request(req):
    txt = req.form.get("risks_text", "").strip()
    if txt:
        items = [r.strip() for r in txt.replace("\n", ",").split(",") if r.strip()]
        return items
    return [r for r in req.form.getlist("risk") if r]

# --- MODIFIED: diagnose function ---
def diagnose(symptoms, risks):
    try:
        list(prolog.query("retractall(patient_symptom(_))."))
    except Exception:
        pass
    for s in symptoms:
        atom = s.replace(" ", "_")
        try:
            prolog.assertz(f"patient_symptom('{atom}')")
        except Exception:
            pass
            
    # We now query for 6 arguments, adding Score (S)
    q = list(prolog.query(f"find_triage_level({to_pl_list(symptoms)}, {to_pl_list(risks)}, L, E, B, S).", maxresult=1))
    
    if not q:
        return {"level":"Unknown","explanation":"No match","breakdown":[], "score": 0}
    r = q[0]
    breakdown = []
    if 'B' in r:
        try:
            breakdown = [str(item) for item in r['B']]
        except Exception:
            breakdown = [str(r['B'])]
            
    return {
        "level": r.get('L','Unknown'), 
        "explanation": r.get('E',''), 
        "breakdown": breakdown,
        "score": r.get('S', 0) 
    }
# ---------------------------------

def get_actions(node):
    out = []
    try:
        for sol in prolog.query(f"action({node}, To, Label, Cost)."):
            out.append((str(sol['To']), str(sol['Label']), int(sol['Cost'])))
    except Exception:
        pass
    return out

def heuristic(node, goal):
    try:
        q = list(prolog.query(f"heuristic({node},{goal},H).", maxresult=1))
        if q and 'H' in q[0]:
            return float(q[0]['H'])
    except Exception:
        pass
    return 1.0
    
def a_star(start, goal):
    frontier = [(0.0, start, [])]
    best = {}
    while frontier:
        frontier.sort(key=lambda x: x[0])
        cost, node, path = frontier.pop(0)
        if node == goal:
            return {"path": path + [node], "cost": cost}
        if node in best and best[node] <= cost:
            continue
        best[node] = cost
        for (to, label, c) in get_actions(node):
            new_cost = cost + c
            est = heuristic(to, goal)
            frontier.append((new_cost + est, to, path + [node]))
    return {"path": [], "cost": None}

def ao_star(root):
    nodes = {}
    edges = {}
    ao_costs = {}
    try:
        for sol in prolog.query("node(N, Type, C)."):
            nodes[str(sol['N'])] = str(sol['Type'])
            ao_costs[str(sol['N'])] = float(sol['C'])
    except Exception:
        pass
    try:
        for sol in prolog.query("edge(From, L)."):
            from_node = str(sol['From'])
            raw = sol['L']
            s = str(raw)
            childs = []
            if s.startswith('[') and s.endswith(']'):
                inner = s[1:-1].strip()
                if inner:
                    childs = [x.strip() for x in inner.split(',')]
            else:
                childs = [s]
            edges[from_node] = childs
    except Exception:
        pass

    def leaf_cost(n):
        try:
            q = list(prolog.query(f"ao_heuristic({n}, C).", maxresult=1))
            if q and 'C' in q[0]:
                return float(q[0]['C'])
        except Exception:
            pass
        return float(ao_costs.get(n, 1.0))

    from functools import lru_cache
    @lru_cache(None)
    def solve(n):
        if n not in edges or not edges[n]:
            return ([n], leaf_cost(n))
        typ = nodes.get(n, 'or')
        best_nodes = None
        best_cost = float('inf')
        for alt in edges[n]:
            if alt.startswith('[') and alt.endswith(']'):
                inner = alt[1:-1].strip()
                children = [c.strip() for c in inner.split(',')] if inner else []
            else:
                children = [alt]
            total_nodes = [n]
            total_cost = 0.0
            for child in children:
                child_nodes, child_cost = solve(child)
                total_nodes += child_nodes
                total_cost += child_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_nodes = total_nodes
        return (best_nodes, best_cost)

    nodes_res, cost = solve(root)
    return {"path": nodes_res or [], "cost": cost}


# --- *** THIS IS THE FINAL BUG FIX for CSV *** ---
def write_case_csv(row):
    header = ["when_iso","case_id","level","symptoms","risks", "score"]
    exists = os.path.exists(CSV_FILE)
    
    # --- THIS IS THE FIX ---
    # Handle empty lists correctly to avoid saving "['']"
    # This now writes "[]" for empty lists
    symptoms_str = "[]" if not row[3] else "['" + "','".join(row[3]) + "']"
    risks_str = "[]" if not row[4] else "['" + "','".join(row[4]) + "']"
    # --- END OF FIX ---

    row_to_write = [
        row[0], 
        row[1], 
        row[2], 
        symptoms_str,  # The new correct string
        risks_str,     # The new correct string
        row[5]         # The score
    ]
    
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row_to_write)
# -------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", symptoms=SYMPTOMS, risks=RISK_FACTORS, result=None)


# --- MODIFIED: run_diagnosis function ---
@app.route("/diagnose", methods=["POST"])
def run_diagnosis():
    symptoms = parse_symptoms_from_request(request)
    risks = parse_risks_from_request(request)
    
    ml_level = "N/A"
    ml_score = "N/A"
    messages = []

    # 1. GET PROLOG PREDICTION (level AND score)
    result = diagnose(symptoms, risks) 
    case_id = uuid.uuid4().hex[:6]
    level = result.get("level", "Unknown")
    pl_score = result.get("score", 0) 

    # 2. GET ML PREDICTIONS (Classification and Regression)
    if mlb_preprocessor:
        try:
            user_features = symptoms + risks
            transformed_input = mlb_preprocessor.transform([user_features])
            
            if ml_model:
                prediction = ml_model.predict(transformed_input)
                ml_level = prediction[0]
            else:
                ml_level = "N/A"
                messages.append("Classification model not loaded.")

            if ml_regression_model:
                score_prediction = ml_regression_model.predict(transformed_input)
                ml_score = f"{score_prediction[0]:.2f}" 
            else:
                ml_score = "N/A"
                messages.append("Regression model not loaded.")

        except Exception as e:
            messages.append(f"ML model error: {e}")
            ml_level = "Error"
            ml_score = "Error"
    else:
            messages.append("ML preprocessor not loaded. Run a training script.")
            ml_level = "N/A"
            ml_score = "N/A"
    
    # 3. LOG THE CASE (with the new score)
    try:
        pl_sym_list = to_pl_list(symptoms)
        pl_risks_list = to_pl_list(risks)
        prolog.assertz(
            f"case_log('{case_id}','{datetime.utcnow().isoformat()}','{level}',{pl_sym_list},{pl_risks_list})"
        )
    except Exception:
        try:
            prolog.query(
                f"log_case('{case_id}','{datetime.utcnow().isoformat()}','{level}',{pl_sym_list},{pl_risks_list})."
            )
        except Exception:
            pass 

    # Pass the new 'pl_score' to the CSV writer
    write_case_csv(
        [datetime.utcnow().isoformat(), case_id, level, symptoms, risks, pl_score]
    )

    if level == "Emergency":
        tips = "⚠️ Immediate medical attention required. Call emergency services..."
    elif level == "Urgent":
        tips = "🩺 Seek medical advice soon. Visit a healthcare provider within 24 hours..."
    elif level == "Routine":
        tips = "✅ Routine check-up suggested. Stay hydrated, rest well..."
    else:
        tips = "ℹ️ No specific recommendations available."
        
    breakdown = result.get("breakdown", [])
    if not breakdown:
        if level == "Emergency":
             breakdown = ["Critical symptom detected (e.g., chest_pain)."]
        else:
             breakdown = ["No detailed score available."]

    ui_result = {
        "level": level,
        "explanation": result.get("explanation", ""),
        "breakdown": breakdown,
        "tips": tips,
        "score": pl_score 
    }

    return render_template(
        "index.html",
        symptoms=SYMPTOMS,
        risks=RISK_FACTORS,
        result=ui_result,        
        ml_level=ml_level,      
        ml_score=ml_score,      
        messages=messages,
        astar={},
        aostar={},
    )
# -------------------------------------


@app.route("/add", methods=["GET","POST"])
def add_entry():
    message = None
    if request.method == "POST":
        message = "Entry added (UI placeholder)."
    return render_template("add_entry.html", message=message)

@app.route("/stats")
def stats_page():
    counts = {"Emergency":0,"Urgent":0,"Routine":0,"Unknown":0}
    rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] != "when_iso":
                    rows.append(row)
                    lvl = row[2] if len(row)>2 else "Unknown"
                    counts[lvl] = counts.get(lvl,0) + 1
    return render_template("stats.html", counts=counts, rows=rows)

@app.route("/astar")
def astar():
    start = request.args.get("start")
    goal = request.args.get("goal")
    path, cost, log = None, None, [] # log will now be filled by Prolog

    if start and goal:
        try:
            # UPDATED: Query for the 5-argument predicate a_star/5
            q = f"a_star('{start}', '{goal}', Path, Cost, Log)"
            
            # Use next() to get the single, best solution
            sol = next(prolog.query(q))
            
            # Process the results from Prolog
            path = [str(p) for p in sol["Path"]]
            cost = sol["Cost"]
            log = [str(l) for l in sol["Log"]] # <-- GET THE LOG
            
        except Exception as e:
            log = [f"Error running query: {e}", f"Query: {q}"]

    return render_template(
        "astar.html",
        path=path,
        cost=cost,
        log=log, # <-- Pass the full exploration log to the template
        ao_solution=None,
        ao_cost=None,
        ao_steps=None,
    )
@app.route("/aostar")
def aostar():
    root = request.args.get("root")
    ao_solution, ao_cost, ao_steps = None, None, None
    if root:
        try:
            q = f"ao_star('{root}', Solution, Cost, Steps)"
            for sol in prolog.query(q):
                ao_solution = sol["Solution"]
                ao_cost = sol["Cost"]
                ao_steps = sol["Steps"]
                break
        except Exception as e:
            ao_solution = [f"Error: {str(e)}"]
    return render_template(
        "astar.html",
        ao_solution=ao_solution,
        ao_cost=ao_cost,
        ao_steps=ao_steps,
        path=None,
        cost=None,
        log=None,
    )
    
@app.route("/total_diseases", methods=["GET", "POST"])
def total_diseases_page():
    term = ""
    total = None
    messages = []
    diagnosis_list = []
    if request.method == "POST":
        term = request.form.get("disease", "").strip().lower()
        if not term:
            messages.append("Please enter a condition or symptom name (e.g., 'fever' or 'influenza').")
        else:
            safe_term = term.replace("'", "\\'")
            try:
                q = list(prolog.query(f"total_diseases('{safe_term}', Total)"))
                if q:
                    total = q[0].get("Total", 0)
                else:
                    total = 0
                    messages.append("No data found for the given term in the knowledge base.")

                diseases = [str(sol['D']) for sol in prolog.query(f"symptom(D,'{safe_term}')")]
                if diseases:
                    for d in sorted(set(diseases)):
                        syms = [str(s['S']) for s in prolog.query(f"symptom('{d}', S)")]
                        diagnosis_list.append({
                            "condition": d,
                            "symptoms": ", ".join(syms) if syms else "—",
                            "severity": "N/A",
                            "advice": "Clinical judgement required"
                        })
                else:
                    syms = [str(sol['S']) for sol in prolog.query(f"symptom('{safe_term}', S)")]
                    if syms:
                        diagnosis_list.append({
                            "condition": term,
                            "symptoms": ", ".join(syms),
                            "severity": "N/A",
                            "advice": "Clinical judgement required"
                        })
            except Exception as e:
                messages.append(f"Error fetching data: {e}")
    return render_template(
        "total_diseases.html",
        disease=term,
        total=total,
        messages=messages,
        diagnosis_list=diagnosis_list
    )

@app.route("/visualize")
def visualize():
    counts = {"Emergency":0,"Urgent":0,"Routine":0,"Unknown":0}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0] != "when_iso":
                    lvl = row[2] if len(row)>2 else "Unknown"
                    counts[lvl] = counts.get(lvl,0) + 1
    return render_template("visualize.html", counts=counts)

@app.route("/cluster")
def run_clustering():
    messages = []
    cluster_results = []
    crosstab_html = None

    if not os.path.exists(CSV_FILE):
        messages.append("Error: cases.csv not found. Please log some cases first.")
        return render_template("cluster.html", messages=messages, cluster_results=cluster_results, crosstab_html=crosstab_html)
    
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

    try:
        with open(MLB_FILE, 'rb') as f:
            mlb = pickle.load(f)
        X = mlb.transform(df['all_features'])
    except Exception as e:
        messages.append(f"Error loading preprocessor ({MLB_FILE}): {e}")
        messages.append("Please run train_model.py first to create the preprocessor.")
        return render_template("cluster.html", messages=messages, cluster_results=cluster_results, crosstab_html=crosstab_html)

    k = 4  
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = mlb.classes_
    
    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :6]] 
        cluster_results.append({
            "name": f"Cluster {i}",
            "top_terms": ", ".join(top_terms)
        })

    df['cluster'] = model.labels_
    crosstab = pd.crosstab(df['cluster'], df['level'])
    crosstab_html = crosstab.to_html(classes="stats-table", border=0)

    messages.append(f"Clustering complete. Found {k} clusters from {len(df)} cases.")
    return render_template(
        "cluster.html", 
        messages=messages, 
        cluster_results=cluster_results, 
        crosstab_html=crosstab_html
    )

# --- NEW, FINAL /rl ROUTE (Triage Policy) ---
@app.route("/rl")
def run_rl():
    messages = []
    learning_log = [] # <-- To show the "happening"
    
    # --- 1. DEFINE THE ENVIRONMENT ---
    STATES = ['Routine', 'Urgent', 'Emergency']
    ACTIONS = ['discharge', 'admit_ward', 'admit_icu']

    # --- 2. DEFINE THE "SIMULATOR" (The Reward Rules) ---
    def get_reward(state, action):
        if state == 'Routine' and action == 'discharge': return 100
        if state == 'Urgent' and action == 'admit_ward': return 100
        if state == 'Emergency' and action == 'admit_icu': return 100
        if state == 'Emergency' and action == 'discharge': return -1000
        if state == 'Urgent' and action == 'discharge': return -100
        if state == 'Routine' and action == 'admit_ward': return -5
        if state == 'Routine' and action == 'admit_icu': return -20
        if state == 'Urgent' and action == 'admit_icu': return -20
        return -10 

    # --- 3. Q-LEARNING ALGORITHM ---
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.2  # Exploration rate
    num_episodes = 10000

    # --- Create the "BEFORE" Q-table ---
    q_table_before = pd.DataFrame(
        0.0,
        index=STATES,
        columns=ACTIONS
    )
    # Create the "AFTER" table, which we will modify
    q_table_after = q_table_before.copy()

    messages.append(f"Training Q-Learning agent for {num_episodes} episodes...")

    # Training Loop (This modifies q_table_after)
    for i in range(num_episodes):
        current_state = random.choice(STATES)
        
        action = ""
        is_exploring = False
        if random.uniform(0, 1) < epsilon:
            action = random.choice(ACTIONS) # Explore
            is_exploring = True
        else:
            action = q_table_after.loc[current_state].idxmax() # Exploit from the *after* table
            
        reward = get_reward(current_state, action)
        
        old_q = q_table_after.loc[current_state, action]
        new_q = old_q + alpha * (reward - old_q)
        q_table_after.loc[current_state, action] = new_q # Update the *after* table
        
        # --- Add to the learning log ---
        if i % (num_episodes // 10) == 0: # Log 10 samples
            log_action = "Explore" if is_exploring else "Exploit"
            log_entry = f"Ep {i:5d}: State='{current_state}', Action='{action}' ({log_action}), Reward={reward:5.0f}, New Q-Val={new_q:8.2f}"
            learning_log.append(log_entry)

    messages.append("--- Training Complete ---")

    # --- 4. PREPARE RESULTS FOR HTML ---
    
    policy = q_table_after.idxmax(axis=1).to_dict() 
    policy_list = [{"state": state, "action": action} for state, action in policy.items()]

    q_table_before_html = q_table_before.round(2).to_html(classes="stats-table", border=0)
    q_table_after_html = q_table_after.round(2).to_html(classes="stats-table", border=0)

    return render_template(
        "rl.html", 
        messages=messages, 
        q_table_before_html=q_table_before_html, # <-- "Before" table
        q_table_after_html=q_table_after_html,   # <-- "After" table
        policy_list=policy_list,
        learning_log=learning_log                # <-- NEW log
    )
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)