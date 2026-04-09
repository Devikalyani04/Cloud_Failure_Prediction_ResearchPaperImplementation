# -*- coding: utf-8 -*-
#install dependancies 
!pip install torch scikit-learn imbalanced-learn xgboost pandas numpy


#connecting notebook console to drive
from google.colab import drive
drive.mount('/content/drive')

#uploading data to the drive
!mkdir -p /content/drive/MyDrive/clusterdata/job_events
!mkdir -p /content/drive/MyDrive/clusterdata/task_events
!gsutil -m cp "gs://clusterdata-2011-2/job_events/*.csv.gz" /content/drive/MyDrive/clusterdata/job_events/
!gsutil -m cp "gs://clusterdata-2011-2/task_events/*.csv.gz" /content/drive/MyDrive/clusterdata/task_events/


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
JOB_EVENTS_DIR   = "/content/drive/MyDrive/clusterdata/job_events"  
TASK_EVENTS_DIR  = "/content/drive/MyDrive/clusterdata/task_events" 
USE_SYNTHETIC_DATA = False   
JOB_SAMPLE_SIZE  = 1_000_000
TASK_SAMPLE_SIZE = 14_000_000
RANDOM_STATE     = 42
TEST_SIZE        = 0.30
LSTM_EPOCHS      = 100
LSTM_PATIENCE    = 10
BATCH_SIZE       = 2048
LEARNING_RATE    = 1e-3

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time, warnings, gzip, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix, classification_report)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed. Run: pip install imbalanced-learn")

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("Cloud Failure Prediction — PyTorch Implementation")
print("=" * 70)
print(f"[INFO] Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS (GCT 2011 schema)
# ─────────────────────────────────────────────────────────────────────────────
JOB_COLS  = ["timestamp","missing_info","job_id","event_type","user_name",
              "scheduling_class","job_name","logical_job_name"]
TASK_COLS = ["timestamp","missing_info","job_id","task_index","machine_id",
             "event_type","user_name","scheduling_class","priority",
             "cpu_request","memory_request","disk_space_request",
             "different_machine_constraint"]

# event_type == 4 → FINISH (success), else → FAILURE
# We keep only terminal events (2=EVICT,3=FAIL,4=FINISH,5=KILL,6=LOST)
TERMINAL_EVENTS = [2, 3, 4, 5, 6]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING & PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_gzip_csvs(directory, columns, sample_size=None):
    """Load all .csv.gz partitions from a directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        return None
    dfs = []
    total = 0
    for f in files:
        try:
            if f.endswith(".gz"):
                df = pd.read_csv(f, header=None, names=columns,
                                 compression="gzip", low_memory=False)
            else:
                df = pd.read_csv(f, header=None, names=columns, low_memory=False)
            dfs.append(df)
            total += len(df)
            if sample_size and total >= sample_size:
                break
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    if sample_size:
        combined = combined.iloc[:sample_size]
    return combined


def generate_synthetic_data(n_job=200_000, n_task=500_000, seed=42):
    """
    Generate synthetic data that mimics GCT distributions.
    Used when real data files are not available.
    """
    rng = np.random.default_rng(seed)
    print("[INFO] Generating synthetic GCT-like data …")

    # Job events
    job_df = pd.DataFrame({
        "timestamp"       : rng.integers(0, 2.5e12, n_job),
        "missing_info"    : rng.choice([0, 1], n_job, p=[0.95, 0.05]),
        "job_id"          : rng.integers(1e8, 9e8, n_job),
        "event_type"      : rng.choice(TERMINAL_EVENTS, n_job,
                                        p=[0.003, 0.015, 0.573, 0.407, 0.002]),
        "user_name"       : rng.integers(1000, 9999, n_job),
        "scheduling_class": rng.choice([0,1,2,3], n_job, p=[0.46,0.32,0.19,0.03]),
        "job_name"        : rng.integers(1000, 9999, n_job),
        "logical_job_name": rng.integers(1000, 9999, n_job),
    })

    # Task events
    task_df = pd.DataFrame({
        "timestamp"       : rng.integers(0, 2.5e12, n_task),
        "missing_info"    : rng.choice([0, 1], n_task, p=[0.93, 0.07]),
        "job_id"          : rng.integers(1e8, 9e8, n_task),
        "task_index"      : rng.integers(0, 1000, n_task),
        "machine_id"      : rng.integers(1e8, 9e8, n_task),
        "event_type"      : rng.choice(TERMINAL_EVENTS, n_task,
                                        p=[0.05, 0.05, 0.44, 0.45, 0.01]),
        "user_name"       : rng.integers(1000, 9999, n_task),
        "scheduling_class": rng.choice([0,1,2,3], n_task, p=[0.46,0.32,0.19,0.03]),
        "priority"        : rng.choice(range(12), n_task,
                                        p=[0.55,0.05,0.04,0.04,0.04,0.04,
                                           0.04,0.04,0.04,0.04,0.04,0.04]),
        "cpu_request"     : rng.uniform(0.0, 0.5, n_task),
        "memory_request"  : rng.uniform(0.0, 0.5, n_task),
        "disk_space_request": rng.uniform(0.0, 0.01, n_task),
        "different_machine_constraint": rng.choice([0,1], n_task, p=[0.9,0.1]),
    })
    return job_df, task_df


def prepare_dataset_A(job_df, task_df):
    """
    Dataset A — Job-level termination.
    Features: scheduling_class, cpu_request(max), memory_request(max),
              disk_space_request(max)
    Label:    1=failure, 0=success
    """
    print("\n[DATA] Preparing Dataset A (Job-level) …")

    # Keep only terminal job events
    job_df = job_df[job_df["event_type"].isin(TERMINAL_EVENTS)].copy()

    # Aggregate task features per job (max of resource requests)
    task_agg = (task_df[["job_id","cpu_request","memory_request","disk_space_request"]]
                .dropna()
                .groupby("job_id")
                .max()
                .reset_index())

    # Merge
    merged = job_df.merge(task_agg, on="job_id", how="left")
    merged.dropna(subset=["cpu_request","memory_request","disk_space_request"],
                  inplace=True)

    # Label
    merged["label"] = (merged["event_type"] != 4).astype(int)

    features = ["scheduling_class","cpu_request","memory_request","disk_space_request"]
    X = merged[features].values.astype(np.float32)
    y = merged["label"].values.astype(int)

    print(f"[DATA] Dataset A shape: {X.shape} | "
          f"Failure: {y.sum()} ({100*y.mean():.1f}%) | "
          f"Success: {(1-y).sum()} ({100*(1-y).mean():.1f}%)")
    return X, y, features


def prepare_dataset_B(task_df):
    """
    Dataset B — Task-level termination.
    Features: scheduling_class, priority, cpu_request, memory_request,
              disk_space_request
    Label:    1=failure, 0=success
    """
    print("\n[DATA] Preparing Dataset B (Task-level) …")

    task_df = task_df[task_df["event_type"].isin(TERMINAL_EVENTS)].copy()
    task_df.dropna(subset=["cpu_request","memory_request",
                            "disk_space_request"], inplace=True)
    task_df["label"] = (task_df["event_type"] != 4).astype(int)

    features = ["scheduling_class","priority","cpu_request",
                "memory_request","disk_space_request"]
    X = task_df[features].values.astype(np.float32)
    y = task_df["label"].values.astype(int)

    print(f"[DATA] Dataset B shape: {X.shape} | "
          f"Failure: {y.sum()} ({100*y.mean():.1f}%) | "
          f"Success: {(1-y).sum()} ({100*(1-y).mean():.1f}%)")
    return X, y, features


def apply_smote(X_train, y_train):
    """Apply SMOTE for class balancing (as per paper Section 3.2.5)."""
    if not SMOTE_AVAILABLE:
        print("[WARN] SMOTE skipped — imbalanced-learn not installed.")
        return X_train, y_train
    print("[DATA] Applying SMOTE …", end=" ")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE → {X_res.shape[0]} samples "
          f"(was {X_train.shape[0]})")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: METRICS (as per paper Table 4–7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=""):
    """
    Compute all metrics used in the paper:
    Accuracy, Error Rate, Precision, Sensitivity (Recall),
    Specificity, F-Score.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    error_rate  = 1 - accuracy
    precision   = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)          # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0              # TNR
    f_score     = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  {'─'*52}")
    print(f"  Results — {label}")
    print(f"  {'─'*52}")
    print(f"  {'Accuracy':<20}: {accuracy*100:.2f}%")
    print(f"  {'Error Rate':<20}: {error_rate*100:.2f}%")
    print(f"  {'Precision':<20}: {precision*100:.2f}%")
    print(f"  {'Sensitivity (Recall)':<20}: {sensitivity*100:.2f}%")
    print(f"  {'Specificity':<20}: {specificity*100:.2f}%")
    print(f"  {'F-Score':<20}: {f_score:.4f}")
    print(f"  Confusion Matrix → TN={tn} FP={fp} FN={fn} TP={tp}")

    return {
        "Accuracy":    round(accuracy*100, 2),
        "Error Rate":  round(error_rate*100, 2),
        "Precision":   round(precision*100, 2),
        "Sensitivity": round(sensitivity*100, 2),
        "Specificity": round(specificity*100, 2),
        "F-Score":     round(f_score, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TRADITIONAL ML MODELS (TML)
# ─────────────────────────────────────────────────────────────────────────────

def get_tml_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boost": GradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
            device="cuda" if torch.cuda.is_available() else "cpu")
    return models


def run_tml_experiments(X_train, y_train, X_test, y_test,
                        dataset_name="Job"):
    """Train and evaluate all TML models."""
    print(f"\n{'='*70}")
    print(f"  TML EXPERIMENTS — {dataset_name}-Level Failure Prediction")
    print(f"{'='*70}")

    # Scale features (needed for LR)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = get_tml_models()
    results = {}

    for name, model in models.items():
        print(f"\n[TML] Training: {name} …")
        t0 = time.time()
        # LR uses scaled data; tree-based methods can use raw
        if "Logistic" in name:
            model.fit(X_tr_s, y_train)
            tr_pred = model.predict(X_tr_s)
            te_pred = model.predict(X_te_s)
        else:
            model.fit(X_train, y_train)
            tr_pred = model.predict(X_train)
            te_pred = model.predict(X_test)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")

        tr_metrics = compute_metrics(y_train, tr_pred,
                                     f"{name} — TRAINING")
        te_metrics = compute_metrics(y_test, te_pred,
                                     f"{name} — TESTING")
        results[name] = {"train": tr_metrics, "test": te_metrics,
                         "model": model}

    return results, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LSTM DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier with configurable number of hidden layers.
    Variants used in paper:
      - Single-Layer LSTM (1 hidden layer)
      - Bi-Layer LSTM    (2 hidden layers)
      - Tri-Layer LSTM   (3 hidden layers)
    Final dense layer outputs a single sigmoid-activated value.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1,
                 dropout=0.2):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        out = self.fc(out)
        return self.sigmoid(out).squeeze(1)


def make_lstm_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """Wrap numpy arrays into a PyTorch DataLoader (seq_len = 1)."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,F)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available())


def train_lstm(model, train_loader, val_loader, epochs=LSTM_EPOCHS,
               patience=LSTM_PATIENCE, lr=LEARNING_RATE, name="LSTM"):
    """
    Train an LSTM model with early stopping on validation loss.
    Paper: epoch=100, early stopping if no improvement for 10 epochs.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val  = float("inf")
    no_improve = 0
    best_state = None

    print(f"\n[DL] Training {name} (max {epochs} epochs, "
          f"patience {patience}) …")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss: {best_val:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_lstm(model, loader):
    """Run inference and return binary predictions."""
    model.eval()
    model = model.to(DEVICE)
    all_preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            prob = model(xb)
            pred = (prob >= 0.5).long().cpu().numpy()
            all_preds.extend(pred.tolist())
    return np.array(all_preds)


def run_dl_experiments(X_train, y_train, X_test, y_test,
                       dataset_name="Job"):
    """Train and evaluate all three LSTM variants."""
    print(f"\n{'='*70}")
    print(f"  DEEP LEARNING EXPERIMENTS — {dataset_name}-Level Failure Prediction")
    print(f"{'='*70}")

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
    X_te_s = scaler.transform(X_test).astype(np.float32)

    # Validation split from training set
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr_s, y_train, test_size=0.1, random_state=RANDOM_STATE,
        stratify=y_train)

    train_loader = make_lstm_loader(X_tr2, y_tr2,   shuffle=True)
    val_loader   = make_lstm_loader(X_val, y_val,   shuffle=False)
    test_loader  = make_lstm_loader(X_te_s, y_test, shuffle=False)
    full_train_loader = make_lstm_loader(X_tr_s, y_train, shuffle=False)

    input_size = X_train.shape[1]
    lstm_configs = [
        ("Single-Layer LSTM", 1),
        ("Bi-Layer LSTM",     2),
        ("Tri-Layer LSTM",    3),
    ]

    results = {}
    for name, num_layers in lstm_configs:
        model = LSTMClassifier(input_size=input_size,
                               hidden_size=64,
                               num_layers=num_layers)
        model = train_lstm(model, train_loader, val_loader,
                           name=name)

        tr_pred = predict_lstm(model, full_train_loader)
        te_pred = predict_lstm(model, test_loader)

        tr_metrics = compute_metrics(y_train, tr_pred,
                                     f"{name} — TRAINING")
        te_metrics = compute_metrics(y_test, te_pred,
                                     f"{name} — TESTING")
        results[name] = {"train": tr_metrics, "test": te_metrics}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FEATURE IMPORTANCE (Permutation-based, mirrors paper's Dalex)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(model, X, y, feature_names,
                           use_scaled=False, scaler=None, n_repeats=3):
    """
    Compute permutation importance as a proxy for Dalex-style
    feature importance used in the paper (Section 3.3.3).
    Higher score = more important feature.
    """
    if use_scaled and scaler:
        Xs = scaler.transform(X)
    else:
        Xs = X.copy()

    baseline = accuracy_score(y, model.predict(Xs))
    importance = np.zeros(len(feature_names))

    for i in range(len(feature_names)):
        drops = []
        for _ in range(n_repeats):
            Xp = Xs.copy()
            np.random.shuffle(Xp[:, i])
            drops.append(baseline - accuracy_score(y, model.predict(Xp)))
        importance[i] = np.mean(drops)

    total = importance.sum()
    if total > 0:
        importance = importance / total

    print("\n  Feature Importance Scores (normalized):")
    for fn, imp in sorted(zip(feature_names, importance),
                          key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {fn:<28}: {imp:.4f}  {bar}")

    return dict(zip(feature_names, importance))


def run_feature_importance(tml_results, X_train, y_train,
                           feature_names, dataset_name,
                           scaler=None):
    """Run feature importance for key models."""
    print(f"\n{'='*70}")
    print(f"  FEATURE IMPORTANCE — {dataset_name}-Level")
    print(f"{'='*70}")

    # Paper reports feature importance for each model; show best models
    key_models = ["XGBoost", "Decision Tree", "Random Forest",
                  "Logistic Regression"]
    for mname in key_models:
        if mname not in tml_results:
            continue
        model = tml_results[mname]["model"]
        print(f"\n  [{mname}]")
        use_sc = ("Logistic" in mname)
        permutation_importance(model, X_train, y_train,
                               feature_names,
                               use_scaled=use_sc, scaler=scaler)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SCALABILITY ANALYSIS (Section 3.3.4)
# ─────────────────────────────────────────────────────────────────────────────

def scalability_analysis(tml_results, X_full, y_full,
                         dataset_name="Job", scaler=None):
    """
    Measure prediction time for increasing data sizes (paper Section 3.3.4).
    Job level: 10K, 100K, 1M rows
    Task level: 10K, 100K, 1M, 10M rows
    """
    print(f"\n{'='*70}")
    print(f"  SCALABILITY ANALYSIS — {dataset_name}-Level")
    print(f"{'='*70}")

    sizes = [10_000, 100_000, min(1_000_000, len(X_full))]
    if dataset_name == "Task":
        sizes.append(min(10_000_000, len(X_full)))
    sizes = [s for s in sizes if s <= len(X_full)]

    models_to_test = {k: v["model"] for k, v in tml_results.items()}

    print(f"\n  {'Model':<25}", end="")
    for s in sizes:
        print(f"  {s:>10,} rows", end="")
    print()
    print("  " + "─" * (25 + 16 * len(sizes)))

    for mname, model in models_to_test.items():
        print(f"  {mname:<25}", end="")
        for s in sizes:
            idx  = np.random.choice(len(X_full), min(s, len(X_full)),
                                    replace=False)
            Xs   = X_full[idx]
            if scaler and "Logistic" in mname:
                Xs = scaler.transform(Xs)
            t0 = time.time()
            _  = model.predict(Xs)
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  {elapsed_ms:>12.1f} ms", end="")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SUMMARY TABLE (mirrors paper Tables 4–7)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(tml_results, dl_results, split="test",
                         dataset_name="Job"):
    """Print a consolidated results table similar to paper Tables 4-7."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE — {dataset_name}-Level | Split: {split.upper()}")
    print(f"{'='*70}")
    header = (f"  {'Model':<26} {'Accuracy':>9} {'Error':>7} "
              f"{'Precision':>10} {'Sensitivity':>12} "
              f"{'Specificity':>12} {'F-Score':>8}")
    print(header)
    print("  " + "─" * 87)

    all_results = {**tml_results, **dl_results}
    for name, res in all_results.items():
        m = res[split]
        print(f"  {name:<26} {m['Accuracy']:>8.2f}% "
              f"{m['Error Rate']:>6.2f}% "
              f"{m['Precision']:>9.2f}% "
              f"{m['Sensitivity']:>11.2f}% "
              f"{m['Specificity']:>11.2f}% "
              f"{m['F-Score']:>8.4f}")

    # Highlight best by accuracy on test set
    best = max(all_results.items(),
               key=lambda x: x[1][split]["Accuracy"])
    print(f"\n  ★ Best model ({split}): {best[0]} "
          f"— Accuracy: {best[1][split]['Accuracy']:.2f}%  "
          f"F-Score: {best[1][split]['F-Score']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load / generate data ──────────────────────────────────────────────
    real_job_dir  = os.path.isdir(JOB_EVENTS_DIR)
    real_task_dir = os.path.isdir(TASK_EVENTS_DIR)

    if (not USE_SYNTHETIC_DATA) and real_job_dir and real_task_dir:
        print("\n[INFO] Loading real GCT data …")
        job_df  = load_gzip_csvs(JOB_EVENTS_DIR,  JOB_COLS,  JOB_SAMPLE_SIZE)
        task_df = load_gzip_csvs(TASK_EVENTS_DIR, TASK_COLS, TASK_SAMPLE_SIZE)
        if job_df is None or task_df is None:
            print("[WARN] Could not load real data — falling back to synthetic.")
            job_df, task_df = generate_synthetic_data()
    else:
        if not USE_SYNTHETIC_DATA:
            print("[WARN] GCT directories not found — using synthetic data.")
        job_df, task_df = generate_synthetic_data()

    print(f"\n[INFO] Job events rows  : {len(job_df):,}")
    print(f"[INFO] Task events rows : {len(task_df):,}")

    # ── 2. Exploratory Data Analysis (EDA) ────────────────────────────────────
    print(f"\n{'='*70}")
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*70}")

    # Job event table
    terminal_jobs = job_df[job_df["event_type"].isin(TERMINAL_EVENTS)]
    job_status    = terminal_jobs["event_type"].value_counts()
    print("\n  Job Event Termination Status Distribution:")
    labels = {2:"EVICT", 3:"FAIL", 4:"FINISH", 5:"KILL", 6:"LOST"}
    for et, cnt in job_status.items():
        print(f"    {labels.get(et, str(et)):8s} (type={et}): {cnt:>8,}  "
              f"({100*cnt/len(terminal_jobs):.1f}%)")

    sc_dist = terminal_jobs["scheduling_class"].value_counts().sort_index()
    print("\n  Job Scheduling Class Distribution:")
    for sc, cnt in sc_dist.items():
        print(f"    Class {sc}: {cnt:>8,}")

    # Task event table
    terminal_tasks = task_df[task_df["event_type"].isin(TERMINAL_EVENTS)]
    task_status    = terminal_tasks["event_type"].value_counts()
    print("\n  Task Event Termination Status Distribution:")
    for et, cnt in task_status.items():
        print(f"    {labels.get(et, str(et)):8s} (type={et}): {cnt:>8,}  "
              f"({100*cnt/len(terminal_tasks):.1f}%)")

    # Resource request stats
    print("\n  Task Resource Request Statistics:")
    res_cols = ["cpu_request", "memory_request", "disk_space_request"]
    for col in res_cols:
        if col in task_df.columns:
            vals = task_df[col].dropna()
            print(f"    {col:<26}: mean={vals.mean():.5f}  "
                  f"std={vals.std():.5f}  "
                  f"min={vals.min():.5f}  "
                  f"max={vals.max():.5f}")

    # ── 3. Dataset A — Job Level ──────────────────────────────────────────────
    X_A, y_A, feat_A = prepare_dataset_A(job_df, task_df)

    X_tr_A, X_te_A, y_tr_A, y_te_A = train_test_split(
        X_A, y_A, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_A)

    # SMOTE on training split
    X_tr_A_sm, y_tr_A_sm = apply_smote(X_tr_A, y_tr_A)

    print(f"\n[DATA] Dataset A — Training: {X_tr_A_sm.shape[0]:,} "
          f"| Testing: {X_te_A.shape[0]:,}")

    # TML
    tml_A, scaler_A = run_tml_experiments(
        X_tr_A_sm, y_tr_A_sm, X_te_A, y_te_A, "Job")

    # Feature Importance — Job level
    run_feature_importance(tml_A, X_tr_A_sm, y_tr_A_sm,
                           feat_A, "Job", scaler=scaler_A)

    # DL
    dl_A = run_dl_experiments(
        X_tr_A_sm, y_tr_A_sm, X_te_A, y_te_A, "Job")

    # Summary tables (train & test — mirrors Tables 4 & 5)
    print_summary_table(tml_A, dl_A, split="train", dataset_name="Job")
    print_summary_table(tml_A, dl_A, split="test",  dataset_name="Job")

    # Scalability — Job level
    scalability_analysis(tml_A, X_A, y_A, "Job", scaler=scaler_A)

    # ── 4. Dataset B — Task Level ─────────────────────────────────────────────
    X_B, y_B, feat_B = prepare_dataset_B(task_df)

    X_tr_B, X_te_B, y_tr_B, y_te_B = train_test_split(
        X_B, y_B, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_B)

    X_tr_B_sm, y_tr_B_sm = apply_smote(X_tr_B, y_tr_B)

    print(f"\n[DATA] Dataset B — Training: {X_tr_B_sm.shape[0]:,} "
          f"| Testing: {X_te_B.shape[0]:,}")

    # TML
    tml_B, scaler_B = run_tml_experiments(
        X_tr_B_sm, y_tr_B_sm, X_te_B, y_te_B, "Task")

    # Feature Importance — Task level
    run_feature_importance(tml_B, X_tr_B_sm, y_tr_B_sm,
                           feat_B, "Task", scaler=scaler_B)

    # DL
    dl_B = run_dl_experiments(
        X_tr_B_sm, y_tr_B_sm, X_te_B, y_te_B, "Task")

    # Summary tables (mirrors Tables 6 & 7)
    print_summary_table(tml_B, dl_B, split="train", dataset_name="Task")
    print_summary_table(tml_B, dl_B, split="test",  dataset_name="Task")

    # Scalability — Task level
    scalability_analysis(tml_B, X_B, y_B, "Task", scaler=scaler_B)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

