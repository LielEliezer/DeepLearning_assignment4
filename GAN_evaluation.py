import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


from models import Generator, Discriminator
from preprocessing import process_data
from utils import decode_synthetic
from constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


preprocessor, X_train, X_test, y_train, y_test = process_data(target_col="income", 
             continuous_cols=CONTINUOUS_COLS, 
            categorical_cols=CATEGORICAL_COLS
            )

G = Generator(NOISE_DIM, INPUT_DIM).to(device)
G.load_state_dict(torch.load("GAN/G_weights.pth"))
G.eval()

n_synth = X_train.shape[0]

with torch.no_grad():
    z = torch.randn(n_synth, NOISE_DIM, device=device)
    X_synth_raw = G(z).cpu().numpy()


df_synth = decode_synthetic(
    X_synth_raw,
    preprocessor,
    CONTINUOUS_COLS,
    CATEGORICAL_COLS
)

real_label = 0
fake_label = 1

X_synth = preprocessor.transform(df_synth)

X_det = np.vstack([
    X_train,
    X_synth
])

y_det = np.concatenate([
    np.zeros(len(X_train)),
    np.ones(len(X_synth))
])



skf = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

aucs = []

for train_idx, test_idx in skf.split(X_det, y_det):

    X_tr = X_det[train_idx]
    X_te = X_det[test_idx]

    y_tr = y_det[train_idx]
    y_te = y_det[test_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr, y_tr)
    a = rf.predict_proba(X_te)

    probs = rf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, probs)

    aucs.append(auc)

print("Detection AUCs:", aucs)
print("Mean Detection AUC:", np.mean(aucs))

X_det_cont = np.vstack([
    X_train[:, :6],
    X_synth[:, :6]
])

y_det = np.concatenate([
    np.zeros(len(X_train)),
    np.ones(len(X_synth))
])


# Real baseline
rf_real = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
a = min(y_train)
b = max(y_train)
rf_real.fit(X_train, y_train)

real_probs = rf_real.predict_proba(X_test)[:, 1]
auc_real = roc_auc_score(y_test, real_probs)

print("AUC real train -> real test:", auc_real)

rng = np.random.default_rng(42)

y_synth = rng.choice(
    y_train,
    size=len(X_synth),
    replace=True
)

rf_synth = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_synth.fit(X_synth, y_synth)

synth_probs = rf_synth.predict_proba(X_test)[:, 1]
auc_synth = roc_auc_score(y_test, synth_probs)

efficacy = auc_synth / auc_real

print("AUC synthetic train -> real test:", auc_synth)
print("Efficacy:", efficacy)