import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


from models import ConditionalGenerator, ConditionalDiscriminator
from preprocessing import process_data
from utils import decode_synthetic
from constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cG = ConditionalGenerator(NOISE_DIM, CONDITION_DIM, INPUT_DIM).to(device)
cD = ConditionalDiscriminator(INPUT_DIM, CONDITION_DIM).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(
    cG.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)
optimizer_D = torch.optim.Adam(
    cD.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

preprocessor, X_train, X_test, y_train, y_test = process_data(target_col="income", 
             continuous_cols=CONTINUOUS_COLS, 
            categorical_cols=CATEGORICAL_COLS
            )

cG.eval()

n_synth = X_train.shape[0]

rng = np.random.default_rng(42)

y_synth_cgan_np = rng.choice(
    y_train.values,
    size=n_synth,
    replace=True
)

y_synth_cgan = torch.tensor(
    y_synth_cgan_np,
    dtype=torch.long,
    device=device
)

# Real baseline
rf_real = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_real.fit(X_train, y_train)

real_probs = rf_real.predict_proba(X_test)[:, 1]
auc_real = roc_auc_score(y_test, real_probs)


with torch.no_grad():
    z = torch.randn(n_synth, NOISE_DIM, device=device)
    X_synth_cgan_raw = cG(z, y_synth_cgan).cpu().numpy()

df_synth_cgan = decode_synthetic(
    X_synth_cgan_raw,
    preprocessor,
    CONTINUOUS_COLS,
    CATEGORICAL_COLS
)

X_synth_cgan = preprocessor.transform(df_synth_cgan)

print(X_synth_cgan.shape)
print(pd.Series(y_synth_cgan_np).value_counts(normalize=True))

X_det_cgan = np.vstack([
    X_train,
    X_synth_cgan
])

y_det_cgan = np.concatenate([
    np.zeros(len(X_train)),
    np.ones(len(X_synth_cgan))
])



skf = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

aucs = []

for train_idx, test_idx in skf.split(X_det_cgan, y_det_cgan):

    X_tr = X_det_cgan[train_idx]
    X_te = X_det_cgan[test_idx]

    y_tr = y_det_cgan[train_idx]
    y_te = y_det_cgan[test_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr, y_tr)

    probs = rf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, probs)

    aucs.append(auc)

print("Detection AUCs:", aucs)
print("Mean Detection AUC:", np.mean(aucs))

rng = np.random.default_rng(42)


rf_synth = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf_synth.fit(X_synth_cgan, y_synth_cgan_np)

synth_probs = rf_synth.predict_proba(X_test)[:, 1]
auc_synth = roc_auc_score(y_test, synth_probs)

efficacy = auc_synth / auc_real

print("AUC synthetic train -> real test:", auc_synth)
print("Efficacy:", efficacy)