import numpy as np
import pandas as pd


def decode_synthetic(X_synth_raw, preprocessor, continuous_cols, categorical_cols):
    scaler = preprocessor.named_transformers_["num"]
    encoder = preprocessor.named_transformers_["cat"]

    n_cont = len(continuous_cols)

    # continuous part
    X_cont_scaled = X_synth_raw[:, :n_cont]
    X_cont = scaler.inverse_transform(X_cont_scaled)

    decoded = pd.DataFrame(X_cont, columns=continuous_cols)

    # categorical part
    start = n_cont

    for col, categories in zip(categorical_cols, encoder.categories_):
        end = start + len(categories)

        logits_block = X_synth_raw[:, start:end]
        idx = np.argmax(logits_block, axis=1)

        decoded[col] = np.array(categories)[idx]

        start = end

    return decoded