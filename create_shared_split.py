import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from hybrid_utils import load_sms_dataset

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "shared_split.json"

df = load_sms_dataset().reset_index(drop=True)
indices = list(range(len(df)))
labels = df["label"].astype(str).str.lower().tolist()

fitval_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=labels
)

fit_idx, val_idx = train_test_split(
    fitval_idx,
    test_size=0.25,
    random_state=43,
    stratify=[labels[i] for i in fitval_idx]
)

payload = {
    "dataset_size": len(df),
    "random_state_test": 42,
    "random_state_validation": 43,
    "fit_indices": fit_idx,
    "validation_indices": val_idx,
    "test_indices": test_idx,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print("Dataset size:", len(df))
print("Fit:", len(fit_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("Saved:", OUTPUT)
