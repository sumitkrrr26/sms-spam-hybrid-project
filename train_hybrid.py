from __future__ import annotations

import argparse

import json

import random

import time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from sklearn.metrics import (

    accuracy_score, classification_report, confusion_matrix,

    f1_score, precision_score, recall_score, roc_auc_score,

    average_precision_score, ConfusionMatrixDisplay,

)

from sklearn.model_selection import train_test_split

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from gsowoa import GSOWOA

from hybrid_model import HybridSpamNet

from hybrid_utils import (

    MAX_LENGTH, encode_texts, fit_word2vec, load_sms_dataset,

)

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models" / "hybrid"

RESULT_DIR = ROOT / "results" / "hybrid"

BERT_NAME = "bert-base-uncased"

def seed_everything(seed: int = 42) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

def make_loader(encoded, labels, batch_size: int, shuffle: bool) -> DataLoader:

    y = torch.as_tensor(labels, dtype=torch.float32)

    ds = TensorDataset(

        torch.as_tensor(encoded["input_ids"], dtype=torch.long),

        torch.as_tensor(encoded["attention_mask"], dtype=torch.long),

        torch.as_tensor(encoded["word_ids"], dtype=torch.long),

        torch.as_tensor(encoded["graphs"], dtype=torch.uint8),

        y,

    )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def forward_batch(model, batch, device):

    input_ids, mask, word_ids, graphs, labels = [item.to(device) for item in batch]

    logits = model(input_ids, mask, word_ids, graphs)

    return logits, labels

def train_one_candidate(

    params: np.ndarray,

    train_loader: DataLoader,

    val_loader: DataLoader,

    embedding_matrix: np.ndarray,

    device: torch.device,

    epochs: int,

    freeze_bert: bool,

    pos_weight: float,

    candidate_seed: int,

):

    seed_everything(candidate_seed)

    learning_rate = float(10 ** params[0])

    dropout = float(params[1])

    threshold = float(params[2])

    model = HybridSpamNet(

        embedding_matrix=embedding_matrix,

        bert_name=BERT_NAME,

        dropout=dropout,

        freeze_bert=freeze_bert,

    ).to(device)

    optimizer = torch.optim.AdamW(

        (p for p in model.parameters() if p.requires_grad),

        lr=learning_rate,

        weight_decay=0.01,

    )

    criterion = nn.BCEWithLogitsLoss(

        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)

    )

    for epoch in range(epochs):

        model.train()

        losses = []

        for batch in train_loader:

            optimizer.zero_grad(set_to_none=True)

            logits, labels = forward_batch(model, batch, device)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        print(

            f"    epoch {epoch + 1}/{epochs}: "

            f"training loss={np.mean(losses) if losses else float('nan'):.4f}"

        )

    model.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():

        for batch in val_loader:

            logits, labels = forward_batch(model, batch, device)

            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(all_labels, dtype=np.int64)

    probabilities = np.asarray(all_probs, dtype=np.float32)

    predicted = (probabilities >= threshold).astype(np.int64)

    val_f1 = f1_score(y_true, predicted, zero_division=0)

    val_precision = precision_score(y_true, predicted, zero_division=0)

    print(

        f"    validation spam F1={val_f1:.4f}; precision={val_precision:.4f}; "

        f"threshold={threshold:.3f}; lr={learning_rate:.2e}; dropout={dropout:.3f}"

    )

    return val_f1, val_precision, model, threshold

def evaluate_model(model, loader, device, threshold: float):

    model.eval()

    labels_all, probs_all = [], []

    with torch.no_grad():

        for batch in loader:

            logits, labels = forward_batch(model, batch, device)

            labels_all.extend(labels.cpu().numpy().astype(int).tolist())

            probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    y_true = np.asarray(labels_all, dtype=np.int64)

    probabilities = np.asarray(probs_all, dtype=np.float32)

    predicted = (probabilities >= threshold).astype(np.int64)

    metrics = {

        "accuracy": float(accuracy_score(y_true, predicted)),

        "spam_precision": float(precision_score(y_true, predicted, zero_division=0)),

        "spam_recall": float(recall_score(y_true, predicted, zero_division=0)),

        "spam_f1": float(f1_score(y_true, predicted, zero_division=0)),

        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),

        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,

        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,

    }

    return y_true, probabilities, predicted, metrics

def main():

    parser = argparse.ArgumentParser(

        description="Train BERT + BiLSTM + three-GCN hybrid model, optimized by GSOWOA."

    )

    parser.add_argument("--epochs", type=int, default=1,

                        help="Training epochs per GSOWOA candidate (default 1 for a practical first run).")

    parser.add_argument("--population", type=int, default=3,

                        help="GSOWOA population size; 3 is a small initial search.")

    parser.add_argument("--iterations", type=int, default=1,

                        help="Number of Whale Optimization iterations.")

    parser.add_argument("--golden-steps", type=int, default=1,

                        help="Golden-section refinement evaluations; use 0 to skip the refinement.")

    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--max-samples", type=int, default=0,

                        help="Optional stratified subset size for a quick smoke test; 0 means all data.")

    parser.add_argument("--freeze-bert", action="store_true",

                        help="Freeze BERT weights to reduce training cost.")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.population < 2:

        parser.error("epochs and batch-size must be positive; population must be at least 2.")

    seed_everything(42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if device.type == "cpu" and not args.freeze_bert:

        print("WARNING: fine-tuning BERT on CPU can be very slow. Google Colab GPU is recommended.")

        print("For a quick local run, pass --freeze-bert and a small --max-samples value.")

    df = load_sms_dataset().reset_index(drop=True)

    # Full-data experiments use the same shared split as train_baselines.py.
    # --max-samples is only a smoke test and uses a separate subset/split.
    if args.max_samples and args.max_samples < len(df):
        if args.max_samples < 100:
            parser.error("--max-samples should be at least 100 for stratified testing.")
        df, _ = train_test_split(
            df, train_size=args.max_samples, random_state=42, stratify=df["label"]
        )
        df = df.reset_index(drop=True)
        texts = df["text"].astype(str).tolist()
        labels = (df["label"].astype(str).str.lower() == "spam").astype(np.int64).to_numpy()
        indices = np.arange(len(texts))
        fitval_idx, test_idx = train_test_split(
            indices, test_size=0.20, random_state=42, stratify=labels
        )
        fit_idx, val_idx = train_test_split(
            fitval_idx, test_size=0.25, random_state=43, stratify=labels[fitval_idx]
        )
        print("WARNING: --max-samples is enabled; this is a smoke test, not a shared-split comparison.")
    else:
        split_path = ROOT / "results" / "shared_split.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Shared split not found: {split_path}. Run create_shared_split.py first."
            )
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if len(df) != split.get("dataset_size"):
            raise ValueError(
                "Dataset size does not match results/shared_split.json. "
                "Recreate the shared split before training."
            )
        fit_idx = np.asarray(split["fit_indices"], dtype=np.int64)
        val_idx = np.asarray(split["validation_indices"], dtype=np.int64)
        test_idx = np.asarray(split["test_indices"], dtype=np.int64)
        combined = np.concatenate([fit_idx, val_idx, test_idx])
        if len(combined) != len(df) or set(combined.tolist()) != set(range(len(df))):
            raise ValueError("Shared split indices are incomplete, duplicated, or invalid.")
        texts = df["text"].astype(str).tolist()
        labels = (df["label"].astype(str).str.lower() == "spam").astype(np.int64).to_numpy()

    fit_texts = [texts[i] for i in fit_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    y_fit, y_val, y_test = labels[fit_idx], labels[val_idx], labels[test_idx]
    print(
        f"Dataset={len(df):,}; fit={len(fit_idx):,}; "
        f"validation={len(val_idx):,}; test={len(test_idx):,}"
    )

    try:

        import spacy

        nlp = spacy.load("en_core_web_sm")

    except Exception as exc:

        raise RuntimeError(

            "The spaCy English parser is required for the syntactic GCN branch.\n"

            "Run these commands in your active environment:\n"

            "  pip install spacy\n"

            "  python -m spacy download en_core_web_sm"

        ) from exc

    print("Training Word2Vec on the fitting partition only...")

    vocab, embedding_matrix = fit_word2vec(fit_texts, MODEL_DIR)

    print("Loading BERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME, use_fast=True)

    print("Encoding fit messages and building the three graphs...")

    fit_encoded = encode_texts(fit_texts, tokenizer, nlp, vocab, show_progress=True)

    print("Encoding validation messages and building the three graphs...")

    val_encoded = encode_texts(val_texts, tokenizer, nlp, vocab, show_progress=True)

    print("Encoding test messages and building the three graphs...")

    test_encoded = encode_texts(test_texts, tokenizer, nlp, vocab, show_progress=True)

    fit_loader = make_loader(fit_encoded, y_fit, args.batch_size, shuffle=True)

    fit_eval_loader = make_loader(fit_encoded, y_fit, args.batch_size, shuffle=False)

    val_loader = make_loader(val_encoded, y_val, args.batch_size, shuffle=False)

    test_loader = make_loader(test_encoded, y_test, args.batch_size, shuffle=False)

    positives = max(1, int(y_fit.sum()))

    negatives = max(1, int(len(y_fit) - y_fit.sum()))

    pos_weight = negatives / positives

    candidate_counter = {"count": 0}

    best = {

        "score": -1.0,

        "precision": -1.0,

        "state": None,

        "params": None,

        "threshold": 0.5,

    }

    search_records = []

    def objective(position):

        candidate_counter["count"] += 1

        count = candidate_counter["count"]

        print(f"\nGSOWOA candidate {count}: {position.tolist()}")

        start = time.perf_counter()

        score, precision, model, threshold = train_one_candidate(

            position,

            fit_loader,

            val_loader,

            embedding_matrix,

            device,

            args.epochs,

            args.freeze_bert,

            pos_weight,

            candidate_seed=42,  # same seed helps make candidate comparisons more stable

        )

        elapsed = time.perf_counter() - start

        record = {

            "candidate": count,

            "log10_learning_rate": float(position[0]),

            "learning_rate": float(10 ** position[0]),

            "dropout": float(position[1]),

            "threshold": float(position[2]),

            "validation_spam_f1": float(score),

            "validation_spam_precision": float(precision),

            "seconds": float(elapsed),

        }

        search_records.append(record)

        if (score > best["score"]) or (

            score == best["score"] and precision > best["precision"]

        ):

            best["score"] = float(score)

            best["precision"] = float(precision)

            best["params"] = position.copy()

            best["threshold"] = float(threshold)

            best["state"] = {

                key: value.detach().cpu().clone()

                for key, value in model.state_dict().items()

            }

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return float(score)

    # GSOWOA dimensions: log10 learning rate, dropout, classification threshold.

    optimizer = GSOWOA(

        bounds=[(-5.0, -3.3), (0.10, 0.50), (0.25, 0.75)],

        population_size=args.population,

        iterations=args.iterations,

        golden_steps=args.golden_steps,

        seed=42,

    )

    _, _, history = optimizer.optimize(objective)

    if best["state"] is None:

        raise RuntimeError("GSOWOA did not produce a trained candidate.")

    best_lr = float(10 ** best["params"][0])

    best_dropout = float(best["params"][1])

    best_threshold = float(best["threshold"])

    print("\nLoading best validation candidate and evaluating the held-out test set...")

    best_model = HybridSpamNet(

        embedding_matrix=embedding_matrix,

        bert_name=BERT_NAME,

        dropout=best_dropout,

        freeze_bert=args.freeze_bert,

    ).to(device)

    best_model.load_state_dict(best["state"])

    y_true, probabilities, predicted, metrics = evaluate_model(

        best_model, test_loader, device, best_threshold

    )

    torch.save(

        {

            "state_dict": best_model.state_dict(),

            "bert_name": BERT_NAME,

            "dropout": best_dropout,

            "freeze_bert": bool(args.freeze_bert),

            "threshold": best_threshold,

            "max_length": MAX_LENGTH,

        },

        MODEL_DIR / "hybrid_model.pt",

    )

    (MODEL_DIR / "hybrid_info.json").write_text(

        json.dumps(

            {

                "model_name": "BERT + Word2Vec/BiLSTM + 3 GCN + feature fusion + self-attention",

                "optimizer": "project GSOWOA implementation",

                "selected_learning_rate": best_lr,

                "selected_dropout": best_dropout,

                "selected_threshold": best_threshold,

                "validation_spam_f1": float(best["score"]),

                "validation_spam_precision": float(best["precision"]),

                "metrics": metrics,

                "samples": {

                    "total": int(len(df)),

                    "fit": int(len(fit_idx)),

                    "validation": int(len(val_idx)),

                    "test": int(len(test_idx)),

                },

                "search_candidates_evaluated": len(search_records),

                "device": str(device),

                "freeze_bert": bool(args.freeze_bert),

            },

            indent=2,

        ),

        encoding="utf-8",

    )

    pd.DataFrame(search_records).to_csv(RESULT_DIR / "gsowoa_search_history.csv", index=False)

    pd.DataFrame([metrics]).to_csv(RESULT_DIR / "test_metrics.csv", index=False)

    (RESULT_DIR / "classification_report.txt").write_text(

        "Selected using validation Spam F1; the test split was not used for GSOWOA.\n"

        f"Threshold: {best_threshold:.4f}\n\n"

        + classification_report(

            y_true, predicted, labels=[0, 1],

            target_names=["ham", "spam"], digits=4, zero_division=0,

        ),

        encoding="utf-8",

    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(

        y_true, predicted, labels=[0, 1],

        display_labels=["Ham", "Spam"], cmap="Blues", ax=ax, colorbar=False

    )

    ax.set_title("Hybrid model — held-out SMS test set")

    fig.tight_layout()

    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)

    plt.close(fig)

    print("\nHELD-OUT TEST METRICS (measured in this run)")

    print(json.dumps(metrics, indent=2))

    print(f"Model saved: {MODEL_DIR / 'hybrid_model.pt'}")

    print(f"Search history: {RESULT_DIR / 'gsowoa_search_history.csv'}")

    print("The test split was not used to choose hyperparameters.")

if __name__ == "__main__":

    main()
