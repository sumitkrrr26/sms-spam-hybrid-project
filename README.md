# SMS Spam Classification: Classical Baselines + Proposed Hybrid Model

This is a research starter project for **SMS only**. It keeps classical baseline experiments separate from a proposed hybrid deep-learning model.

## Project architecture

**Hybrid model**

SMS → preprocessing → BERT contextual token features and Word2Vec/BiLSTM sequence features → three graph convolution branches → feature fusion → self-attention → Dense/logit → HAM or SPAM.

Graph branches:
1. **Co-occurrence graph:** local token-window edges plus links between repeated subword surfaces.
2. **Heterogeneous lexical graph:** local edges plus same-type relations for word, number, URL-like and punctuation nodes.
3. **Syntactic graph:** dependency edges from spaCy `en_core_web_sm`, mapped from parser token character offsets to BERT subword positions.

**GSOWOA** runs a small Golden-Section + Whale Optimization search over learning rate, dropout and decision threshold. The objective is validation Spam F1. The implementation here is a clearly documented project-level hybrid; check the specific reference paper's algorithm definition before claiming it reproduces that paper exactly.

## Folder structure

- `train_baselines.py` — TF-IDF and eight classical models (including XGBoost if installed).
- `train_hybrid.py` — BERT + Word2Vec/BiLSTM + 3 GCN + self-attention + GSOWOA.
- `hybrid_model.py` — neural architecture.
- `hybrid_utils.py` — dataset download, Word2Vec, tokenization, graph generation.
- `gsowoa.py` — hyperparameter search implementation.
- `app.py` — Streamlit SMS classifier for any trained baseline/hybrid model.
- `data/` — downloaded UCI SMS file.
- `models/` — saved models.
- `results/` — measured metrics, benchmark results and confusion matrices.

## Important hardware advice

BERT training and repeated GSOWOA evaluations are computationally expensive. **Google Colab with a GPU is recommended** for the hybrid experiment. Running on a CPU can take a long time. On a limited laptop, first run the classical baseline and test the hybrid pipeline on a small subset with BERT frozen.

## 1. Create environment and install packages

In the VS Code terminal, from this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

PyTorch wheels can be large. For a GPU, install the appropriate PyTorch build for your environment, using the official PyTorch install selector. In Google Colab, PyTorch is usually already installed.

## 2. Run the classical baseline (recommended first)

```bash
python train_baselines.py
```

It downloads the UCI SMS Spam Collection if absent, compares TF-IDF classifiers, selects a model using validation spam F1, and reports final metrics on a separate test split.

The eight intended classifiers are Multinomial Naive Bayes, Linear SVM, Logistic Regression, Random Forest, K-Nearest Neighbors, Decision Tree, AdaBoost and XGBoost. If XGBoost is missing, the script prints a warning and runs the other seven; install it via `pip install xgboost` to include all eight.

## 3. Run a small hybrid smoke test

This exercises the complete architecture on a smaller stratified sample, freezes BERT to reduce compute, and uses a small GSOWOA search:

```bash
python train_hybrid.py --max-samples 500 --freeze-bert --epochs 1 --population 2 --iterations 0 --golden-steps 0 --batch-size 4
```

This is a code-pipeline smoke test, not a final experiment for a paper. The small subset and skipped search steps are only for checking that the components connect.

## 4. Run the proposed hybrid experiment

```bash
python train_hybrid.py --epochs 1 --population 3 --iterations 1 --golden-steps 1 --batch-size 8
```

This evaluates multiple candidate configurations on validation Spam F1. The held-out test set is not used by GSOWOA for selecting hyperparameters. Increase epochs, population, and iterations only after confirming the pipeline runs and you have enough compute.

If you're using Colab, upload/extract this project there, enable GPU under Runtime settings, install dependencies, run the spaCy model download, and run the commands above. BERT tokenizer/model weights are downloaded from Hugging Face on first use.

## 5. Run the app

After training at least one model:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Choose a trained model, paste an SMS, and click **Classify SMS**.

## Output files

Baseline:
- `models/classical_best.joblib`
- `models/classical_info.json`
- `results/classical/benchmark_results.csv`
- `results/classical/classification_report.txt`
- `results/classical/confusion_matrix.png`

Hybrid:
- `models/hybrid/hybrid_model.pt`
- `models/hybrid/hybrid_info.json`
- `models/hybrid/word_vocab.json`
- `models/hybrid/word2vec_embeddings.npy`
- `models/hybrid/word2vec.model`
- `results/hybrid/gsowoa_search_history.csv`
- `results/hybrid/test_metrics.csv`
- `results/hybrid/classification_report.txt`
- `results/hybrid/confusion_matrix.png`

## Methodology / research cautions

- `train_hybrid.py` uses a stratified fit/validation/test split. The test set is held out from GSOWOA model/threshold selection.
- Word2Vec is fitted on the fitting partition only.
- The co-occurrence and heterogeneous graphs are explicit project definitions. The syntactic graph uses spaCy dependency parses with a simple fallback only for individual messages where no dependency edges map to BERT tokens.
- This code is a research prototype. The added combination is a proposed framework; do not claim novelty, superiority, or an accuracy target until literature verification and controlled experiments support it.
- Report only real metrics generated by the run. For model comparison, keep the classical and hybrid test results clearly labelled and use equivalent split protocol where possible.
- The demo classifies pasted message text. It does not connect to a phone or silently read real SMS messages.

## Dataset

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
