from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
SMS_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MAX_LENGTH = 64
WORD_EMBEDDING_DIM = 100
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_sms_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load UCI SMS Spam Collection, downloading it if it is absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        archive_path = path.parent / "sms_spam_collection.zip"
        request = urllib.request.Request(
            SMS_ZIP_URL,
            headers={"User-Agent": "SMS-Spam-Hybrid-Research-Project/1.0"},
        )
        print("Dataset not found. Downloading the UCI SMS Spam Collection...")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                archive_path.write_bytes(response.read())
            with zipfile.ZipFile(archive_path, "r") as archive:
                member = next(
                    (
                        name for name in archive.namelist()
                        if Path(name).name == "SMSSpamCollection"
                    ),
                    None,
                )
                if member is None:
                    raise FileNotFoundError(
                        "The UCI archive did not contain SMSSpamCollection."
                    )
                path.write_bytes(archive.read(member))
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                "Dataset download failed. Check internet access or manually place "
                f"the UCI 'SMSSpamCollection' file at {path}."
            ) from exc
        finally:
            archive_path.unlink(missing_ok=True)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8",
        encoding_errors="replace",
        quoting=3,
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)

    if len(df) < 10 or set(df["label"].unique()) != {"ham", "spam"}:
        raise ValueError("Dataset parse failed; expected rows containing ham/spam and an SMS.")
    return df


def word_tokens(text: str) -> list[str]:
    """Simple word tokens used only by the Word2Vec/BiLSTM branch."""
    return re.findall(r"[a-zA-Z]+|[0-9]+|[^\w\s]", str(text).lower())


def fit_word2vec(
    train_texts: list[str],
    output_dir: Path,
    vector_size: int = WORD_EMBEDDING_DIM,
    seed: int = 42,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Train Word2Vec on the fitting partition only and build the embedding matrix.
    Validation and test text are deliberately excluded to reduce leakage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = [word_tokens(text) or [UNK_TOKEN.lower()] for text in train_texts]
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=15,
        seed=seed,
    )

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in w2v.wv.index_to_key:
        if token not in vocab:
            vocab[token] = len(vocab)

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.05, size=(len(vocab), vector_size)).astype(np.float32)
    matrix[0] = 0.0
    matrix[1] = 0.0
    for token, idx in vocab.items():
        if token in w2v.wv:
            matrix[idx] = w2v.wv[token]

    with (output_dir / "word_vocab.json").open("w", encoding="utf-8") as handle:
        import json
        json.dump(vocab, handle, ensure_ascii=False)
    np.save(output_dir / "word2vec_embeddings.npy", matrix)
    w2v.save(str(output_dir / "word2vec.model"))
    return vocab, matrix


def encode_word_sequence(text: str, vocab: dict[str, int], max_words: int = MAX_LENGTH) -> np.ndarray:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in word_tokens(text)]
    ids = ids[:max_words]
    if not ids:
        ids = [vocab[UNK_TOKEN]]
    ids.extend([vocab[PAD_TOKEN]] * (max_words - len(ids)))
    return np.asarray(ids, dtype=np.int64)


def _add_undirected_edge(adjacency: np.ndarray, i: int, j: int) -> None:
    if i == j:
        adjacency[i, i] = 1
    else:
        adjacency[i, j] = 1
        adjacency[j, i] = 1


def _lexical_type(surface: str) -> str:
    lower = surface.lower()
    if "http" in lower or "www" in lower or lower.endswith(".com"):
        return "url"
    if any(char.isdigit() for char in lower):
        return "number"
    if not any(char.isalnum() for char in lower):
        return "punctuation"
    if any(char.isalpha() for char in lower):
        return "word"
    return "other"


def build_graph_adjacencies(
    text: str,
    offsets: list[tuple[int, int]],
    attention_mask: list[int],
    spacy_doc: Any,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """
    Create three undirected adjacency matrices over BERT subword positions:
      0. local-window co-occurrence graph,
      1. lexical-type-aware heterogeneous relation graph,
      2. syntactic dependency graph mapped from spaCy tokens to BERT subwords.

    Special tokens/padding have no text offsets and are excluded from graph edges.
    """
    graphs = np.zeros((3, max_length, max_length), dtype=np.uint8)
    active = [
        i for i, ((start, end), mask) in enumerate(zip(offsets, attention_mask))
        if mask and end > start
    ]

    for i in active:
        for graph_index in range(3):
            graphs[graph_index, i, i] = 1

    # 1) Local co-occurrence: link nearby subword positions within a sliding window.
    for pos, i in enumerate(active):
        for j in active[pos + 1: pos + 4]:
            _add_undirected_edge(graphs[0], i, j)
    # Repeated surface tokens are linked as an additional co-occurrence cue.
    surfaces = {}
    for i in active:
        start, end = offsets[i]
        surface = text[start:end].lower().strip()
        if surface:
            surfaces.setdefault(surface, []).append(i)
    for positions in surfaces.values():
        if len(positions) > 1:
            for i, j in zip(positions, positions[1:]):
                _add_undirected_edge(graphs[0], i, j)

    # 2) Heterogeneous lexical graph: nodes are categorized as word/number/URL/
    # punctuation/other. Local edges and same-type edges encode relation patterns.
    types = {}
    for i in active:
        start, end = offsets[i]
        types[i] = _lexical_type(text[start:end])
    for pos, i in enumerate(active):
        for j in active[pos + 1: pos + 5]:
            if (j - i) <= 1 or types[i] == types[j]:
                _add_undirected_edge(graphs[1], i, j)

    # 3) Syntactic graph: use real dependency links from spaCy's English parser.
    # Map parser-token character spans to the overlapping BERT wordpieces.
    char_to_subwords: dict[int, list[int]] = {}
    for token in spacy_doc:
        if token.is_space:
            continue
        token_start = int(token.idx)
        token_end = token_start + len(token.text)
        pieces = [
            i for i in active
            if offsets[i][0] < token_end and offsets[i][1] > token_start
        ]
        if pieces:
            char_to_subwords[token.i] = pieces

    for token in spacy_doc:
        if token.is_space or token.head.i == token.i:
            continue
        child_pieces = char_to_subwords.get(token.i, [])
        parent_pieces = char_to_subwords.get(token.head.i, [])
        for i in child_pieces:
            for j in parent_pieces:
                _add_undirected_edge(graphs[2], i, j)

    # Robust fallback for parser edge cases: connect consecutive lexical nodes,
    # while retaining self-loops. Normal runs with en_core_web_sm use dependency edges.
    if len(active) > 1 and int(graphs[2].sum()) <= len(active):
        for i, j in zip(active, active[1:]):
            _add_undirected_edge(graphs[2], i, j)

    return graphs


def encode_texts(
    texts: list[str],
    tokenizer: Any,
    nlp: Any,
    word_vocab: dict[str, int],
    max_length: int = MAX_LENGTH,
    show_progress: bool = False,
) -> dict[str, np.ndarray]:
    """Tokenize a collection and construct the three graph matrices."""
    input_ids, attention_masks, word_ids, graph_list = [], [], [], []
    docs = nlp.pipe((str(text) for text in texts), batch_size=128)
    total = len(texts)

    for idx, (text, doc) in enumerate(zip(texts, docs), start=1):
        item = tokenizer(
            str(text),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in item["offset_mapping"]]
        mask = list(item["attention_mask"])
        graph = build_graph_adjacencies(
            str(text), offsets, mask, doc, max_length=max_length
        )
        input_ids.append(item["input_ids"])
        attention_masks.append(mask)
        word_ids.append(encode_word_sequence(str(text), word_vocab, max_words=max_length))
        graph_list.append(graph)
        if show_progress and (idx % 500 == 0 or idx == total):
            print(f"  Encoded {idx:,}/{total:,} messages")

    return {
        "input_ids": np.asarray(input_ids, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "word_ids": np.asarray(word_ids, dtype=np.int64),
        "graphs": np.asarray(graph_list, dtype=np.uint8),
    }
