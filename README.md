# rna-sequencing

# RNA Embedding Binary Classifier

This project trains a lightweight **neural network** to classify DNA sequences into binary labels using pretrained **DNABERT-2** token embeddings.

---

## Overview

**Script:** `scripts/binary_classifier.py`  
**Framework:** PyTorch + Hugging Face Transformers.

The network predicts a single label (0 or 1) per input sequence.

---

## Data

The dataset should be a CSV file named:

```
./data/data_sequences.csv
```

with columns:

| Column     | Type  | Description                               |
| ---------- | ----- | ----------------------------------------- |
| `sequence` | `str` | Nucleotide sequence (A/C/G/T or A/C/G/U). |
| `label`    | `int` | Binary target label (0 or 1).             |

As the API is too large to upload to GitHub, the sequences and labels have already been added to a new `CSV` file. If you download the API and put it into the `/api` folder, you can recreate the sequences and labels by running:

```bash
python3 /scripts/get_seq_labels.py
```

Example from `data_sequences.csv`:

```csv
sequence,label
ACGTGCTA -> 270 chars,1
TTGCGGCA -> 270 chars ,0
```

Sequences are tokenized using the pretrained **[`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M)** tokenizer, which was trained on genomic text and supports direct A/C/G/T tokenization — no k-mer preprocessing required.

---

## Requirements

All dependencies are listed in `requirements.txt`. I used Python 3.12.9, should work for anything 3.12+

First create a virtual environment in Python:

Create

```bash
python3 -m venv *venv_name*
```

Activate

```bash
source *venv_name*/bin/activate
```

Then install with:

```bash
pip install -r requirements.txt
```

---

## Training

Run the script directly:

```bash
python scripts/binary_classifier.py
```

- Uses the `mps` device on macOS (Apple Silicon) if available, else CPU.
- Defaults: `batch_size=32`, `epochs=20`, `lr=2e-4`, `weight_decay=1e-5`.
- Splits 80/20 train–validation randomly (can be changed in `make_dataloaders()`).

Console output shows per-epoch training loss and final validation accuracy.

---
