# openai-categorizer-nlp

Entity extraction **and** text classification powered by OpenAI “Structured Outputs” (strict JSON-schema mode).

Works great for quick notebook calls **and** large-scale batch jobs.

---

## 📚 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Custom Prompts](#custom-prompts)
- [Batch Workflow](#batch-workflow)
- [Testing](#testing)

---

## ✨ Features

| Feature                   | Details                                                                 |
|--------------------------|------------------------------------------------------------------------|
| **Named-entity extraction** | Returns `entities` as a flat list, e.g. `["Joe Biden", "Afghanistan"]` |
| **User-defined categories** | Pass any label set, e.g. `["politics", "military_event"]`              |
| **Single vs multi-label**   | `single_label=True` forces exactly one category                         |
| **Custom prompt & schema**  | Override `system_prompt` or plug in your own Pydantic model             |
| **Batch API helper**        | DataFrame → JSONL → automatic Batch submission                          |
| **Typed internals**         | Pydantic validates responses before you see them                        |
| **Modern packaging**        | Python ≥ 3.10, `pyproject.toml`, editable installs                      |

---

## 📦 Installation

You can install the latest release here

```bash
$ pip install git+https://github.com/dsottimano/openai_categorizer_nlp.git@0.1.0
```

Or for local development

```bash
# Clone the repo and set up a virtual environment
$ git clone https://github.com/dsottimano/openai-categorizer-nlp.git
$ cd openai-categorizer-nlp
$ python -m venv .venv && source .venv/bin/activate
$ pip install -e .
```

Set your OpenAI key:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🚀 Quick Start

### Single Request

```python
from openai_categorizer_nlp import parse_one

text = "biden afghanistan withdrawal"
cats = ["politics", "international_relations", "military_event"]

out = parse_one(
    text,
    categories=cats,
    single_label=False,            # multi-label
    model="gpt-4o-2024-08-06",
)

print(out)
# {
#   'input_text': 'biden afghanistan withdrawal',
#   'entities': ['Joe Biden', 'Afghanistan'],
#   'categories': ['politics', 'international_relations']
# }
```

---

## 📝 Custom Prompts

You can override the system prompt for custom extraction/classification behavior:

```python
prompt = (
    "Extract every named entity (people, places, organisations, countries) "
    "in canonical form, then classify the text."
)

parse_one(text, categories=cats, system_prompt=prompt)
```

---

## 🗄️ Batch Workflow

Process large datasets with batch helpers:

```python
import pandas as pd
from openai_categorizer_nlp import prepare_batch_files, upload_and_process_batches

df = pd.read_csv("queries.csv")   # must contain a 'query' column
cats = ["politics", "international_relations", "military_event"]

# 1 · Create request files
prepare_batch_files(
    df,
    text_column="query",
    output_dir="batches",
    categories=cats,
    model_name="gpt-4o-2024-08-06",
)

# 2 · Upload & launch jobs
upload_and_process_batches("batches", "batch_log.csv")
```

> **Note:** `batch_log.csv` stores `file_id`, `batch_id`, `timestamp`, and `status` for later polling.

---

## 🧪 Testing

```bash
pip install pytest           # or: pip install -e ".[dev]"
pytest -q                   # mocked unit tests
OPENAI_API_KEY=sk-... pytest -m integration -v   # live-API tests
```
