"""
core.py – entities now list[str] instead of list[{full_name: ...}]
"""

from __future__ import annotations
import json, math, os
from datetime import datetime
from typing import List, Sequence, Tuple, Type, Union

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

client = OpenAI()
__all__ = ["parse_one", "prepare_batch_files", "upload_and_process_batches"]

# ------------------------------------------------------------------ Schema / prompt
def _build_schema(cats: Sequence[str]) -> Type[BaseModel]:
    return create_model(              # entities is now List[str]
        "Extraction",
        input_text=(str, ...),
        entities=(List[str], Field(default_factory=list)),
        categories=(List[str], Field(default_factory=list)),
    )

def _build_prompt(cats: Sequence[str], *, single_label: bool) -> str:
    label = "exactly **one**" if single_label else "all **relevant**"
    cat_line = ", ".join(cats)
    return (
        "You are an expert information extractor.\n"
        "1. List every *named entity* (people, places, organisations, etc.) "
        "as simple strings: `['Joe Biden', 'Afghanistan']`.\n"
        f"2. Pick {label} category from: {cat_line}.\n"
        "Return JSON that matches the schema."
    )

def _resp_fmt(schema: Type[BaseModel]) -> dict:
    return {"type": "json_schema", "strict": True, "schema": schema.model_json_schema()}

# ------------------------------------------------------------------ Single request
def parse_one(
    text: str,
    *,
    categories: Sequence[str],
    single_label: bool = False,
    model: str = "gpt-4o-2024-08-06",
    system_prompt: str | None = None,
    client_override: OpenAI | None = None,
) -> dict:
    """
    Extract entities and classify the text using an OpenAI model.

    Args:
        text (str): The input text to extract entities and classify.
        categories (Sequence[str]): List of possible categories for classification.
        single_label (bool, optional): If True, only one category will be selected. If False, all relevant categories may be selected. Defaults to False.
        model (str, optional): The OpenAI model to use. Defaults to "gpt-4o-2024-08-06".
        system_prompt (str, optional): Custom system prompt to override the default. Defaults to None.
        client_override (OpenAI, optional): Custom OpenAI client instance. Defaults to None.

    Returns:
        dict: A dictionary with extracted entities and categories.
    """
    oa = client_override or client
    schema = _build_schema(categories)
    prompt = system_prompt or _build_prompt(categories, single_label=single_label)

    rsp = oa.responses.parse(
        model=model,
        input=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        text_format=schema,
    )
    out: Union[dict, BaseModel] = rsp.output_parsed
    return out.model_dump() if isinstance(out, BaseModel) else out

# ------------------------------------------------------------------ Batch helpers (unchanged logic; new schema/prompt propagate automatically)
def prepare_batch_files(
    df: pd.DataFrame,
    text_column: str,
    output_dir: str,
    *,
    categories: Sequence[str],
    single_label: bool = False,
    model_name: str = "gpt-4o-2024-08-06",
    system_prompt: str | None = None,
    temperature: float = 0.0,
    rows_per_batch: int = 10_000,
) -> int:
    """
    Prepare batch files for entity extraction and classification jobs.

    Args:
        df (pd.DataFrame): DataFrame containing the data to process.
        text_column (str): Name of the column in df containing the text to process.
        output_dir (str): Directory to write batch files to.
        categories (Sequence[str]): List of possible categories for classification.
        single_label (bool, optional): If True, only one category will be selected. If False, all relevant categories may be selected. Defaults to False.
        model_name (str, optional): The OpenAI model to use. Defaults to "gpt-4o-2024-08-06".
        system_prompt (str, optional): Custom system prompt to override the default. Defaults to None.
        temperature (float, optional): Sampling temperature for the model. Defaults to 0.0.
        rows_per_batch (int, optional): Number of rows per batch file. Defaults to 10,000.

    Returns:
        int: The number of batch files created.
    """
    schema = _build_schema(categories)
    prompt = system_prompt or _build_prompt(categories, single_label=single_label)
    fmt = _resp_fmt(schema)

    total = math.ceil(len(df) / rows_per_batch)
    df = df.copy()
    df["_batch"] = df.index // rows_per_batch + 1
    os.makedirs(output_dir, exist_ok=True)

    for n in range(1, total + 1):
        sub = df[df["_batch"] == n]
        lines: List[str] = []
        for idx, row in sub.iterrows():
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(row[text_column])},
                ],
                "temperature": temperature,
                "response_format": fmt,
            }
            lines.append(json.dumps({"custom_id": f"req-{idx}",
                                     "method": "POST",
                                     "url": "/v1/chat/completions",
                                     "body": body}))
        with open(os.path.join(output_dir, f"batch_{n:03d}.jsonl"), "w") as f:
            f.write("\n".join(lines))
    return total

def upload_and_process_batches(
    directory: str,
    batch_info_csv: str,
    *,
    model: str = "gpt-4o-2024-08-06",
    client_override: OpenAI | None = None,
) -> List[Tuple[str, str]]:
    """
    Upload batch files and submit processing jobs to OpenAI.

    Args:
        directory (str): Directory containing batch .jsonl files to upload.
        batch_info_csv (str): Path to CSV file for logging batch job info.
        model (str, optional): The OpenAI model to use for batch jobs. Defaults to "gpt-4o-2024-08-06".
        client_override (OpenAI, optional): Custom OpenAI client instance. Defaults to None.

    Returns:
        List[Tuple[str, str]]: List of tuples (file_id, batch_id) for each submitted batch.
    """
    oa = client_override or client
    files = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    processed, rows = [], []

    for f in tqdm(files, desc="Submitting batch jobs"):
        with open(os.path.join(directory, f), "rb") as fh:
            up = oa.files.create(file=fh, purpose="batch")
        job = oa.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Entity/class batch • {f}", "model": model},
        )
        processed.append((up.id, job.id))
        rows.append({"timestamp": ts, "batch_file": f,
                     "file_id": up.id, "batch_id": job.id, "status": "submitted"})

    if rows:
        os.makedirs(os.path.dirname(batch_info_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(batch_info_csv,
                                  mode="a" if os.path.exists(batch_info_csv) else "w",
                                  header=not os.path.exists(batch_info_csv), index=False)
    return processed
