"""
core.py · v1.0
Flexible extraction/classification helper with **feature toggles**

New kwargs
----------
• include_entities        (bool, default True)   – return entities?
• include_classification  (bool, default True)   – return categories?
• single_label            (bool, default False)  – only 1 category if True

If you disable a feature its schema field vanishes and the prompt no longer
mentions it, guaranteeing the model won’t hallucinate that output.
"""

from __future__ import annotations
import json, math, os
from datetime import datetime
from typing import List, Sequence, Tuple, Type, Union, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

client = OpenAI()
__all__ = ["parse_one", "prepare_batch_files", "upload_and_process_batches"]


# ────────────────────────────────────────────────────────────────────────────
#  Schema / prompt builders
# ────────────────────────────────────────────────────────────────────────────
def _build_schema(
    cats: Sequence[str],
    *,
    include_entities: bool,
    include_classification: bool,
    single_label: bool,
) -> Type[BaseModel]:
    """Return a dynamic Pydantic model tailored to user toggles."""
    fields: dict[str, tuple] = {
        "input_text": (str, ...),
    }

    if include_entities:
        fields["entities"] = (List[str], Field(default_factory=list))

    if include_classification:
        Allowed = Literal[tuple(cats)]  # type: ignore[arg-type]
        list_type = List[Allowed] if not single_label else Allowed  # single label = single Literal
        default = [] if not single_label else None
        fields["categories"] = (list_type, default)

    return create_model("Extraction", **fields)  # type: ignore[misc]


def _build_prompt(
    cats: Sequence[str],
    *,
    include_entities: bool,
    include_classification: bool,
    single_label: bool,
) -> str:
    """Generate a system prompt consistent with the requested outputs."""
    lines: List[str] = ["You are an expert information extractor."]

    if include_entities:
        lines.append(
            "1. List every *named entity* (people, places, organisations, etc.) "
            "as simple canonical strings: `['Joe Biden', 'Afghanistan']`."
        )

    if include_classification:
        label_note = "exactly **one**" if single_label else "all **relevant**"
        cat_line = ", ".join(cats)
        step_nr = "2" if include_entities else "1"
        lines.append(f"{step_nr}. Pick {label_note} category from: {cat_line}.")

    lines.append("Return JSON matching the provided schema.")
    return "\n".join(lines)


def _resp_fmt(schema: Type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "strict": True,
        "schema": schema.model_json_schema(),
    }


# ────────────────────────────────────────────────────────────────────────────
#  Single request
# ────────────────────────────────────────────────────────────────────────────
def parse_one(
    text: str,
    *,
    categories: Sequence[str] = (),
    include_entities: bool = True,
    include_classification: bool = True,
    single_label: bool = False,
    model: str = "gpt-4o-2024-08-06",
    system_prompt: str | None = None,
    client_override: OpenAI | None = None,
) -> dict:
    """
    Extract entities and/or classify *text*.

    Parameters
    ----------
    text : str
    categories : Sequence[str]
        List of allowed category names (ignored if include_classification=False).
    include_entities : bool
    include_classification : bool
    single_label : bool
        If True and classification enabled, exactly one category is returned.
    model : str
        OpenAI snapshot to use.
    system_prompt : str | None
        Override the auto-generated system prompt.
    client_override : OpenAI | None
        Supply a custom OpenAI client.

    Returns
    -------
    dict
        Keys present depend on enabled features.
    """
    if include_classification and not categories:
        raise ValueError("categories must be non-empty when include_classification=True")

    oa = client_override or client
    schema = _build_schema(
        categories,
        include_entities=include_entities,
        include_classification=include_classification,
        single_label=single_label,
    )
    prompt = system_prompt or _build_prompt(
        categories,
        include_entities=include_entities,
        include_classification=include_classification,
        single_label=single_label,
    )

    rsp = oa.responses.parse(
        model=model,
        input=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        text_format=schema,
    )
    out: Union[dict, BaseModel] = rsp.output_parsed
    return out.model_dump() if isinstance(out, BaseModel) else out


# ────────────────────────────────────────────────────────────────────────────
#  Batch helpers
# ────────────────────────────────────────────────────────────────────────────
def prepare_batch_files(
    df: pd.DataFrame,
    text_column: str,
    output_dir: str,
    *,
    categories: Sequence[str] = (),
    include_entities: bool = True,
    include_classification: bool = True,
    single_label: bool = False,
    model_name: str = "gpt-4o-2024-08-06",
    system_prompt: str | None = None,
    temperature: float = 0.0,
    rows_per_batch: int = 10_000,
) -> int:
    """
    Split DataFrame into JSONL files for OpenAI Batch API.

    All feature-toggle kwargs mirror those in `parse_one`.
    """
    if include_classification and not categories:
        raise ValueError("categories must be provided when include_classification=True")

    schema = _build_schema(
        categories,
        include_entities=include_entities,
        include_classification=include_classification,
        single_label=single_label,
    )
    prompt = system_prompt or _build_prompt(
        categories,
        include_entities=include_entities,
        include_classification=include_classification,
        single_label=single_label,
    )
    fmt = _resp_fmt(schema)

    total_batches = math.ceil(len(df) / rows_per_batch)
    df = df.copy()
    df["_batch"] = df.index // rows_per_batch + 1
    os.makedirs(output_dir, exist_ok=True)

    for n in range(1, total_batches + 1):
        subset = df[df["_batch"] == n]
        lines: List[str] = []
        for idx, row in subset.iterrows():
            body = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(row[text_column])},
                ],
                "temperature": temperature,
                "response_format": fmt,
            }
            lines.append(
                json.dumps(
                    {
                        "custom_id": f"req-{idx}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )
        with open(os.path.join(output_dir, f"batch_{n:03d}.jsonl"), "w") as fh:
            fh.write("\n".join(lines))
    return total_batches


def upload_and_process_batches(
    directory: str,
    batch_info_csv: str,
    *,
    model: str = "gpt-4o-2024-08-06",
    client_override: OpenAI | None = None,
) -> List[Tuple[str, str]]:
    """
    Upload JSONL files in *directory* to the Batch API and log job IDs.
    """
    oa = client_override or client
    files = sorted(p for p in os.listdir(directory) if p.endswith(".jsonl"))
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
        rows.append(
            {"timestamp": ts, "batch_file": f, "file_id": up.id, "batch_id": job.id, "status": "submitted"}
        )

    if rows:
        os.makedirs(os.path.dirname(batch_info_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(
            batch_info_csv,
            mode="a" if os.path.exists(batch_info_csv) else "w",
            header=not os.path.exists(batch_info_csv),
            index=False,
        )
    return processed



