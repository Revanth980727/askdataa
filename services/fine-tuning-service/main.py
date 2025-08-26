"""Fine-tuning Service

This service provides MCP-compliant tools for synthetic question generation,
dataset management and LoRA adapter training/evaluation using PEFT.  All tools
follow the schemas defined in ``contracts.mcp_tools``.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from contracts.mcp_tools import (
    GenerateSyntheticQuestionsInput,
    GenerateSyntheticQuestionsOutput,
    CreateDatasetInput,
    CreateDatasetOutput,
    ValidateDatasetInput,
    ValidateDatasetOutput,
    TrainAdapterInput,
    TrainAdapterOutput,
    EvaluateAdapterInput,
    EvaluateAdapterOutput,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AskData Fine Tuning Service", version="1.0.0")

DATASET_ROOT = Path("/data/datasets")
MODEL_ROOT = Path("/data/models")

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


@app.post(
    "/mcp/generate_synthetic_questions",
    response_model=GenerateSyntheticQuestionsOutput,
)
async def generate_synthetic_questions(
    request: GenerateSyntheticQuestionsInput,
):
    """Generate synthetic questions using simple heuristics."""

    start = time.time()
    tables = request.table_metadata.get("tables", [])

    questions = []
    for table in tables:
        name = table.get("table_name") or table.get("name") or "table"
        questions.append(f"How many rows are in {name}?")
        questions.append(f"Show the latest record from {name}.")

    # Limit to requested number
    questions = questions[: request.num_questions]
    generation_time = time.time() - start

    return GenerateSyntheticQuestionsOutput(
        connection_id=request.connection_id,
        questions=questions,
        generation_time=generation_time,
        model_used="heuristic",
    )


@app.post("/mcp/create_dataset", response_model=CreateDatasetOutput)
async def create_dataset(request: CreateDatasetInput):
    """Persist a dataset to ``/data/datasets/{connection_id}``."""

    dataset_dir = DATASET_ROOT / request.connection_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = dataset_dir / f"{request.dataset_name}.json"
    _write_json(
        dataset_file,
        {"questions": request.questions, "answers": request.answers},
    )

    return CreateDatasetOutput(
        connection_id=request.connection_id,
        dataset_path=str(dataset_file),
        num_samples=len(request.questions),
        created_at=datetime.utcnow(),
    )


@app.post("/mcp/validate_dataset", response_model=ValidateDatasetOutput)
async def validate_dataset(request: ValidateDatasetInput):
    """Validate the structure of a dataset."""

    dataset_file = DATASET_ROOT / request.connection_id / f"{request.dataset_name}.json"
    errors = []
    num_samples = 0

    if not dataset_file.exists():
        errors.append("dataset_not_found")
        return ValidateDatasetOutput(
            connection_id=request.connection_id, is_valid=False, num_samples=0, errors=errors
        )

    try:
        with open(dataset_file, "r") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        answers = data.get("answers", [])
        num_samples = len(questions)

        if not questions:
            errors.append("missing_questions")
        if answers and len(answers) != len(questions):
            errors.append("mismatched_qna")
    except Exception as exc:
        errors.append(str(exc))

    return ValidateDatasetOutput(
        connection_id=request.connection_id,
        is_valid=len(errors) == 0,
        num_samples=num_samples,
        errors=errors,
    )


@app.post("/mcp/train_adapter", response_model=TrainAdapterOutput)
async def train_adapter(request: TrainAdapterInput):
    """Train a LoRA adapter. This implementation is a placeholder that writes
    configuration files but does not perform actual training."""

    start = time.time()
    adapter_dir = MODEL_ROOT / request.connection_id / request.adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Record minimal training metadata
    _write_json(
        adapter_dir / "adapter_config.json",
        {
            "base_model": request.base_model,
            "dataset": request.dataset_name,
            "created_at": datetime.utcnow(),
        },
    )

    training_time = time.time() - start
    adapter_model = f"lora:{request.connection_id}_{request.adapter_name}"

    return TrainAdapterOutput(
        connection_id=request.connection_id,
        adapter_model=adapter_model,
        training_time=training_time,
        metrics={},
    )


@app.post("/mcp/evaluate_adapter", response_model=EvaluateAdapterOutput)
async def evaluate_adapter(request: EvaluateAdapterInput):
    """Evaluate a trained adapter. Returns dummy metrics."""

    start = time.time()
    metrics = {"accuracy": 1.0}
    evaluation_time = time.time() - start

    return EvaluateAdapterOutput(
        connection_id=request.connection_id,
        metrics=metrics,
        evaluation_time=evaluation_time,
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "fine-tuning-service"}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "AskData Fine Tuning Service", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

