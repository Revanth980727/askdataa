"""
RLHF Service for feedback collection and adapter training.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings

# Add the contracts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "contracts"))
from mcp_tools import (
    CaptureFeedbackInput,
    CaptureFeedbackOutput,
    PromoteAdapterInput,
    PromoteAdapterOutput,
)


class Settings(BaseSettings):
    """Application settings."""

    environment: str = "local"
    log_level: str = "INFO"
    feedback_dir: str = "./data/feedback"
    model_registry_path: str = "./data/models/model_registry.json"
    models_dir: str = "./data/models"


class ModelRegistry:
    """Simple file-based model registry."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.models, f, indent=2)

    def set_adapter(self, connection_id: str, adapter_model: str, description: Optional[str] = None) -> None:
        if connection_id not in self.models:
            self.models[connection_id] = {}
        self.models[connection_id]["adapter_model"] = adapter_model
        if description:
            self.models[connection_id]["description"] = description
        self.models[connection_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save()


class RLHFService:
    """Service implementing feedback capture and adapter promotion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.feedback_dir = Path(settings.feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(settings.model_registry_path)
        self.models_dir = Path(settings.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _feedback_path(self, connection_id: str) -> Path:
        return self.feedback_dir / f"{connection_id}.jsonl"

    def save_feedback(self, payload: CaptureFeedbackInput) -> None:
        path = self._feedback_path(payload.active_connection_id)
        event = payload.model_dump()
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def load_feedback(self, connection_id: str) -> List[Dict[str, Any]]:
        path = self._feedback_path(connection_id)
        if not path.exists():
            return []
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def build_preference_pairs(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        events_by_run: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            events_by_run.setdefault(event["run_id"], []).append(event)

        for run_id, run_events in events_by_run.items():
            approvals = [e for e in run_events if e["event_type"] == "approval"]
            rejections = [e for e in run_events if e["event_type"] == "rejection"]
            edits = [e for e in run_events if e["event_type"] == "edit"]

            for edit in edits:
                if edit.get("edited_response"):
                    pairs.append({
                        "prompt": edit["prompt"],
                        "chosen": edit["edited_response"],
                        "rejected": edit["response"],
                        "reward_chosen": 1.0,
                        "reward_rejected": -1.0,
                    })

            for approval in approvals:
                for rejection in rejections:
                    pairs.append({
                        "prompt": approval["prompt"],
                        "chosen": approval["response"],
                        "rejected": rejection["response"],
                        "reward_chosen": 1.0,
                        "reward_rejected": -1.0,
                    })
        return pairs

    def train_adapter(self, connection_id: str, pairs: List[Dict[str, Any]]) -> str:
        if not pairs:
            raise ValueError("No preference pairs available for training")
        adapter_name = f"{connection_id}_adapter_{int(time.time())}"
        adapter_path = self.models_dir / f"{adapter_name}.json"
        with open(adapter_path, "w") as f:
            json.dump({"connection_id": connection_id, "pairs": pairs}, f, indent=2)
        return f"lora:{adapter_name}"

    def promote_adapter(self, connection_id: str, description: Optional[str] = None) -> str:
        events = self.load_feedback(connection_id)
        pairs = self.build_preference_pairs(events)
        adapter_model = self.train_adapter(connection_id, pairs)
        self.registry.set_adapter(connection_id, adapter_model, description)
        return adapter_model


# Initialize logging
settings = Settings()
logging.basicConfig(level=settings.log_level)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

service = RLHFService(settings)

app = FastAPI(title="RLHF Service", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"]
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/capture_feedback", response_model=CaptureFeedbackOutput)
async def capture_feedback(payload: CaptureFeedbackInput) -> CaptureFeedbackOutput:
    try:
        service.save_feedback(payload)
        return CaptureFeedbackOutput(status="ok")
    except Exception as e:
        logging.error(f"Failed to capture feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/promote_adapter", response_model=PromoteAdapterOutput)
async def promote_adapter(payload: PromoteAdapterInput) -> PromoteAdapterOutput:
    try:
        adapter_model = service.promote_adapter(payload.connection_id, payload.description)
        return PromoteAdapterOutput(adapter_model=adapter_model, message="adapter promoted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Failed to promote adapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
