"""
REST API endpoints.

GET  /api/v1/status   → system health and component status
POST /api/v1/config   → hot-update runtime config
POST /api/v1/models/reload → hot-swap a model
GET  /api/v1/voices   → list available Piper voice models
GET  /metrics         → Prometheus exposition (if prometheus_client is installed)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.core.metrics import MetricsCollector
from server.core.pipeline import PipelineFactory
from server.core.scheduler import ModelRegistry

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1")


# ── Request / response models ───────────────────────────────────────────────

class RuntimeConfigUpdate(BaseModel):
    target_lang: Optional[str] = None
    vad_threshold: Optional[float] = None
    noise_reduction_enabled: Optional[bool] = None
    tts_speed: Optional[float] = None


class ModelReloadRequest(BaseModel):
    component: str   # "asr" | "translation" | "tts"
    model_name: Optional[str] = None


# ── Dependency holders (populated by main.py at startup) ───────────────────
# FastAPI's DI works well here; we use module-level singletons for simplicity.

_registry: Optional[ModelRegistry] = None
_factory: Optional[PipelineFactory] = None
_metrics: Optional[MetricsCollector] = None
_models_dir: str = "models"


def configure(
    registry: ModelRegistry,
    factory: PipelineFactory,
    metrics: MetricsCollector,
    models_dir: str = "models",
) -> None:
    """Called once at startup to inject dependencies."""
    global _registry, _factory, _metrics, _models_dir
    _registry = registry
    _factory = factory
    _metrics = metrics
    _models_dir = models_dir


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status() -> dict:
    """Return system health, VRAM usage, and per-component status."""
    if _registry is None:
        raise HTTPException(status_code=503, detail="Server not fully initialized")

    gpu = _registry.gpu_manager
    bundle = _registry.bundle
    metrics_snapshot = _metrics.snapshot() if _metrics else None

    components: dict = {}
    if bundle:
        components["asr"] = {
            "loaded": bundle.asr.is_loaded,
            "model": bundle.asr.model_name,
            "vram_gb": round(bundle.asr.vram_usage_gb, 2),
        }
        components["translation"] = {
            "loaded": bundle.translator.is_loaded,
            "model": bundle.translator.model_name,
            "vram_gb": round(bundle.translator.vram_usage_gb, 2),
        }
        components["tts"] = {
            "loaded": bundle.tts.is_loaded,
            "model": bundle.tts.model_name,
        }

    return {
        "healthy": bundle is not None,
        "active_sessions": _factory.active_sessions if _factory else 0,
        "components": components,
        "gpu": gpu.to_dict(),
        "metrics": metrics_snapshot.to_dict() if metrics_snapshot else {},
    }


@router.post("/config")
async def update_config(update: RuntimeConfigUpdate) -> dict:
    """Hot-update runtime config without restarting the server."""
    if _registry is None or _registry.bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    changes: list[str] = []

    if update.target_lang is not None:
        # Propagate to all active sessions
        if _factory:
            for pipeline in _factory._sessions.values():
                pipeline.update_target_lang(update.target_lang)
        changes.append(f"target_lang → {update.target_lang}")

    if update.vad_threshold is not None:
        if _registry.bundle:
            _registry.bundle.vad_segmenter._cfg.threshold = update.vad_threshold
        changes.append(f"vad_threshold → {update.vad_threshold}")

    if update.tts_speed is not None:
        if _registry.bundle and hasattr(_registry.bundle.tts, "_config"):
            _registry.bundle.tts._config.length_scale = update.tts_speed
        changes.append(f"tts_speed → {update.tts_speed}")

    logger.info("Runtime config updated", changes=changes)
    return {"success": True, "changes": changes}


@router.post("/models/reload")
async def reload_model(request: ModelReloadRequest) -> dict:
    """Hot-swap a model component without full server restart."""
    if _registry is None or _registry.bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if request.component not in ("asr", "translation", "tts"):
        raise HTTPException(status_code=400, detail="component must be 'asr', 'translation', or 'tts'")

    # Refuse reload if sessions are active (would cause inference races)
    if _factory and _factory.active_sessions > 0:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reload model while {_factory.active_sessions} session(s) are active",
        )

    # For now, full reload of the requested component
    bundle = _registry.bundle
    try:
        if request.component == "tts":
            bundle.tts.unload()
            bundle.tts.load()
        elif request.component == "translation":
            bundle.translator.unload()
            _registry.gpu_manager.register_unloaded("seamless_m4t_v2_large_fp16")
            bundle.translator.load()
            _registry.gpu_manager.register_loaded("seamless_m4t_v2_large_fp16")
        elif request.component == "asr":
            bundle.asr.unload()
            bundle.asr.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "success": True,
        "component": request.component,
        "gpu": _registry.gpu_manager.to_dict(),
    }


@router.get("/voices")
async def list_voices() -> dict:
    """Return available Piper voice model files."""
    piper_dir = Path(_models_dir) / "piper"
    voices: list[dict] = []

    if piper_dir.exists():
        for onnx_file in sorted(piper_dir.glob("*.onnx")):
            voices.append({
                "name": onnx_file.stem,
                "model_path": str(onnx_file),
                "config_path": str(onnx_file) + ".json",
                "has_config": (onnx_file.parent / (onnx_file.name + ".json")).exists(),
            })

    return {"voices": voices}


@router.get("/health")
async def health_check() -> dict:
    healthy = _registry is not None and _registry.bundle is not None
    return {"status": "ok" if healthy else "starting"}
