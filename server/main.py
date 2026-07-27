"""
Nepali Speech Translator — server entry point.

Modes:
  server (default): Start the FastAPI + uvicorn server. Serves the web UI
    at / and the WebSocket translation endpoint at /ws/translate.

  offline: Transcribe a WAV file without starting the HTTP server.
    Useful for testing the inference pipeline without a browser.

Usage:
  python -m server.main --config configs/default.yaml
  python -m server.main --mode offline --input test.wav --output translated.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import structlog

# NOTE: This import must stay at module level. `from __future__ import
# annotations` turns every annotation into a string, and FastAPI resolves
# those strings against the function's __globals__ (the module namespace).
# If WebSocket is imported inside run_server() it is only a function local,
# the forward reference fails to resolve, and FastAPI silently downgrades
# the parameter to a query parameter -- rejecting every handshake with 403.
from fastapi import WebSocket


def _configure_logging(level: str, fmt: str) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if fmt == "text"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nepali real-time speech translator"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "offline"],
        default="server",
        help="Run mode (default: server)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override server host (default from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server port (default from config)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="[offline mode] Input WAV file path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="[offline mode] Output WAV file path",
    )
    return parser.parse_args()


async def run_server(config_path: str, host: str | None, port: int | None) -> None:
    """Start the FastAPI + uvicorn server."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    from server.core.config import Config
    from server.core.metrics import MetricsCollector
    from server.core.pipeline import PipelineFactory
    from server.core.scheduler import ModelRegistry
    from server.api import rest as rest_module
    from server.api.websocket import handle_translate_stream

    config = Config.from_yaml(config_path) if Path(config_path).exists() else Config.default()
    _configure_logging(config.system.log_level, config.system.log_format)

    logger = structlog.get_logger(__name__)
    logger.info("Starting translator server", config=config_path)

    # Load all models
    registry = ModelRegistry(config)
    bundle = await registry.load_all()

    metrics = MetricsCollector(
        window_size=config.monitoring.latency_window_size,
        device_index=config.system.cuda_device,
    )
    await metrics.start_gpu_poller()

    factory = PipelineFactory(bundle, config, metrics)

    rest_module.configure(registry, factory, metrics, models_dir="models")

    # Build FastAPI app
    app = FastAPI(
        title="Nepali Speech Translator",
        version="1.0.0",
        description="Real-time Nepali → English/Hindi speech translation",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rest_module.router)

    # FastAPI closes the socket silently on a validation failure, which makes
    # a rejected handshake indistinguishable from a routing miss. Log it.
    from fastapi.exceptions import WebSocketRequestValidationError

    @app.exception_handler(WebSocketRequestValidationError)
    async def _ws_validation_failed(
        websocket: WebSocket,
        exc: WebSocketRequestValidationError,
    ) -> None:
        logger.error("WebSocket handshake rejected by validation", errors=exc.errors())
        await websocket.close(code=1008)

    @app.websocket("/ws/translate")
    async def ws_translate(websocket: WebSocket) -> None:
        await handle_translate_stream(websocket, factory)

    # Prometheus metrics endpoint
    if config.monitoring.prometheus.enabled:
        try:
            from prometheus_client import make_asgi_app
            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
        except ImportError:
            logger.warning("prometheus_client not installed — /metrics unavailable")

    # Serve the compiled web client if the dist directory exists
    client_dist = Path("client/dist")
    if client_dist.exists():
        app.mount("/", StaticFiles(directory=str(client_dist), html=True), name="ui")
    else:
        @app.get("/")
        async def root():
            return {
                "message": "Nepali Speech Translator API",
                "ws": "/ws/translate",
                "docs": "/docs",
                "status": "/api/v1/status",
            }

    @app.on_event("shutdown")
    async def on_shutdown():
        await metrics.stop_gpu_poller()
        await registry.unload_all()

    srv_host = host or config.server.host
    srv_port = port or config.server.port

    # Route table, so a missing/shadowed WebSocket route is visible at a glance
    logger.info(
        "Registered routes",
        routes=[
            f"{type(r).__name__}:{getattr(r, 'path', '?')}"
            for r in app.routes
        ],
    )

    logger.info("Server ready", host=srv_host, port=srv_port)

    uv_config = uvicorn.Config(
        app,
        host=srv_host,
        port=srv_port,
        log_level="warning",   # uvicorn logs suppressed; structlog handles our logging
        ws_ping_interval=20,
        ws_ping_timeout=60,
    )
    if config.server.tls.enabled:
        uv_config.ssl_certfile = config.server.tls.cert_file
        uv_config.ssl_keyfile = config.server.tls.key_file

    server = uvicorn.Server(uv_config)
    await server.serve()


async def run_offline(config_path: str, input_path: str, output_path: str) -> None:
    """Run the inference pipeline on a WAV file without starting HTTP server."""
    import soundfile as sf
    import numpy as np

    from server.core.config import Config
    from server.core.metrics import MetricsCollector
    from server.core.scheduler import ModelRegistry

    config = Config.from_yaml(config_path) if Path(config_path).exists() else Config.default()
    _configure_logging(config.system.log_level, config.system.log_format)

    logger = structlog.get_logger(__name__)
    logger.info("Offline mode", input=input_path, output=output_path)

    registry = ModelRegistry(config)
    bundle = await registry.load_all()

    # Read and resample input audio to 16kHz mono float32
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal as ss
        audio = ss.resample_poly(audio, 16000, sr).astype(np.float32)

    logger.info("Audio loaded", samples=len(audio), duration_s=round(len(audio) / 16000, 2))

    # Run ASR
    from server.asr.engine import ASRResult
    result: ASRResult = await bundle.asr.transcribe(audio)
    logger.info("ASR complete", text=result.text, confidence=round(result.confidence, 3))

    # Translate
    translation = await bundle.translator.translate(
        result.text,
        config.translation.source_lang,
        config.translation.target_lang,
    )
    logger.info("Translation complete", translated=translation.translated_text)

    # Synthesize
    tts_result = await bundle.tts.synthesize(translation.translated_text)
    logger.info("TTS complete", duration_s=round(tts_result.duration_s, 2))

    # Write output
    sf.write(output_path, tts_result.audio, tts_result.sample_rate)
    logger.info("Output written", path=output_path)

    await registry.unload_all()


def main() -> None:
    args = _parse_args()

    if args.mode == "offline":
        if not args.input or not args.output:
            print("--input and --output are required in offline mode", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_offline(args.config, args.input, args.output))
    else:
        asyncio.run(run_server(args.config, args.host, args.port))


if __name__ == "__main__":
    main()
