"""
FastAPI WebSocket handler for the translation stream.

Each browser session connects to /ws/translate.
Incoming: binary PCM frames + JSON control messages.
Outgoing: binary PCM frames (synthesized audio) + JSON event messages.

Backpressure: the pipeline's event_stream() is an async generator;
if the WebSocket send falls behind, the event_q fills up and the pipeline
applies backpressure at the VAD → ASR → TTS stages.
"""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from server.core.pipeline import (
    AudioChunkEvent,
    PipelineErrorEvent,
    PipelineFactory,
)

logger = structlog.get_logger(__name__)


async def handle_translate_stream(
    websocket: WebSocket,
    factory: PipelineFactory,
) -> None:
    """Entry point for each /ws/translate WebSocket connection."""
    await websocket.accept()
    session_id = str(uuid4())
    connect_time = time.monotonic()

    logger.info(
        "WebSocket connected",
        session_id=session_id,
        active_sessions=factory.active_sessions + 1,
    )

    try:
        pipeline = await factory.create_session(session_id)
    except RuntimeError as exc:
        await websocket.send_text(json.dumps({
            "type": "error",
            "stage": "session",
            "message": str(exc),
            "recoverable": False,
            "session_id": session_id,
        }))
        await websocket.close(code=1013)  # 1013 = Try Again Later
        return

    # Task: forward pipeline events to the WebSocket
    async def send_events() -> None:
        async for event in pipeline.event_stream():
            if not _is_connected(websocket):
                break
            try:
                if isinstance(event, AudioChunkEvent):
                    if event.pcm_bytes:  # skip empty sentinel
                        await websocket.send_bytes(event.pcm_bytes)
                else:
                    await websocket.send_text(json.dumps(event.to_dict()))
            except Exception as exc:
                logger.warning("Failed to send event", error=str(exc), session_id=session_id)
                break

    send_task = asyncio.create_task(send_events(), name=f"ws-send-{session_id[:8]}")

    # Main receive loop: binary = audio PCM, text = JSON control
    try:
        async for message in websocket.iter_text():
            await _handle_text_message(message, pipeline, session_id)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("WebSocket receive error", error=str(exc), session_id=session_id)
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass

        await factory.destroy_session(session_id)
        duration_s = time.monotonic() - connect_time
        logger.info(
            "WebSocket disconnected",
            session_id=session_id,
            duration_s=round(duration_s, 1),
        )


async def handle_audio_bytes(
    websocket: WebSocket,
    factory: PipelineFactory,
) -> None:
    """Separate endpoint for binary-only audio streams (alternative to combined)."""
    await websocket.accept()
    session_id = str(uuid4())

    try:
        pipeline = await factory.create_session(session_id)
    except RuntimeError as exc:
        await websocket.close(code=1013)
        return

    async def send_events() -> None:
        async for event in pipeline.event_stream():
            if not _is_connected(websocket):
                break
            if isinstance(event, AudioChunkEvent) and event.pcm_bytes:
                await websocket.send_bytes(event.pcm_bytes)
            elif not isinstance(event, AudioChunkEvent):
                await websocket.send_text(json.dumps(event.to_dict()))

    send_task = asyncio.create_task(send_events())

    try:
        async for data in websocket.iter_bytes():
            await pipeline.feed_audio(data)
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        await factory.destroy_session(session_id)


async def _handle_text_message(
    raw: str,
    pipeline,
    session_id: str,
) -> None:
    """Parse and act on a JSON control message from the browser."""
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Non-JSON text frame received", session_id=session_id)
        return

    msg_type = msg.get("type", "")

    if msg_type == "audio":
        # Client sends audio as base64-encoded binary in a JSON frame (alternative mode)
        import base64
        pcm_bytes = base64.b64decode(msg.get("data", ""))
        await pipeline.feed_audio(pcm_bytes)

    elif msg_type == "config":
        target_lang = msg.get("target_lang")
        if target_lang:
            pipeline.update_target_lang(target_lang)

    elif msg_type == "mute":
        pipeline.set_muted(True)

    elif msg_type == "unmute":
        pipeline.set_muted(False)

    elif msg_type == "session_start":
        pass   # Pipeline is already started on connect

    elif msg_type == "session_end":
        pipeline.set_muted(True)

    elif msg_type == "ping":
        pass   # Keep-alive; no response needed (WebSocket handles it)

    elif msg_type == "binary_audio":
        # Client sends PCM bytes directly in binary frames — handled by iter_bytes
        pass

    else:
        logger.debug("Unknown message type", msg_type=msg_type, session_id=session_id)


def _is_connected(websocket: WebSocket) -> bool:
    """Check if the WebSocket connection is still open."""
    try:
        return websocket.client_state.value == 1  # CONNECTED
    except Exception:
        return False
