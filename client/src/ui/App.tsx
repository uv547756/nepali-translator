/**
 * Root application component.
 *
 * Owns the session lifecycle: connects WebSocket, starts mic capture,
 * routes incoming events to the Zustand store, and pushes audio to the player.
 */

import React, { useCallback, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';

import { AudioPlayer } from '../audio/AudioPlayer';
import { MicCapture } from '../audio/MicCapture';
import { WebSocketTransport } from '../transport/WebSocketTransport';
import { useSessionStore } from '../store/sessionStore';
import { ControlBar } from './ControlBar';
import { DeviceSelector } from './DeviceSelector';
import { MetricsPanel } from './MetricsPanel';
import { TranscriptPanel } from './TranscriptPanel';

function buildWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/translate`;
}

export const App: React.FC = () => {
  const store = useSessionStore();
  const transportRef = useRef<WebSocketTransport | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);

  const handleStart = useCallback(async () => {
    store.setStatus('connecting');
    store.setError(null);

    // Init audio player first (requires user gesture)
    const player = new AudioPlayer({
      serverSampleRate: 22050,
      outputDeviceId: store.outputDeviceId,
      onPlaybackStart: () => store.setPlayingAudio(true),
      onPlaybackEnd: () => store.setPlayingAudio(false),
    });
    await player.init();
    playerRef.current = player;

    // Connect WebSocket
    const transport = new WebSocketTransport({
      url: buildWsUrl(),
      onStatusChange: (s) => {
        if (s === 'connected') store.setStatus('connected');
        else if (s === 'disconnected') store.setStatus('disconnected');
        else if (s === 'error') store.setError('Connection failed');
      },
      onJsonEvent: (event) => handleServerEvent(event, store, player),
      onBinaryAudio: (buf) => {
        player.push(buf);
        player.resume();
      },
    });
    transportRef.current = transport;
    transport.connect();

    // Wait briefly for WS to open before starting mic
    await new Promise<void>((resolve) => {
      const check = setInterval(() => {
        if (transport.status === 'connected') {
          clearInterval(check);
          resolve();
        }
      }, 50);
      setTimeout(() => { clearInterval(check); resolve(); }, 3000);
    });

    // Send initial config
    transport.sendJSON({
      type: 'config',
      source_lang: 'npi',
      target_lang: store.targetLang,
    });

    // Start microphone capture
    const mic = new MicCapture();
    micRef.current = mic;
    try {
      await mic.start({
        deviceId: store.inputDeviceId,
        onAudioChunk: (buf) => transport.sendBinary(buf),
        onError: (err) => store.setError(err.message),
      });
      store.setStatus('listening');
    } catch (err) {
      store.setError(`Microphone error: ${(err as Error).message}`);
      store.setStatus('error');
    }
  }, [store]);

  const handleStop = useCallback(() => {
    micRef.current?.stop();
    micRef.current = null;

    transportRef.current?.sendJSON({ type: 'session_end' });
    transportRef.current?.close();
    transportRef.current = null;

    playerRef.current?.flush();
    playerRef.current?.stop();
    playerRef.current = null;

    store.setStatus('idle');
    store.setPlayingAudio(false);
  }, [store]);

  const handleMuteToggle = useCallback(() => {
    const next = !store.isMuted;
    store.setMuted(next);
    transportRef.current?.sendJSON({ type: next ? 'mute' : 'unmute' });
  }, [store]);

  const handleLangChange = useCallback((lang: string) => {
    store.setTargetLang(lang);
    transportRef.current?.sendJSON({ type: 'config', target_lang: lang });
  }, [store]);

  // Cleanup on unmount
  useEffect(() => {
    return () => { handleStop(); };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Nepali Translator</h1>
        <p className="app-subtitle">Real-time Nepali → English / Hindi</p>
      </header>

      <div className="app-body">
        <main className="app-main">
          <ControlBar
            onStart={handleStart}
            onStop={handleStop}
            onMuteToggle={handleMuteToggle}
            onLangChange={handleLangChange}
            onClear={store.clearTranscript}
          />

          {store.errorMessage && (
            <div className="error-banner" role="alert">
              {store.errorMessage}
            </div>
          )}

          <TranscriptPanel />
        </main>

        <aside className="app-sidebar">
          <DeviceSelector />
          <MetricsPanel />
        </aside>
      </div>
    </div>
  );
};

function handleServerEvent(
  event: Record<string, unknown>,
  store: ReturnType<typeof useSessionStore.getState>,
  player: AudioPlayer,
): void {
  const type = event.type as string;

  if (type === 'vad_event') {
    const state = event.state as string;
    if (state === 'speech_start') store.setStatus('processing');
    else if (state === 'speech_end') store.setStatus('listening');

  } else if (type === 'partial_transcript') {
    store.setPartialNepali(event.text as string);

  } else if (type === 'final_transcript') {
    store.addFinalEntry({
      id: uuidv4(),
      nepali: event.text as string,
      english: event.translated as string,
      confidence: (event.confidence as number) ?? 0,
      timestamp: Date.now(),
      isPartial: false,
    });

  } else if (type === 'metrics') {
    store.setMetrics({
      asr_ms: (event.asr_ms as number) ?? 0,
      translation_ms: (event.translation_ms as number) ?? 0,
      tts_ms: (event.tts_ms as number) ?? 0,
      total_ms: (event.total_ms as number) ?? 0,
      gpu_util_pct: (event.gpu_util_pct as number) ?? 0,
      gpu_mem_used_gb: (event.gpu_mem_used_gb as number) ?? 0,
      gpu_mem_total_gb: (event.gpu_mem_total_gb as number) ?? 0,
      utterances: (event.utterances as number) ?? 0,
      wps: (event.wps as number) ?? 0,
    });

  } else if (type === 'error') {
    store.setError(`${event.stage}: ${event.message}`);
    if (!(event.recoverable as boolean)) {
      store.setStatus('error');
    }
  }
}
