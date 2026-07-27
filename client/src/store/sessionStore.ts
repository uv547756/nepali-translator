/**
 * Zustand store — global UI state for the translation session.
 *
 * Transcript entries accumulate over the session.
 * Metrics are overwritten on each update event.
 * Connection status drives the UI button states.
 */

import { create } from 'zustand';

export interface TranscriptEntry {
  id: string;
  nepali: string;
  english: string;
  confidence: number;
  timestamp: number;
  isPartial: boolean;
}

export interface MetricsState {
  asr_ms: number;
  translation_ms: number;
  tts_ms: number;
  total_ms: number;
  gpu_util_pct: number;
  gpu_mem_used_gb: number;
  gpu_mem_total_gb: number;
  utterances: number;
  wps: number;
}

export type SessionStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'listening'
  | 'processing'
  | 'error'
  | 'disconnected';

export interface SessionStore {
  status: SessionStatus;
  isMuted: boolean;
  targetLang: string;
  inputDeviceId: string | undefined;
  outputDeviceId: string | undefined;
  transcript: TranscriptEntry[];
  partialNepali: string;
  metrics: MetricsState | null;
  errorMessage: string | null;
  isPlayingAudio: boolean;

  // Actions
  setStatus: (s: SessionStatus) => void;
  setMuted: (m: boolean) => void;
  setTargetLang: (lang: string) => void;
  setInputDeviceId: (id: string | undefined) => void;
  setOutputDeviceId: (id: string | undefined) => void;
  setPartialNepali: (text: string) => void;
  addFinalEntry: (entry: TranscriptEntry) => void;
  setMetrics: (m: MetricsState) => void;
  setError: (msg: string | null) => void;
  setPlayingAudio: (playing: boolean) => void;
  clearTranscript: () => void;
  reset: () => void;
}

export const useSessionStore = create<SessionStore>((set) => ({
  status: 'idle',
  isMuted: false,
  targetLang: 'eng',
  inputDeviceId: undefined,
  outputDeviceId: undefined,
  transcript: [],
  partialNepali: '',
  metrics: null,
  errorMessage: null,
  isPlayingAudio: false,

  setStatus: (s) => set({ status: s }),
  setMuted: (m) => set({ isMuted: m }),
  setTargetLang: (lang) => set({ targetLang: lang }),
  setInputDeviceId: (id) => set({ inputDeviceId: id }),
  setOutputDeviceId: (id) => set({ outputDeviceId: id }),
  setPartialNepali: (text) => set({ partialNepali: text }),
  addFinalEntry: (entry) =>
    set((state) => ({
      transcript: [
        ...state.transcript.slice(-99), // keep last 100 entries
        entry,
      ],
      partialNepali: '',
    })),
  setMetrics: (m) => set({ metrics: m }),
  setError: (msg) => set({ errorMessage: msg }),
  setPlayingAudio: (playing) => set({ isPlayingAudio: playing }),
  clearTranscript: () => set({ transcript: [], partialNepali: '' }),
  reset: () =>
    set({
      status: 'idle',
      isMuted: false,
      transcript: [],
      partialNepali: '',
      metrics: null,
      errorMessage: null,
      isPlayingAudio: false,
    }),
}));
