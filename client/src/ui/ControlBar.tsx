import React from 'react';
import { useSessionStore } from '../store/sessionStore';

interface ControlBarProps {
  onStart: () => void;
  onStop: () => void;
  onMuteToggle: () => void;
  onLangChange: (lang: string) => void;
  onClear: () => void;
}

const TARGET_LANGS = [
  { code: 'eng', label: 'English' },
  { code: 'hin', label: 'Hindi' },
];

export const ControlBar: React.FC<ControlBarProps> = ({
  onStart,
  onStop,
  onMuteToggle,
  onLangChange,
  onClear,
}) => {
  const { status, isMuted, targetLang } = useSessionStore();

  const isRunning = status !== 'idle' && status !== 'disconnected' && status !== 'error';
  const canStart = !isRunning;
  const canStop = isRunning;

  return (
    <div className="control-bar">
      <button
        className="btn btn-primary"
        onClick={onStart}
        disabled={!canStart}
        aria-label="Start translation"
      >
        ▶ Start
      </button>

      <button
        className="btn btn-danger"
        onClick={onStop}
        disabled={!canStop}
        aria-label="Stop translation"
      >
        ■ Stop
      </button>

      <button
        className={`btn ${isMuted ? 'btn-warning' : 'btn-secondary'}`}
        onClick={onMuteToggle}
        disabled={!isRunning}
        aria-label={isMuted ? 'Unmute microphone' : 'Mute microphone'}
      >
        {isMuted ? '🔇 Unmute' : '🎙 Mute'}
      </button>

      <select
        className="lang-select"
        value={targetLang}
        onChange={(e) => onLangChange(e.target.value)}
        aria-label="Target translation language"
      >
        {TARGET_LANGS.map((l) => (
          <option key={l.code} value={l.code}>
            → {l.label}
          </option>
        ))}
      </select>

      <button
        className="btn btn-ghost"
        onClick={onClear}
        aria-label="Clear transcript"
      >
        ✕ Clear
      </button>

      <StatusBadge status={status} />
    </div>
  );
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const labels: Record<string, { text: string; cls: string }> = {
    idle: { text: 'Idle', cls: 'status-idle' },
    connecting: { text: 'Connecting…', cls: 'status-connecting' },
    connected: { text: 'Ready', cls: 'status-ready' },
    listening: { text: '● Listening', cls: 'status-listening' },
    processing: { text: '⟳ Processing', cls: 'status-processing' },
    error: { text: '⚠ Error', cls: 'status-error' },
    disconnected: { text: 'Disconnected', cls: 'status-idle' },
  };
  const { text, cls } = labels[status] ?? { text: status, cls: '' };
  return <span className={`status-badge ${cls}`}>{text}</span>;
};
