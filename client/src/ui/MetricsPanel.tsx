import React from 'react';
import { useSessionStore } from '../store/sessionStore';

export const MetricsPanel: React.FC = () => {
  const { metrics, isPlayingAudio } = useSessionStore();

  if (!metrics) {
    return (
      <div className="metrics-panel metrics-empty">
        Metrics will appear once translation starts.
      </div>
    );
  }

  const gpuPct = Math.round(metrics.gpu_util_pct);
  const memUsed = metrics.gpu_mem_used_gb.toFixed(1);
  const memTotal = metrics.gpu_mem_total_gb.toFixed(1);

  return (
    <div className="metrics-panel">
      <h3 className="metrics-title">Performance</h3>

      <div className="metrics-grid">
        <LatencyBar label="ASR" ms={metrics.asr_ms} max={500} warn={250} />
        <LatencyBar label="Translation" ms={metrics.translation_ms} max={200} warn={75} />
        <LatencyBar label="TTS" ms={metrics.tts_ms} max={400} warn={200} />
        <LatencyBar label="End-to-end" ms={metrics.total_ms} max={1500} warn={800} highlight />
      </div>

      <div className="metrics-divider" />

      <div className="metrics-gpu">
        <div className="metric-row">
          <span className="metric-label">GPU utilization</span>
          <GpuBar pct={gpuPct} />
          <span className="metric-value">{gpuPct}%</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">VRAM used</span>
          <span className="metric-value">{memUsed} / {memTotal} GB</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Utterances</span>
          <span className="metric-value">{metrics.utterances}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Words/sec</span>
          <span className="metric-value">{metrics.wps.toFixed(1)}</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Audio output</span>
          <span className={`metric-value ${isPlayingAudio ? 'playing' : ''}`}>
            {isPlayingAudio ? '▶ Playing' : '–'}
          </span>
        </div>
      </div>
    </div>
  );
};

const LatencyBar: React.FC<{
  label: string;
  ms: number;
  max: number;
  warn: number;
  highlight?: boolean;
}> = ({ label, ms, max, warn, highlight = false }) => {
  const pct = Math.min((ms / max) * 100, 100);
  const isWarn = ms > warn;
  return (
    <div className={`latency-row ${highlight ? 'highlight' : ''}`}>
      <span className="latency-label">{label}</span>
      <div className="latency-bar-bg">
        <div
          className={`latency-bar-fill ${isWarn ? 'warn' : 'ok'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`latency-value ${isWarn ? 'warn' : ''}`}>
        {Math.round(ms)} ms
      </span>
    </div>
  );
};

const GpuBar: React.FC<{ pct: number }> = ({ pct }) => (
  <div className="latency-bar-bg" style={{ flex: 1, margin: '0 8px' }}>
    <div
      className={`latency-bar-fill ${pct > 90 ? 'warn' : 'ok'}`}
      style={{ width: `${pct}%` }}
    />
  </div>
);
