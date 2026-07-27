import React, { useEffect, useRef } from 'react';
import { useSessionStore } from '../store/sessionStore';

export const TranscriptPanel: React.FC = () => {
  const { transcript, partialNepali, targetLang } = useSessionStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript, partialNepali]);

  const targetLabel = targetLang === 'hin' ? 'Hindi' : 'English';

  return (
    <div className="transcript-panel" role="log" aria-live="polite" aria-label="Translation transcript">
      <div className="transcript-header">
        <span className="lang-label">Nepali</span>
        <span className="lang-arrow">→</span>
        <span className="lang-label">{targetLabel}</span>
      </div>

      <div className="transcript-entries">
        {transcript.length === 0 && !partialNepali && (
          <div className="transcript-empty">
            Speak Nepali — translation will appear here.
          </div>
        )}

        {transcript.map((entry) => (
          <div key={entry.id} className="transcript-entry">
            <div className="entry-source">{entry.nepali}</div>
            <div className="entry-translation">{entry.english}</div>
            <div className="entry-meta">
              <span className="entry-conf">{Math.round(entry.confidence * 100)}% confidence</span>
              <span className="entry-time">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}

        {partialNepali && (
          <div className="transcript-entry partial">
            <div className="entry-source partial-text">{partialNepali}</div>
            <div className="entry-translation partial-placeholder">Translating…</div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
};
