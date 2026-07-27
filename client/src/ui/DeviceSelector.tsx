import React, { useEffect, useState } from 'react';
import { useSessionStore } from '../store/sessionStore';
import { MicCapture } from '../audio/MicCapture';

export const DeviceSelector: React.FC = () => {
  const { inputDeviceId, outputDeviceId, setInputDeviceId, setOutputDeviceId } =
    useSessionStore();

  const [inputDevices, setInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [outputDevices, setOutputDevices] = useState<MediaDeviceInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const refreshDevices = async () => {
    setLoading(true);
    try {
      // Request permission first so labels are populated
      await navigator.mediaDevices.getUserMedia({ audio: true }).then((s) =>
        s.getTracks().forEach((t) => t.stop())
      );
      const ins = await MicCapture.listInputDevices();
      const outs = await MicCapture.listOutputDevices();
      setInputDevices(ins);
      setOutputDevices(outs);
    } catch {
      // Permission denied — show empty lists
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshDevices();
    navigator.mediaDevices.addEventListener('devicechange', refreshDevices);
    return () =>
      navigator.mediaDevices.removeEventListener('devicechange', refreshDevices);
  }, []);

  return (
    <div className="device-selector">
      <div className="device-row">
        <label htmlFor="input-device" className="device-label">
          🎙 Microphone
        </label>
        <select
          id="input-device"
          className="device-select"
          value={inputDeviceId ?? ''}
          onChange={(e) => setInputDeviceId(e.target.value || undefined)}
          disabled={loading}
        >
          <option value="">Default</option>
          {inputDevices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Microphone ${d.deviceId.slice(0, 6)}`}
            </option>
          ))}
        </select>
      </div>

      <div className="device-row">
        <label htmlFor="output-device" className="device-label">
          🔊 Speaker
        </label>
        <select
          id="output-device"
          className="device-select"
          value={outputDeviceId ?? ''}
          onChange={(e) => setOutputDeviceId(e.target.value || undefined)}
          disabled={loading}
        >
          <option value="">Default</option>
          {outputDevices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Speaker ${d.deviceId.slice(0, 6)}`}
            </option>
          ))}
        </select>
      </div>

      <button
        className="btn btn-ghost btn-sm"
        onClick={refreshDevices}
        disabled={loading}
        aria-label="Refresh device list"
      >
        ↻ Refresh
      </button>
    </div>
  );
};
