/**
 * MicCapture — manages getUserMedia + AudioWorklet for microphone capture.
 *
 * Capture path:
 *   getUserMedia → MediaStreamSource → CaptureProcessor (AudioWorklet)
 *     → postMessage(Int16Array) → onAudioChunk callback
 *
 * The AudioWorklet runs on the real-time audio thread and resamples to 16kHz
 * before posting Int16 PCM chunks to the main thread.
 */

export type AudioChunkCallback = (int16Buffer: ArrayBuffer) => void;

export interface CaptureOptions {
  deviceId?: string;
  onAudioChunk: AudioChunkCallback;
  onError?: (err: Error) => void;
}

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private _active = false;

  async start(options: CaptureOptions): Promise<void> {
    if (this._active) {
      throw new Error('MicCapture is already running');
    }

    const constraints: MediaStreamConstraints = {
      audio: {
        deviceId: options.deviceId ? { exact: options.deviceId } : undefined,
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: false, // server-side DeepFilterNet handles this
        autoGainControl: true,
      },
      video: false,
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);

    // AudioContext: use the actual device rate (browser may ignore sampleRate hint)
    this.ctx = new AudioContext({
      latencyHint: 'interactive',
    });

    // Load the capture AudioWorklet processor
    await this.ctx.audioWorklet.addModule('/worklets/capture-processor.js');

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.workletNode = new AudioWorkletNode(this.ctx, 'capture-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'discrete',
    });

    this.workletNode.port.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'audio' && e.data.buffer) {
        options.onAudioChunk(e.data.buffer);
      }
    };

    this.workletNode.port.onmessageerror = () => {
      options.onError?.(new Error('AudioWorklet message error'));
    };

    this.source.connect(this.workletNode);
    this._active = true;
  }

  stop(): void {
    if (!this._active) return;
    this._active = false;

    this.workletNode?.port.postMessage({ type: 'stop' });
    this.workletNode?.disconnect();
    this.source?.disconnect();

    this.stream?.getTracks().forEach((t) => t.stop());

    if (this.ctx && this.ctx.state !== 'closed') {
      this.ctx.close().catch(() => {});
    }

    this.workletNode = null;
    this.source = null;
    this.stream = null;
    this.ctx = null;
  }

  get isActive(): boolean {
    return this._active;
  }

  /** List available audio input devices. */
  static async listInputDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'audioinput');
  }

  /** List available audio output devices. */
  static async listOutputDevices(): Promise<MediaDeviceInfo[]> {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'audiooutput');
  }
}
