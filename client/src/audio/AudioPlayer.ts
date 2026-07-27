/**
 * AudioPlayer — plays back Int16 LE PCM chunks via Web Audio API.
 *
 * Playback path:
 *   WebSocket binary frames (Int16 PCM, 22050 Hz)
 *     → AudioPlayer.push(buffer)
 *     → postMessage to PlaybackProcessor (AudioWorklet jitter buffer)
 *     → AudioContext destination (user's chosen output device)
 *
 * The jitter buffer in the worklet absorbs network latency variation.
 * Target buffer depth: 150ms (configurable). Glitch-free playback
 * requires the buffer to never run empty during continuous speech.
 */

export interface PlayerOptions {
  serverSampleRate?: number;   // default 22050 (Piper)
  outputDeviceId?: string;
  onPlaybackStart?: () => void;
  onPlaybackEnd?: () => void;
}

export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private _active = false;
  private _serverSR: number;
  private _outputDeviceId: string | undefined;
  private _onStart?: () => void;
  private _onEnd?: () => void;

  constructor(options: PlayerOptions = {}) {
    this._serverSR = options.serverSampleRate ?? 22050;
    this._outputDeviceId = options.outputDeviceId;
    this._onStart = options.onPlaybackStart;
    this._onEnd = options.onPlaybackEnd;
  }

  async init(): Promise<void> {
    if (this._active) return;

    this.ctx = new AudioContext({ latencyHint: 'interactive' });

    await this.ctx.audioWorklet.addModule('/worklets/playback-processor.js');

    this.workletNode = new AudioWorkletNode(this.ctx, 'playback-processor', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: {
        serverSampleRate: this._serverSR,
      },
    });

    this.workletNode.connect(this.ctx.destination);

    // Set output device if specified (Chrome only via setSinkId)
    if (this._outputDeviceId && 'setSinkId' in this.ctx) {
      await (this.ctx as any).setSinkId(this._outputDeviceId);
    }

    this._active = true;
  }

  /** Push a PCM audio chunk to the jitter buffer. */
  push(int16Buffer: ArrayBuffer): void {
    if (!this._active || !this.workletNode) return;

    // Transfer ownership to worklet for zero-copy (on supporting browsers)
    const copy = int16Buffer.slice(0);
    this.workletNode.port.postMessage({ type: 'audio', buffer: copy }, [copy]);
    this._onStart?.();
  }

  /** Signal end of utterance — caller can track when speech ends. */
  onUtteranceEnd(): void {
    this._onEnd?.();
  }

  /** Flush the jitter buffer (e.g., on mute or session reset). */
  flush(): void {
    this.workletNode?.port.postMessage({ type: 'flush' });
  }

  stop(): void {
    if (!this._active) return;
    this._active = false;

    this.workletNode?.port.postMessage({ type: 'stop' });
    this.workletNode?.disconnect();
    this.ctx?.close().catch(() => {});

    this.workletNode = null;
    this.ctx = null;
  }

  /** Resume AudioContext after a user gesture (required by browsers). */
  async resume(): Promise<void> {
    if (this.ctx?.state === 'suspended') {
      await this.ctx.resume();
    }
  }

  get isActive(): boolean {
    return this._active;
  }
}
