/**
 * Capture AudioWorkletProcessor.
 *
 * Runs on the Web Audio real-time thread. Receives Float32 frames from
 * getUserMedia (128 samples/frame at the AudioContext sample rate), accumulates
 * them into 512-sample chunks, converts to Int16 LE, and posts to main thread.
 *
 * IMPORTANT: This file must never allocate objects inside process() —
 * allocations on the real-time thread cause GC pauses and audio glitches.
 * All buffers are pre-allocated in the constructor.
 */

const TARGET_CHUNK_SAMPLES = 512;
const TARGET_SAMPLE_RATE = 16000;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    // Pre-allocate accumulation buffer (never resize after construction)
    // Max accumulation: 2x chunk size to handle any frame size safely
    this._accum = new Float32Array(TARGET_CHUNK_SAMPLES * 2);
    this._accumLen = 0;

    // Output buffer — reused each flush
    this._int16Out = new Int16Array(TARGET_CHUNK_SAMPLES);

    // Resampling: AudioContext rate → 16kHz
    // sampleRate is a global in AudioWorkletGlobalScope
    this._ratio = TARGET_SAMPLE_RATE / sampleRate;
    this._resampleBuf = new Float32Array(TARGET_CHUNK_SAMPLES * 4); // enough headroom

    this._active = true;
    this.port.onmessage = (e) => {
      if (e.data.type === 'stop') this._active = false;
    };
  }

  process(inputs, outputs, parameters) {
    if (!this._active) return false;

    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channel = input[0]; // mono

    // Resample if needed (linear interpolation — adequate for speech)
    let src;
    if (Math.abs(this._ratio - 1.0) < 0.001) {
      src = channel;
    } else {
      const outLen = Math.round(channel.length * this._ratio);
      for (let i = 0; i < outLen; i++) {
        const srcIdx = i / this._ratio;
        const lo = Math.floor(srcIdx);
        const hi = Math.min(lo + 1, channel.length - 1);
        const frac = srcIdx - lo;
        this._resampleBuf[i] = channel[lo] * (1 - frac) + channel[hi] * frac;
      }
      src = this._resampleBuf.subarray(0, outLen);
    }

    // Accumulate into internal buffer
    for (let i = 0; i < src.length; i++) {
      this._accum[this._accumLen++] = src[i];

      if (this._accumLen >= TARGET_CHUNK_SAMPLES) {
        // Convert Float32 [-1,1] → Int16 and transfer to main thread
        for (let j = 0; j < TARGET_CHUNK_SAMPLES; j++) {
          const s = Math.max(-1, Math.min(1, this._accum[j]));
          this._int16Out[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Transfer ownership of a copy — zero-copy transfer via SharedArrayBuffer
        // would need special headers; we clone for compatibility
        const buf = this._int16Out.buffer.slice(0);
        this.port.postMessage({ type: 'audio', buffer: buf }, [buf]);

        // Shift remaining samples down
        const remaining = this._accumLen - TARGET_CHUNK_SAMPLES;
        this._accum.copyWithin(0, TARGET_CHUNK_SAMPLES, this._accumLen);
        this._accumLen = remaining;
      }
    }

    return true;
  }
}

registerProcessor('capture-processor', CaptureProcessor);
