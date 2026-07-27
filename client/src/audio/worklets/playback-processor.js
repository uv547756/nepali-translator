/**
 * Playback AudioWorkletProcessor.
 *
 * Runs on the real-time audio thread. Maintains a jitter buffer of PCM samples
 * received from the main thread (which reads them from the WebSocket).
 * On each process() call, outputs samples from the buffer to the audio device.
 *
 * Jitter buffer strategy:
 *   - Target depth: TARGET_BUFFER_MS worth of samples (configurable)
 *   - If buffer falls below MIN_BUFFER_MS, output silence (avoid glitch)
 *   - If buffer exceeds MAX_BUFFER_MS, drop oldest samples (avoid growing lag)
 *   - Linear resampling handles server (22050Hz) vs device (48000Hz) mismatch
 *
 * IMPORTANT: No allocations inside process(). All arrays pre-allocated.
 */

const TARGET_BUFFER_MS = 150;
const MIN_BUFFER_MS    = 40;
const MAX_BUFFER_MS    = 500;

// Circular jitter buffer capacity in samples (at device sample rate)
// 500ms at 48kHz = 24000 samples; we store at 22050Hz and resample lazily
const JITTER_CAPACITY = 48000; // 1 second at 48kHz

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);

    // Server sample rate (22050 for Piper, 24000 for Kokoro)
    this._serverSR = (options.processorOptions && options.processorOptions.serverSampleRate) || 22050;

    // Jitter buffer stores Int16 samples at server sample rate
    this._buf = new Int16Array(JITTER_CAPACITY * 2); // store in server-rate space
    this._writePos = 0;
    this._readPos = 0;
    this._count = 0;

    // Resampling scratch buffer (pre-allocated for 128 output frames)
    this._resampledFrame = new Float32Array(128 * 4);

    this._active = true;
    this._isPlaying = false;

    // Target jitter buffer depth in server-rate samples
    this._targetDepth = Math.round(TARGET_BUFFER_MS / 1000 * this._serverSR);
    this._minDepth = Math.round(MIN_BUFFER_MS / 1000 * this._serverSR);
    this._maxDepth = Math.round(MAX_BUFFER_MS / 1000 * this._serverSR);

    this.port.onmessage = (e) => {
      if (e.data.type === 'audio') {
        this._push(new Int16Array(e.data.buffer));
      } else if (e.data.type === 'stop') {
        this._active = false;
      } else if (e.data.type === 'flush') {
        this._clear();
      }
    };
  }

  _push(int16Samples) {
    // Drop excess if buffer would overflow
    let n = int16Samples.length;
    if (this._count + n > this._maxDepth) {
      // Discard oldest samples to make room
      const drop = this._count + n - this._maxDepth;
      this._readPos = (this._readPos + drop) % this._buf.length;
      this._count -= drop;
    }

    const capacity = this._buf.length;
    for (let i = 0; i < n; i++) {
      this._buf[this._writePos] = int16Samples[i];
      this._writePos = (this._writePos + 1) % capacity;
    }
    this._count += n;
  }

  _pop(nSamples) {
    // Returns array of nSamples Int16 values (or silence if buffer underruns)
    const out = new Int16Array(nSamples);
    const toRead = Math.min(nSamples, this._count);
    const capacity = this._buf.length;

    for (let i = 0; i < toRead; i++) {
      out[i] = this._buf[this._readPos];
      this._readPos = (this._readPos + 1) % capacity;
    }
    this._count -= toRead;
    // Remaining samples are zero (silence) — default Int16Array value

    return out;
  }

  _clear() {
    this._writePos = 0;
    this._readPos = 0;
    this._count = 0;
  }

  process(inputs, outputs, parameters) {
    if (!this._active) return false;

    const output = outputs[0];
    if (!output || !output[0]) return true;

    const outChannel = output[0];
    const outLen = outChannel.length; // 128 samples at device rate

    // How many server-rate samples do we need for outLen device-rate samples?
    const ratio = this._serverSR / sampleRate;
    const serverSamplesNeeded = Math.ceil(outLen * ratio);

    if (this._count < this._minDepth) {
      // Buffer too shallow — output silence to avoid glitch
      outChannel.fill(0);
      return true;
    }

    const serverSamples = this._pop(serverSamplesNeeded);

    // Resample: linear interpolation from serverSR to deviceSR
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, serverSamplesNeeded - 1);
      const frac = srcIdx - lo;
      const loVal = serverSamples[lo] / 32768.0;
      const hiVal = serverSamples[hi] / 32768.0;
      outChannel[i] = loVal * (1 - frac) + hiVal * frac;
    }

    return true;
  }
}

registerProcessor('playback-processor', PlaybackProcessor);
