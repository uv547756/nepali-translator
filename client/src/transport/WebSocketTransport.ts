/**
 * WebSocketTransport — manages the WebSocket connection to the server.
 *
 * Handles:
 *   - Connection lifecycle (connect, reconnect with exponential backoff)
 *   - Sending binary PCM frames
 *   - Sending JSON control messages
 *   - Dispatching incoming JSON events to registered handlers
 *   - Dispatching incoming binary PCM frames (synthesized audio)
 */

export type JSONHandler = (event: Record<string, unknown>) => void;
export type BinaryHandler = (buffer: ArrayBuffer) => void;
export type StatusHandler = (status: ConnectionStatus) => void;

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface TransportOptions {
  url: string;
  onJsonEvent: JSONHandler;
  onBinaryAudio: BinaryHandler;
  onStatusChange: StatusHandler;
  reconnectDelayMs?: number;
  maxReconnectDelayMs?: number;
}

export class WebSocketTransport {
  private ws: WebSocket | null = null;
  private _status: ConnectionStatus = 'disconnected';
  private _reconnectDelay: number;
  private _maxReconnectDelay: number;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _intentionalClose = false;
  private _options: TransportOptions;

  constructor(options: TransportOptions) {
    this._options = options;
    this._reconnectDelay = options.reconnectDelayMs ?? 1000;
    this._maxReconnectDelay = options.maxReconnectDelayMs ?? 16000;
  }

  connect(): void {
    this._intentionalClose = false;
    this._doConnect();
  }

  private _doConnect(): void {
    this._setStatus('connecting');
    const ws = new WebSocket(this._options.url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      this.ws = ws;
      this._reconnectDelay = this._options.reconnectDelayMs ?? 1000;
      this._setStatus('connected');
    };

    ws.onmessage = (e: MessageEvent) => {
      if (e.data instanceof ArrayBuffer) {
        this._options.onBinaryAudio(e.data);
      } else if (typeof e.data === 'string') {
        try {
          const parsed = JSON.parse(e.data) as Record<string, unknown>;
          this._options.onJsonEvent(parsed);
        } catch {
          console.warn('WebSocket: non-JSON text frame', e.data);
        }
      }
    };

    ws.onerror = () => {
      this._setStatus('error');
    };

    ws.onclose = () => {
      this.ws = null;
      if (this._intentionalClose) {
        this._setStatus('disconnected');
        return;
      }
      this._setStatus('disconnected');
      this._scheduleReconnect();
    };
  }

  private _scheduleReconnect(): void {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._intentionalClose) {
        this._doConnect();
        this._reconnectDelay = Math.min(
          this._reconnectDelay * 2,
          this._maxReconnectDelay,
        );
      }
    }, this._reconnectDelay);
  }

  sendBinary(buffer: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(buffer);
    }
  }

  sendJSON(payload: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  close(): void {
    this._intentionalClose = true;
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this.ws?.close(1000);
    this.ws = null;
    this._setStatus('disconnected');
  }

  get status(): ConnectionStatus {
    return this._status;
  }

  private _setStatus(s: ConnectionStatus): void {
    if (this._status !== s) {
      this._status = s;
      this._options.onStatusChange(s);
    }
  }
}
