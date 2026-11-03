/** Helpers for listening in on a live phone call and speaking into it: 16-bit PCM over a WebSocket as base64. */

export function base64ToInt16(b64: string): Int16Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length - (bin.length % 2));
  for (let i = 0; i < bytes.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

export function int16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

export function int16ToFloat32(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  return out;
}

export function float32ToInt16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
  }
  return out;
}

/** Lower the sample rate by averaging each output sample's window of input samples (enough for a phone line). */
export function downsample(input: Float32Array, inRate: number, outRate: number): Float32Array {
  if (outRate >= inRate) return Float32Array.from(input);
  const ratio = inRate / outRate;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

/** wss:// (or ws://) address of a call's listen-in socket, from the API address and the monitor token. */
export function monitorUrl(apiUrl: string, token: string): string {
  const base = apiUrl.replace(/\/+$/, "").replace(/^http/i, "ws");
  return `${base}/api/phone/exotel/monitor/${encodeURIComponent(token)}`;
}

export type Voice = "caller" | "agent" | "human";

/** Plays call audio as it arrives. Each side keeps its own clock, so the caller and the agent can overlap like a real call. */
export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private cursor: Record<string, number> = {};
  private muted = false;

  setMuted(value: boolean): void {
    this.muted = value;
  }

  constructor(private make: () => AudioContext = () => new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()) {}

  /** Call from a click: browsers only start audio after a gesture. */
  async resume(): Promise<void> {
    if (!this.ctx) this.ctx = this.make();
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  play(who: Voice, pcm: Int16Array, rate: number): void {
    if (this.muted || !this.ctx || pcm.length === 0) return;
    const buffer = this.ctx.createBuffer(1, pcm.length, rate);
    buffer.copyToChannel(int16ToFloat32(pcm) as Float32Array<ArrayBuffer>, 0);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    const at = Math.max(now + 0.03, this.cursor[who] ?? 0);
    source.start(at);
    this.cursor[who] = at + buffer.duration;
  }

  close(): void {
    void this.ctx?.close();
    this.ctx = null;
    this.cursor = {};
  }
}

/** The microphone as 16-bit PCM chunks at the call's sample rate. */
export class MicStreamer {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private node: ScriptProcessorNode | null = null;

  async start(outRate: number, onChunk: (pcm: Int16Array) => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    // ScriptProcessor is deprecated but works everywhere and needs no separate worklet file.
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    const inRate = this.ctx.sampleRate;
    this.node.onaudioprocess = (e) => onChunk(float32ToInt16(downsample(e.inputBuffer.getChannelData(0), inRate, outRate)));
    source.connect(this.node);
    this.node.connect(this.ctx.destination);   // a processor only runs when connected; its own output is silent
  }

  stop(): void {
    if (this.node) this.node.onaudioprocess = null;
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    this.node = null;
    this.stream = null;
    this.ctx = null;
  }
}
