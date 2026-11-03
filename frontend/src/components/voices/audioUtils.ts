export interface AudioInfo {
  duration: number;
  peaks: number[];
}

function getContext(): AudioContext | null {
  const Ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

async function decode(blob: Blob): Promise<AudioBuffer | null> {
  const ctx = getContext();
  if (!ctx) return null;
  try {
    const buf = await blob.arrayBuffer();
    return await ctx.decodeAudioData(buf);
  } catch {
    return null;
  } finally {
    ctx.close().catch(() => {});
  }
}

/** Duration + 80 normalized peaks. Returns null when the browser cannot decode the format. */
export async function readAudioInfo(blob: Blob, bins = 80): Promise<AudioInfo | null> {
  const buffer = await decode(blob);
  if (!buffer) return null;
  const data = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / bins));
  const peaks: number[] = [];
  let max = 0;
  for (let i = 0; i < bins; i++) {
    let peak = 0;
    const start = i * step;
    const end = Math.min(data.length, start + step);
    for (let j = start; j < end; j += Math.max(1, Math.floor(step / 64))) {
      const v = Math.abs(data[j]);
      if (v > peak) peak = v;
    }
    peaks.push(peak);
    if (peak > max) max = peak;
  }
  return { duration: buffer.duration, peaks: max > 0 ? peaks.map((p) => p / max) : peaks };
}

/** Re-encode any decodable audio blob as 16-bit mono PCM WAV. */
export async function toWav(blob: Blob): Promise<Blob | null> {
  const buffer = await decode(blob);
  if (!buffer) return null;
  const samples = buffer.getChannelData(0);
  const rate = buffer.sampleRate;
  const out = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(out);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([out], { type: "audio/wav" });
}
