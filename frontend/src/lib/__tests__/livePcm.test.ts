import { base64ToInt16, downsample, float32ToInt16, int16ToBase64, int16ToFloat32, monitorUrl } from "@/lib/livePcm";

test("pcm survives the trip through base64", () => {
  const pcm = Int16Array.from([0, 1, -1, 32767, -32768, 1234]);
  expect(Array.from(base64ToInt16(int16ToBase64(pcm)))).toEqual(Array.from(pcm));
});

test("an odd trailing byte is dropped, not turned into noise", () => {
  const bytes = String.fromCharCode(1, 0, 2); // 3 bytes
  expect(Array.from(base64ToInt16(btoa(bytes)))).toEqual([1]);
});

test("float and int16 conversion clamps and round trips", () => {
  expect(Array.from(float32ToInt16(Float32Array.from([0, 1, -1, 2, -2])))).toEqual([0, 32767, -32768, 32767, -32768]);
  const back = int16ToFloat32(Int16Array.from([0, 16384, -32768]));
  expect(back[1]).toBeCloseTo(0.5, 3);
  expect(back[2]).toBe(-1);
});

test("downsampling averages each window and never upsamples", () => {
  const input = Float32Array.from([1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1]); // 12 samples at 48 kHz
  const out = downsample(input, 48000, 8000);                                 // ratio 6
  expect(out.length).toBe(2);
  expect(out[0]).toBeCloseTo((4 * 1) / 6 + 0, 5);
  expect(downsample(input, 8000, 16000)).toHaveLength(12);
});

test("the listen-in address follows the API address and the token is escaped", () => {
  expect(monitorUrl("http://localhost:8000/", "a.b")).toBe("ws://localhost:8000/api/phone/exotel/monitor/a.b");
  expect(monitorUrl("https://api.example.com", "x/y")).toBe("wss://api.example.com/api/phone/exotel/monitor/x%2Fy");
});
