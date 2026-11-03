# Silero VAD assets (self-hosted, no CDN)

Served at `/vad/` and used by hands-free mode (`src/hooks/useHandsFree.ts`). Copied out of `node_modules` by
`npm run vad:assets`, which `npm install` runs too, so they always match the pinned package versions.

| File | From | Size | Licence | Committed |
| --- | --- | --- | --- | --- |
| `silero_vad_v5.onnx` | `@ricky0123/vad-web` (Silero VAD v5 model) | 2.3 MB | MIT (snakers4/silero-vad) | yes |
| `vad.worklet.bundle.min.js` | `@ricky0123/vad-web` | 2.5 KB | ISC | yes |
| `ort-wasm-simd-threaded.mjs` | `onnxruntime-web` | 24 KB | MIT | no (regenerated) |
| `ort-wasm-simd-threaded.wasm` | `onnxruntime-web` | 14 MB | MIT | no (regenerated) |

If these files are missing or fail to load, hands-free falls back to the built-in loudness detector (`src/lib/vad.ts`).
Upgrading: bump the two packages together (pinned exactly in `package.json`), run `npm run vad:assets`, and commit
the model and worklet if they changed.
