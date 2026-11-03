import { useVoiceStore } from "@/stores";
import type { Voice } from "@/types/api";

const v = (id: string): Voice => ({ id, name: id, created_at: "2026-01-01" });

describe("useVoiceStore", () => {
  beforeEach(() => useVoiceStore.setState({ voices: [], selectedVoiceId: null }));

  it("adds voices to the front without duplicates", () => {
    const s = useVoiceStore.getState();
    s.addVoice(v("a"));
    s.addVoice(v("b"));
    s.addVoice(v("a"));
    expect(useVoiceStore.getState().voices.map((x) => x.id)).toEqual(["a", "b"]);
  });

  it("clears selection when selected voice is removed", () => {
    const s = useVoiceStore.getState();
    s.setVoices([v("a"), v("b")]);
    s.selectVoice("a");
    s.removeVoice("a");
    expect(useVoiceStore.getState().selectedVoiceId).toBeNull();
    expect(useVoiceStore.getState().voices).toHaveLength(1);
  });

  it("drops stale selection on setVoices", () => {
    const s = useVoiceStore.getState();
    s.selectVoice("gone");
    s.setVoices([v("a")]);
    expect(useVoiceStore.getState().selectedVoiceId).toBeNull();
  });
});
