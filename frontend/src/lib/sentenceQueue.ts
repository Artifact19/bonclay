/**
 * The bookkeeping of a Studio render that streams sentence by sentence (POST /api/voices/generate/stream): which
 * sentence clips are ready to play and in what order, and the "Rendering sentence 2 of 3" line. No audio here: the
 * caller hands each ready clip to an AudioQueue (src/lib/audioQueue.ts), which plays them back to back.
 */
export interface StreamedSentence {
  index: number;
  text: string;
  audio_url: string;
  /** How many sentences the text was split into (known from the first event on). */
  total?: number;
}

export class SentenceQueue {
  private pending = new Map<number, StreamedSentence>();
  private next = 0; // the index the player needs next
  private seen = new Set<number>();
  private knownTotal: number | null = null;
  private finished = false;

  /**
   * Add a `sentence` event. Returns the clips that can start playing now, in order: usually just this one, more when
   * it filled a gap; none when an earlier sentence has not arrived yet or the event is a repeat.
   */
  push(s: StreamedSentence): StreamedSentence[] {
    if (!s || typeof s.audio_url !== "string" || !s.audio_url || !Number.isInteger(s.index) || s.index < 0) return [];
    if (typeof s.total === "number" && s.total > 0) this.knownTotal = s.total;
    if (this.seen.has(s.index)) return [];
    this.seen.add(s.index);
    this.pending.set(s.index, s);
    const ready: StreamedSentence[] = [];
    for (let c = this.pending.get(this.next); c; c = this.pending.get(this.next)) {
      ready.push(c);
      this.pending.delete(this.next);
      this.next += 1;
    }
    return ready;
  }

  /** The `done` event arrived: the whole clip is rendered. */
  finish(): void {
    this.finished = true;
  }

  /** Sentences received so far. */
  get received(): number {
    return this.seen.size;
  }

  get total(): number | null {
    return this.knownTotal;
  }

  get isFinished(): boolean {
    return this.finished;
  }

  /** The sentence being rendered now (1-based): the one after the last that arrived, never past the total. */
  get rendering(): number {
    const n = this.seen.size + 1;
    return this.knownTotal ? Math.min(n, this.knownTotal) : n;
  }

  /** What to show while the render runs: "Rendering sentence 2 of 3", "Finishing the clip" once every sentence is in. */
  label(): string {
    if (this.finished) return "";
    if (this.knownTotal && this.seen.size >= this.knownTotal) return "Finishing the clip";
    return this.knownTotal ? `Rendering sentence ${this.rendering} of ${this.knownTotal}` : `Rendering sentence ${this.rendering}`;
  }

  /** How far along the render is, 0 to 1 (0 until the first event says how many sentences there are). */
  fraction(): number {
    if (this.finished) return 1;
    return this.knownTotal ? Math.min(1, this.seen.size / this.knownTotal) : 0;
  }
}
