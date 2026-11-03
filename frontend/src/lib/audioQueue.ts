/**
 * Plays audio clips back to back, in the order they were added. A streamed reply arrives sentence by sentence: the
 * first clip starts as soon as it is added, later ones are preloaded while the earlier ones play.
 */
export interface QueueAudio {
  src: string;
  preload: string;
  play(): Promise<void> | void;
  pause(): void;
  addEventListener(type: string, fn: () => void, opts?: { once?: boolean }): void;
}

export class AudioQueue {
  private clips: QueueAudio[] = [];
  private played = 0; // clips started so far
  private playing = false;
  private stopped = false;
  private closed = false;
  private waiters: (() => void)[] = [];
  /** Clips that could not be played (autoplay blocked, a failed download). */
  failures = 0;

  constructor(private make: (url: string) => QueueAudio = (url) => new Audio(url) as unknown as QueueAudio) {}

  /** Add a clip; it plays right away when nothing is playing. */
  add(url: string): void {
    if (this.stopped) return;
    const clip = this.make(url);
    clip.preload = "auto";
    this.clips.push(clip);
    if (!this.playing) this.next();
  }

  private next(): void {
    if (this.stopped) return;
    if (this.played >= this.clips.length) {
      this.playing = false;
      this.settle();
      return;
    }
    const clip = this.clips[this.played++];
    this.playing = true;
    let advanced = false;
    const advance = (failed: boolean) => {
      if (advanced) return;
      advanced = true;
      if (failed) this.failures += 1;
      this.next();
    };
    clip.addEventListener("ended", () => advance(false), { once: true });
    clip.addEventListener("error", () => advance(true), { once: true });
    try {
      const p = clip.play();
      if (p && typeof (p as Promise<void>).catch === "function") (p as Promise<void>).catch(() => advance(true));
    } catch {
      advance(true);
    }
  }

  /** No more clips will be added: `finished()` resolves once the last one has played. */
  close(): void {
    this.closed = true;
    this.settle();
  }

  private settle(): void {
    if (this.closed && !this.playing && this.waiters.length) {
      const w = this.waiters;
      this.waiters = [];
      w.forEach((f) => f());
    }
  }

  /** Resolves when the queue was closed and everything has been played (or it was stopped). */
  finished(): Promise<void> {
    return new Promise((resolve) => {
      this.waiters.push(resolve);
      this.settle();
    });
  }

  /** Whether a clip is playing or waiting to play. */
  get busy(): boolean {
    return !this.stopped && (this.playing || this.played < this.clips.length);
  }

  /** Stop playback and drop everything queued (a new recording starts, or the page goes away). */
  stop(): void {
    this.stopped = true;
    this.playing = false;
    for (const c of this.clips) {
      try {
        c.pause();
      } catch {
        /* already torn down */
      }
    }
    this.clips = [];
    this.settle();
  }
}
