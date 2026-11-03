// One sample plays at a time across the landing page: a player calls claimAudio() just before play().
let current: HTMLAudioElement | null = null;

export function claimAudio(el: HTMLAudioElement) {
  if (current && current !== el && !current.paused) current.pause();
  current = el;
}
