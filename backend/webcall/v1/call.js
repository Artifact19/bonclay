/*! BonClay web call page v1. A dependency-free script served by the phone-only public app (routes in
 * backend/routers/web_call.py). It talks only to its own origin: POST /call/api/{link}/session, POST .../turn,
 * GET .../clip/{turn}/{n}, POST .../end. Hands-free: the microphone stays open, a loudness detector (the same state
 * machine as frontend/src/lib/vad.ts `stepVad`) finds each utterance, MediaRecorder records it, and the reply is
 * played sentence by sentence. Speaking over the agent cuts it off (barge-in).
 */
(function () {
  'use strict';

  var TICK_MS = 40;
  var CAPTION_LINES = 60;
  var ONE_MINUTE_S = 60;
  var SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=';
  var AUDIO_CONSTRAINTS = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };

  /* ------------------------------------------------------------------ pure logic (unit tested) */

  /** Incremental server-sent events parser: push(text) returns the events completed by that text. */
  function SSEParser() { this.buffer = ''; }
  SSEParser.prototype.push = function (text) {
    this.buffer += text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    var out = [];
    var at;
    while ((at = this.buffer.indexOf('\n\n')) !== -1) {
      var block = this.buffer.slice(0, at);
      this.buffer = this.buffer.slice(at + 2);
      var name = 'message';
      var data = [];
      block.split('\n').forEach(function (line) {
        if (!line || line.charAt(0) === ':') return;
        var i = line.indexOf(':');
        var field = i === -1 ? line : line.slice(0, i);
        var value = i === -1 ? '' : line.slice(i + 1).replace(/^ /, '');
        if (field === 'event') name = value;
        else if (field === 'data') data.push(value);
      });
      if (!data.length) continue;
      var parsed;
      try { parsed = JSON.parse(data.join('\n')); } catch (e) { parsed = { raw: data.join('\n') }; }
      out.push({ event: name, data: parsed });
    }
    return out;
  };

  /**
   * Plays sentence clips one after another, in the order they were added, while later clips are already being fetched.
   * fetchClip(item) -> Promise<url>; play(url, item) -> Promise (resolves when the clip ends); onStart(item) fires as
   * a clip begins (or is skipped), onError(err, item) when a clip cannot be fetched or played. A failed clip never
   * blocks the ones after it. stop() drops everything queued or playing. `running` is true from the first clip until
   * the queue has drained (also while the next clip is still arriving).
   */
  function AudioQueue(opts) {
    this.opts = opts;
    this.items = [];
    this.running = false;
    this.generation = 0;
    this.waiters = [];
  }
  AudioQueue.prototype.add = function (item) {
    var entry = { item: item, ready: null };
    try { entry.ready = Promise.resolve(this.opts.fetchClip(item)); } catch (e) { entry.ready = Promise.reject(e); }
    entry.ready.catch(function () { /* handled when its turn comes */ });
    this.items.push(entry);
    if (!this.running) this._run();
  };
  AudioQueue.prototype._run = async function () {
    var gen = this.generation;
    this.running = true;
    while (this.items.length && gen === this.generation) {
      var entry = this.items.shift();
      var url = null;
      try { url = await entry.ready; } catch (e) { if (gen === this.generation) this.opts.onError(e, entry.item); }
      if (gen !== this.generation) break;
      if (this.opts.onStart) this.opts.onStart(entry.item);
      if (url) {
        try { await this.opts.play(url, entry.item); } catch (e) { if (gen === this.generation) this.opts.onError(e, entry.item); }
      }
    }
    if (gen === this.generation) {
      this.running = false;
      var w = this.waiters; this.waiters = [];
      w.forEach(function (fn) { fn(); });
    }
  };
  AudioQueue.prototype.idle = function () {
    var self = this;
    if (!this.running && !this.items.length) return Promise.resolve();
    return new Promise(function (resolve) { self.waiters.push(resolve); });
  };
  AudioQueue.prototype.stop = function () {
    this.generation++;
    this.items = [];
    this.running = false;
    var w = this.waiters; this.waiters = [];
    w.forEach(function (fn) { fn(); });
  };

  /* Voice activity: a pure state machine fed one loudness (RMS, 0..1) reading per tick. Same rules and numbers as
   * frontend/src/lib/vad.ts. Events: candidate (start recording early), abort (it was a blip), start (speech
   * confirmed), barge-in (speech confirmed over the agent's voice), end (send the recording), discard (too short). */
  var DEFAULT_VAD = {
    calibrationMs: 500, startMs: 150, endSilenceMs: 800, minUtteranceMs: 400, dipMs: 120, floorRatio: 3, margin: 0.012,
    floorAdaptMs: 4000, bargeMs: 300, bargeRatio: 2.5, bargeMin: 0.06, bargeThresholdFactor: 2, echoGraceMs: 500,
    tailGraceMs: 400, echoDecayPerSec: 0.5
  };

  function initVad() {
    return { phase: 'calibrating', t0: null, lastT: null, calSamples: [], floor: 0, aboveSince: null, lastAbove: 0,
      speechStart: 0, wasPlaying: false, playStart: 0, graceUntil: 0, echo: 0 };
  }
  function speechThreshold(floor, cfg) {
    cfg = cfg || DEFAULT_VAD;
    return Math.max(floor * cfg.floorRatio, floor + cfg.margin);
  }
  function bargeThreshold(s, cfg) {
    cfg = cfg || DEFAULT_VAD;
    return Math.max(speechThreshold(s.floor, cfg) * cfg.bargeThresholdFactor, cfg.bargeMin, s.echo * cfg.bargeRatio);
  }
  function quietMean(samples) {
    if (!samples.length) return 0;
    var sorted = samples.slice().sort(function (a, b) { return a - b; });
    var keep = sorted.slice(0, Math.max(1, Math.ceil(sorted.length * 0.7)));
    return keep.reduce(function (a, b) { return a + b; }, 0) / keep.length;
  }
  function stepVad(prev, input, cfg) {
    cfg = cfg || DEFAULT_VAD;
    var rms = input.rms, now = input.now, playing = input.playing;
    var s = Object.assign({}, prev);
    var events = [];
    if (s.t0 === null) s.t0 = now;
    var dt = s.lastT === null ? 0 : Math.max(0, now - s.lastT);
    s.lastT = now;
    if (playing && !prev.wasPlaying) { s.playStart = now; s.echo = 0; }
    if (!playing && prev.wasPlaying) s.graceUntil = now + cfg.tailGraceMs;
    s.wasPlaying = playing;

    if (s.phase === 'calibrating') {
      s.calSamples = prev.calSamples.concat([rms]);
      s.floor = quietMean(s.calSamples);
      if (now - s.t0 >= cfg.calibrationMs) { s.phase = 'idle'; s.calSamples = []; }
      return { state: s, events: events };
    }
    if (s.phase === 'speech') {
      if (rms > speechThreshold(s.floor, cfg)) {
        s.lastAbove = now;
      } else if (now - s.lastAbove >= cfg.endSilenceMs) {
        s.phase = 'idle';
        s.aboveSince = null;
        events.push(s.lastAbove - s.speechStart >= cfg.minUtteranceMs ? 'end' : 'discard');
      }
      return { state: s, events: events };
    }
    var inTail = !playing && now < s.graceUntil;
    var inEchoGrace = playing && now - s.playStart < cfg.echoGraceMs;
    var threshold = playing ? bargeThreshold(s, cfg) : speechThreshold(s.floor, cfg);
    if (playing) {
      if (inEchoGrace || rms < threshold) s.echo = Math.max(rms, s.echo * Math.pow(cfg.echoDecayPerSec, dt / 1000));
      else s.echo = s.echo * Math.pow(cfg.echoDecayPerSec, dt / 1000);
    }
    var blocked = inTail || inEchoGrace;
    var loud = !blocked && rms > threshold;
    if (loud) {
      s.lastAbove = now;
      if (s.aboveSince === null) { s.aboveSince = now; events.push('candidate'); }
      var hold = playing ? cfg.bargeMs : cfg.startMs;
      if (now - s.aboveSince >= hold) {
        s.phase = 'speech';
        s.speechStart = s.aboveSince;
        events.push(playing ? 'barge-in' : 'start');
      }
    } else {
      if (s.aboveSince !== null && (blocked || now - s.lastAbove > cfg.dipMs)) { s.aboveSince = null; events.push('abort'); }
      if (s.aboveSince === null && !playing && !inTail && rms <= threshold && dt > 0) {
        s.floor += (rms - s.floor) * (1 - Math.exp(-dt / cfg.floorAdaptMs));
      }
    }
    return { state: s, events: events };
  }
  function frameRms(frame) {
    if (!frame || !frame.length) return 0;
    var sum = 0;
    for (var i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    return Math.sqrt(sum / frame.length);
  }

  var MESSAGES = {
    CALL_LINK_INACTIVE: 'This call link is no longer active. Ask the person who shared it for a new one.',
    CALL_DAY_LIMIT: 'This line has reached its limit for today. Please try again tomorrow.',
    CALL_VISITOR_LIMIT: 'You have used your time on this line for today. Please try again tomorrow.',
    CALL_RATE_LIMITED: 'Lots of people are calling right now. Please try again in a few minutes.',
    CALL_SESSION_EXPIRED: 'Your call reached its time limit. Thanks for talking!',
    CALL_SESSION_INVALID: 'This call is no longer valid. Please start a new one.',
    CALL_FORBIDDEN: 'This page cannot start a call from here. Open the link you were given directly.',
    NO_SPEECH: 'I did not catch that. Please try again.',
    FILE_TOO_LARGE: 'That was too long. Please say it in shorter pieces.',
    ASR_UNAVAILABLE: 'The agent cannot hear right now. Please try again later.',
    network: 'Could not reach the agent. Check your connection and try again.',
    mic: 'The microphone is blocked. Allow microphone access for this site in your browser, then try again.',
    nodevice: 'No microphone was found. Plug one in and try again.',
    nomic: 'This browser cannot record audio. Try a recent Chrome, Edge, Firefox or Safari.',
    generic: 'Something went wrong. Please try again.'
  };
  function friendlyError(code) { return MESSAGES[code] || MESSAGES.generic; }

  function pickMime() {
    if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
    var list = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
    for (var i = 0; i < list.length; i++) { if (MediaRecorder.isTypeSupported(list[i])) return list[i]; }
    return '';
  }
  function extFor(mime) {
    mime = (mime || '').toLowerCase();
    if (mime.indexOf('ogg') !== -1) return 'ogg';
    if (mime.indexOf('mp4') !== -1 || mime.indexOf('aac') !== -1) return 'm4a';
    return 'webm';
  }
  function formatClock(seconds) {
    seconds = Math.max(0, Math.floor(seconds));
    var s = seconds % 60;
    return Math.floor(seconds / 60) + ':' + (s < 10 ? '0' : '') + s;
  }
  function linkIdFromPath(path) {
    var m = /^\/call\/(awc_[A-Za-z0-9_-]{24,64})\/?$/.exec(path || '');
    return m ? m[1] : null;
  }
  function coded(code, status) {
    var err = new Error(code);
    err.code = code;
    if (status) err.status = status;
    return err;
  }

  /* ------------------------------------------------------------------ the page */

  var END_TEXT = {
    user: 'Thanks for talking. You can start a new call any time.',
    time: 'Your call reached its time limit. Thanks for talking! You can start a new call.',
    inactive: MESSAGES.CALL_LINK_INACTIVE,
    limit: MESSAGES.CALL_DAY_LIMIT,
    error: MESSAGES.generic
  };
  var STATE_TEXT = {
    listening: 'Listening. Just speak.', hearing: 'Hearing you...', thinking: 'Thinking...',
    speaking: 'Speaking. Talk over me to interrupt.', muted: 'Your microphone is muted.', starting: 'Getting ready'
  };

  function CallPage(opts) {
    opts = opts || {};
    var doc = this.doc = opts.document || document;
    var win = this.win = opts.window || window;
    this.fetch = opts.fetch || function () { return win.fetch.apply(win, arguments); };
    this.mediaDevices = opts.mediaDevices || (win.navigator && win.navigator.mediaDevices);
    this.AudioCtor = opts.Audio || win.Audio;
    this.AudioContextCtor = opts.AudioContext || win.AudioContext || win.webkitAudioContext;
    this.RecorderCtor = opts.MediaRecorder || win.MediaRecorder;
    this.now = opts.now || function () { return (win.performance && win.performance.now) ? win.performance.now() : Date.now(); };
    this.reduceMotion = !!(win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches);
    var $ = function (id) { return doc.getElementById(id); };
    this.root = $('call-root');
    this.el = { start: $('start'), startError: $('start-error'), vStart: $('v-start'), vLive: $('v-live'), vEnd: $('v-end'),
      title: $('live-title'), clock: $('clock'), captions: $('captions'), state: $('state'), level: $('level'),
      liveError: $('live-error'), mute: $('mute'), end: $('end'), endTitle: $('end-title'), endText: $('end-text'),
      again: $('again') };
    this.linkId = this.root ? this.root.getAttribute('data-link') : '';
    this.api = '/call/api/' + this.linkId;
    this.phase = 'idle';           // idle | starting | live | ended
    this.state = 'starting';
    this.muted = false;
    this.session = null;
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
    this.buf = null;
    this.timer = null;
    this.clockTimer = null;
    this.vad = initVad();
    this.recorder = null;
    this.queue = null;
    this.audio = null;
    this.turnCtl = null;
    this.turnError = null;
    this.warnedOneMinute = false;
    this.ended = false;
    var self = this;
    this.el.start.addEventListener('click', function () { self.start(); });
    this.el.mute.addEventListener('click', function () { self.toggleMute(); });
    this.el.end.addEventListener('click', function () { self.end('user'); });
    this.el.again.addEventListener('click', function () { self.reset(); });
    win.addEventListener('pagehide', function () { self.hangUpQuietly(); });
  }

  /* ---- starting a call */
  CallPage.prototype.start = async function () {
    if (this.phase === 'starting' || this.phase === 'live') return;
    this.phase = 'starting';
    this.el.start.disabled = true;
    this.showStartError('');
    this.unlockAudio();   // inside the tap: phones only let a page play sound it has unlocked this way
    try {
      if (!this.mediaDevices || !this.mediaDevices.getUserMedia || !this.RecorderCtor) throw coded('nomic');
      try {
        this.stream = await this.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
      } catch (e) {
        throw coded(e && e.name === 'NotFoundError' ? 'nodevice' : 'mic');
      }
      await this.openSession();
    } catch (e) {
      this.releaseMic();
      this.phase = 'idle';
      this.el.start.disabled = false;
      this.showStartError(friendlyError(e && e.code ? e.code : 'generic'));
      return;
    }
    this.beginLive();
  };

  CallPage.prototype.openSession = async function () {
    var res;
    try {
      res = await this.fetch(this.api + '/session', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    } catch (e) { throw coded('network'); }
    var body = await res.json().catch(function () { return {}; });
    if (!res.ok) throw coded(body.code || 'generic', res.status);
    this.session = { token: body.token, maxS: body.max_session_s || body.expires_in || 600,
      name: body.agent && body.agent.name ? body.agent.name : '' };
  };

  CallPage.prototype.beginLive = function () {
    var self = this;
    this.phase = 'live';
    this.ended = false;
    this.muted = false;
    this.warnedOneMinute = false;
    this.vad = initVad();
    this.el.captions.textContent = '';
    this.el.mute.setAttribute('aria-pressed', 'false');
    this.el.mute.textContent = 'Mute microphone';
    this.showLiveError('');
    this.showView('live');
    this.ensureQueue();
    try {
      this.ctx = new this.AudioContextCtor();
      if (this.ctx.state === 'suspended' && this.ctx.resume) this.ctx.resume().catch(function () {});
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.ctx.createMediaStreamSource(this.stream).connect(this.analyser);   // not connected to the speakers: no feedback
      this.buf = new Float32Array(this.analyser.fftSize);
    } catch (e) {
      this.end('error', friendlyError('nomic'));
      return;
    }
    this.endAt = Date.now() + this.session.maxS * 1000;
    this.updateClock();
    this.clockTimer = setInterval(function () { self.updateClock(); }, 1000);
    this.timer = setInterval(function () { self.tick(); }, TICK_MS);
    this.setState('listening');
    this.addLine('hint', '', 'Say hello to begin. The agent answers out loud, and you can interrupt it by speaking.');
    if (this.el.title && this.el.title.focus) this.el.title.focus();
  };

  CallPage.prototype.updateClock = function () {
    var left = Math.ceil((this.endAt - Date.now()) / 1000);
    this.el.clock.textContent = formatClock(left) + ' left';
    if (left <= ONE_MINUTE_S && !this.warnedOneMinute && left > 0) {
      this.warnedOneMinute = true;
      this.addLine('note', '', 'One minute left.');
    }
    if (left <= 0) this.end('time');
  };

  /* ---- listening */
  CallPage.prototype.isPlaying = function () { return !!(this.queue && this.queue.running); };

  CallPage.prototype.tick = function () {
    if (this.phase !== 'live' || !this.analyser) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    var rms = frameRms(this.buf);
    this.setLevel(Math.min(1, rms * 6));
    if (this.muted) return;
    var step = stepVad(this.vad, { rms: rms, now: this.now(), playing: this.isPlaying() }, DEFAULT_VAD);
    this.vad = step.state;
    for (var i = 0; i < step.events.length; i++) this.onVad(step.events[i]);
  };

  CallPage.prototype.onVad = function (ev) {
    if (ev === 'candidate') this.startRecorder();
    else if (ev === 'abort' || ev === 'discard') { this.finishRecorder(false); this.settle(); }
    else if (ev === 'start') { this.interrupt(); this.setState('hearing'); }
    else if (ev === 'barge-in') { this.interrupt(); this.setState('hearing'); }
    else if (ev === 'end') { this.finishRecorder(true); }
  };

  /** Back to the resting state after something that did not turn into a turn. */
  CallPage.prototype.settle = function () {
    if (this.state === 'hearing') this.setState(this.muted ? 'muted' : 'listening');
  };

  CallPage.prototype.startRecorder = function () {
    if (!this.stream || this.recorder) return;
    var self = this;
    try {
      var mime = pickMime();
      var rec = mime ? new this.RecorderCtor(this.stream, { mimeType: mime }) : new this.RecorderCtor(this.stream);
      var handle = { rec: rec, send: false, chunks: [], mime: rec.mimeType || mime || 'audio/webm' };
      rec.ondataavailable = function (e) { if (e.data && e.data.size > 0) handle.chunks.push(e.data); };
      rec.onstop = function () {
        if (!handle.send || self.phase !== 'live') return;
        var blob = new Blob(handle.chunks, { type: handle.mime });
        if (blob.size > 0) self.sendTurn(blob, extFor(handle.mime));
      };
      rec.start();
      this.recorder = handle;
    } catch (e) {
      this.recorder = null;
    }
  };

  CallPage.prototype.finishRecorder = function (send) {
    var handle = this.recorder;
    this.recorder = null;
    if (!handle) return;
    handle.send = send;
    try { if (handle.rec.state !== 'inactive') handle.rec.stop(); } catch (e) { /* already stopped */ }
  };

  /* ---- a turn */
  CallPage.prototype.interrupt = function () {
    if (this.turnCtl) { this.turnCtl.abort(); this.turnCtl = null; }
    if (this.queue) this.queue.stop();
    if (this.audio) { try { this.audio.pause(); } catch (e) { /* ignore */ } this.audio.onended = this.audio.onerror = null; }
  };

  CallPage.prototype.sendTurn = async function (blob, ext) {
    this.interrupt();
    var ctl = new AbortController();
    this.turnCtl = ctl;
    this.turnError = null;
    this.showLiveError('');
    this.setState('thinking');
    this.ensureQueue();
    try {
      var fd = new FormData();
      fd.append('audio', blob, 'recording.' + ext);
      var res = await this.fetch(this.api + '/turn', { method: 'POST', body: fd, signal: ctl.signal,
        headers: { Authorization: 'Bearer ' + this.session.token } });
      if (!res.ok) {
        var body = await res.json().catch(function () { return {}; });
        throw coded(body.code || 'generic', res.status);
      }
      await this.readStream(res, ctl);
      if (ctl.signal.aborted) return;
      if (this.turnError) throw coded(this.turnError);
      await this.queue.idle();
      if (this.turnCtl === ctl) {
        this.turnCtl = null;
        if (this.phase === 'live' && this.state !== 'hearing') this.setState(this.muted ? 'muted' : 'listening');
      }
    } catch (e) {
      if (ctl.signal.aborted) return;
      if (this.turnCtl === ctl) this.turnCtl = null;
      this.queue.stop();
      this.onTurnError(e && e.code ? e.code : 'network');
    }
  };

  CallPage.prototype.onTurnError = function (code) {
    if (this.phase !== 'live') return;
    if (code === 'CALL_SESSION_EXPIRED') return this.end('time');
    if (code === 'CALL_LINK_INACTIVE') return this.end('inactive', MESSAGES.CALL_LINK_INACTIVE);
    if (code === 'CALL_DAY_LIMIT' || code === 'CALL_VISITOR_LIMIT') return this.end('limit', friendlyError(code));
    if (code === 'CALL_SESSION_INVALID' || code === 'CALL_FORBIDDEN') return this.end('error', friendlyError(code));
    this.showLiveError(friendlyError(code));
    this.setState(this.muted ? 'muted' : 'listening');
  };

  CallPage.prototype.readStream = async function (res, ctl) {
    var self = this;
    var parser = new SSEParser();
    var handle = function (ev) { self.onEvent(ev); };
    if (!res.body || !res.body.getReader) { parser.push(await res.text()).forEach(handle); return; }
    var reader = res.body.getReader();
    var dec = new TextDecoder();
    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      parser.push(dec.decode(r.value, { stream: true })).forEach(handle);
      if (ctl.signal.aborted) { try { reader.cancel(); } catch (e) { /* ignore */ } return; }
    }
  };

  CallPage.prototype.onEvent = function (ev) {
    var d = ev.data || {};
    if (ev.event === 'transcript') {
      if (d.user_text) this.addLine('you', 'You', d.user_text);
    } else if (ev.event === 'sentence') {
      if (d.filler) this.setState('thinking');
      this.queue.add({ index: d.index, text: d.text || '', url: d.audio_url || '', filler: !!d.filler });
    } else if (ev.event === 'error') {
      this.turnError = d.code || 'generic';
    }
  };

  /* ---- playing the reply */
  CallPage.prototype.unlockAudio = function () {
    try {
      if (!this.audio) this.audio = new this.AudioCtor();
      this.audio.src = SILENT_WAV;
      var p = this.audio.play();
      if (p && p.catch) p.catch(function () {});
    } catch (e) { /* the first clip will try again */ }
  };

  CallPage.prototype.ensureQueue = function () {
    if (this.queue) return;
    var self = this;
    this.queue = new AudioQueue({
      fetchClip: function (item) { return self.fetchClip(item); },
      play: function (url) { return self.play(url); },
      onStart: function (item) {
        if (!item.filler && item.text) self.addLine('agent', 'Agent', item.text);
        if (self.phase === 'live') self.setState('speaking');
      },
      onError: function () { /* the caption is still shown; a silent clip should not stop the reply */ }
    });
  };

  CallPage.prototype.fetchClip = async function (item) {
    var url = item.url;
    if (!url) return null;
    if (url.indexOf('data:') === 0) return url;
    if (url.indexOf('/call/api/' + this.linkId + '/clip/') !== 0) return null;   // only this link's own clips
    var res = await this.fetch(url, { headers: { Authorization: 'Bearer ' + this.session.token } });
    if (!res.ok) throw coded('clip');
    return URL.createObjectURL(await res.blob());
  };

  CallPage.prototype.play = function (url) {
    var a = this.audio || (this.audio = new this.AudioCtor());
    return new Promise(function (resolve, reject) {
      var done = function (fn, v) {
        a.onended = a.onerror = null;
        if (url.indexOf('blob:') === 0) URL.revokeObjectURL(url);
        fn(v);
      };
      a.onended = function () { done(resolve); };
      a.onerror = function () { done(reject, new Error('audio')); };
      a.src = url;
      var p = a.play();
      if (p && p.catch) p.catch(function (e) { done(reject, e); });
    });
  };

  /* ---- controls */
  CallPage.prototype.toggleMute = function () {
    if (this.phase !== 'live') return;
    this.muted = !this.muted;
    if (this.stream) this.stream.getAudioTracks().forEach(function (t) { t.enabled = !this.muted; }, this);
    this.el.mute.setAttribute('aria-pressed', this.muted ? 'true' : 'false');
    this.el.mute.textContent = this.muted ? 'Unmute microphone' : 'Mute microphone';
    this.finishRecorder(false);
    this.vad = initVad();
    this.setLevel(0);
    if (this.state !== 'thinking' && this.state !== 'speaking') this.setState(this.muted ? 'muted' : 'listening');
  };

  CallPage.prototype.end = function (reason, message) {
    if (this.phase !== 'live' && this.phase !== 'starting') return;
    this.phase = 'ended';
    this.interrupt();
    this.finishRecorder(false);
    this.stopTimers();
    this.hangUpQuietly();
    this.el.endText.textContent = message || END_TEXT[reason] || END_TEXT.error;
    this.showView('end');
    if (this.el.endTitle && this.el.endTitle.focus) this.el.endTitle.focus();
  };

  /** Release the microphone and tell the server the call is over (best effort; the minutes are settled there). */
  CallPage.prototype.hangUpQuietly = function () {
    var token = this.session && this.session.token;
    this.stopTimers();
    this.releaseMic();
    if (token && !this.ended) {
      this.ended = true;
      try {
        var p = this.fetch(this.api + '/end', { method: 'POST', keepalive: true, headers: { Authorization: 'Bearer ' + token } });
        if (p && p.catch) p.catch(function () {});
      } catch (e) { /* ignore */ }
    }
  };

  CallPage.prototype.reset = function () {
    this.phase = 'idle';
    this.session = null;
    this.ended = false;
    this.el.start.disabled = false;
    this.showStartError('');
    this.showView('start');
    this.el.start.focus();
  };

  CallPage.prototype.stopTimers = function () {
    if (this.timer) clearInterval(this.timer);
    if (this.clockTimer) clearInterval(this.clockTimer);
    this.timer = this.clockTimer = null;
  };

  CallPage.prototype.releaseMic = function () {
    var stream = this.stream;
    this.stream = null;
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    var ctx = this.ctx;
    this.ctx = null;
    this.analyser = null;
    if (ctx && ctx.state !== 'closed' && ctx.close) ctx.close().catch(function () {});
  };

  /* ---- the page */
  CallPage.prototype.showView = function (name) {
    this.el.vStart.hidden = name !== 'start';
    this.el.vLive.hidden = name !== 'live';
    this.el.vEnd.hidden = name !== 'end';
  };
  CallPage.prototype.showStartError = function (text) { this.el.startError.hidden = !text; this.el.startError.textContent = text || ''; };
  CallPage.prototype.showLiveError = function (text) { this.el.liveError.hidden = !text; this.el.liveError.textContent = text || ''; };

  CallPage.prototype.setState = function (state) {
    this.state = state;
    this.el.state.setAttribute('data-s', state);
    this.el.state.textContent = STATE_TEXT[state] || '';
  };

  CallPage.prototype.setLevel = function (level) {
    if (this.reduceMotion || !this.el.level) return;
    this.el.level.style.setProperty('--lvl', level.toFixed(2));
  };

  /** One completed line in the captions log (one element per line: a screen reader announces each once). */
  CallPage.prototype.addLine = function (kind, who, text) {
    var caps = this.el.captions;
    var p = this.doc.createElement('p');
    p.className = 'line ' + kind;
    if (who) {
      var w = this.doc.createElement('span');
      w.className = 'who';
      w.textContent = who + ':';
      p.appendChild(w);
      p.appendChild(this.doc.createTextNode(' '));
    }
    p.appendChild(this.doc.createTextNode(text));
    if (kind !== 'hint') {
      var hints = caps.querySelectorAll('.hint');
      for (var i = 0; i < hints.length; i++) caps.removeChild(hints[i]);
    }
    caps.appendChild(p);
    while (caps.children.length > CAPTION_LINES) caps.removeChild(caps.firstChild);
    caps.scrollTop = caps.scrollHeight;
    return p;
  };

  /* ------------------------------------------------------------------ boot */

  if (typeof document !== 'undefined' && document.getElementById && document.getElementById('call-root') &&
      !(typeof module !== 'undefined' && module.exports)) {
    window.bonclayCall = new CallPage();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SSEParser: SSEParser, AudioQueue: AudioQueue, DEFAULT_VAD: DEFAULT_VAD, initVad: initVad,
      stepVad: stepVad, speechThreshold: speechThreshold, bargeThreshold: bargeThreshold, frameRms: frameRms,
      friendlyError: friendlyError, extFor: extFor, formatClock: formatClock, linkIdFromPath: linkIdFromPath,
      MESSAGES: MESSAGES, CallPage: CallPage };
  }
})();
