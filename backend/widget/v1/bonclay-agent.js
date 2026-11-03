/*! BonClay website widget v1: <bonclay-agent agent="PUBLIC_KEY"></bonclay-agent>
 *
 * A dependency-free Web Component (shadow DOM). It talks only to the BonClay host this script was loaded from
 * (override with the `api` attribute): POST /api/widget/session, POST /api/widget/turn/stream, GET /api/widget/clip/...
 *
 * Attributes: agent (required, the public widget key), api (optional API origin), label (launcher text),
 *             position ("right" | "left"), color (accent, #rrggbb).
 */
(function () {
  'use strict';

  var SCRIPT_SRC = (typeof document !== 'undefined' && document.currentScript && document.currentScript.src) || '';
  var DEFAULT_API = '';
  try { DEFAULT_API = SCRIPT_SRC ? new URL(SCRIPT_SRC).origin : ''; } catch (e) { DEFAULT_API = ''; }

  var TAP_MS = 350;            // a press shorter than this latches the recording (tap to start, tap to send)
  var MIN_RECORD_MS = 400;     // shorter recordings are not sent
  var MAX_RECORD_MS = 30000;   // a recording stops by itself here
  var RENEW_BEFORE_MS = 60000; // renew the session token this long before it expires
  var SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=';

  /* ------------------------------------------------------------------ pure logic (unit tested) */

  /** Incremental server-sent events parser: push(text) returns the events completed by that text. */
  function SSEParser() {
    this.buffer = '';
  }
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
   * Plays sentence clips one after another, in the order they were added, while later clips are already being
   * fetched. fetchClip(item) -> Promise<url>; play(url, item) -> Promise (resolves when the clip ends);
   * onStart(item) fires as a clip begins (or is skipped), onError(err, item) when a clip cannot be fetched or played.
   * A failed clip never blocks the ones after it. stop() drops everything that is queued or playing.
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

  var MESSAGES = {
    WIDGET_RATE_LIMITED: 'Lots of people are talking to this assistant right now. Please try again in a few minutes.',
    WIDGET_FORBIDDEN: 'This assistant is not available on this site.',
    NO_SPEECH: 'I did not catch that. Please hold the button and speak.',
    FILE_TOO_LARGE: 'That recording was too long. Please keep it short.',
    network: 'Could not reach the assistant. Check your connection and try again.',
    mic: 'The microphone is blocked. Allow microphone access for this site and try again.',
    nomic: 'This browser cannot record audio.',
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
  function safeColor(value) { return /^#[0-9a-fA-F]{6}$/.test(value || '') ? value : '#3b5bdb'; }
  function safeApi(value) {
    try {
      var u = new URL(value);
      return u.protocol === 'http:' || u.protocol === 'https:' ? u.origin : '';
    } catch (e) { return ''; }
  }

  /* ------------------------------------------------------------------ component */

  var CSS = [
    ':host{all:initial;position:fixed;bottom:20px;right:20px;z-index:2147483000;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;',
    '--a:#3b5bdb;--bg:#fff;--fg:#1c2230;--mut:#5b6478;--line:#dfe3ec;--soft:#f3f5fa}',
    ':host([position="left"]){right:auto;left:20px}',
    '@media (prefers-color-scheme:dark){:host{--bg:#1b2030;--fg:#eef1f8;--mut:#a3adc2;--line:#343b52;--soft:#252b3f}}',
    '*{box-sizing:border-box}',
    'button{font:inherit;cursor:pointer;border:0;color:inherit}',
    'button:focus-visible{outline:3px solid var(--a);outline-offset:2px}',
    '.launcher{display:flex;align-items:center;gap:8px;background:var(--a);color:#fff;border-radius:999px;padding:14px;box-shadow:0 6px 20px rgba(0,0,0,.25)}',
    '.launcher.text{padding:14px 18px 14px 14px}',
    '.launcher svg{width:24px;height:24px;flex:none}',
    '.panel{position:absolute;bottom:68px;right:0;width:min(340px,calc(100vw - 32px));background:var(--bg);color:var(--fg);border:1px solid var(--line);',
    'border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.3);overflow:hidden}',
    ':host([position="left"]) .panel{right:auto;left:0}',
    '.panel[hidden],.launcher[hidden],[hidden]{display:none!important}',
    '.head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 14px;border-bottom:1px solid var(--line)}',
    '.title{font-weight:600;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
    '.x{background:transparent;font-size:22px;line-height:1;padding:2px 8px;border-radius:8px;color:var(--mut)}',
    '.body{padding:14px}',
    '.notice{font-size:14px;line-height:1.5;margin:0 0 14px}',
    '.primary{background:var(--a);color:#fff;border-radius:10px;padding:10px 16px;font-weight:600;width:100%}',
    '.caps{min-height:84px;max-height:200px;overflow:auto;font-size:14px;line-height:1.45;display:flex;flex-direction:column;gap:8px;margin-bottom:12px}',
    '.caps p{margin:0;padding:8px 10px;border-radius:10px;background:var(--soft);max-width:92%}',
    '.caps p.you{align-self:flex-end;background:var(--a);color:#fff}',
    '.caps p.hint{background:transparent;color:var(--mut);padding:0}',
    '.status{font-size:13px;color:var(--mut);text-align:center;min-height:18px;margin:0 0 8px}',
    '.err{font-size:13px;color:#b42318;background:#fef3f2;border-radius:8px;padding:8px 10px;margin:0 0 10px}',
    '@media (prefers-color-scheme:dark){.err{color:#fda29b;background:#3a1b1b}}',
    '.talk{display:flex;margin:0 auto;width:76px;height:76px;border-radius:50%;background:var(--a);color:#fff;touch-action:none;user-select:none;-webkit-user-select:none;',
    '-webkit-touch-callout:none;align-items:center;justify-content:center}',
    '.talk svg{width:32px;height:32px;pointer-events:none}',
    '.talk.rec{background:#d92d20;animation:pulse 1.2s ease-in-out infinite}',
    '.talk[disabled]{opacity:.55;cursor:default}',
    '@keyframes pulse{50%{box-shadow:0 0 0 12px rgba(217,45,32,.18)}}',
    '@media (prefers-reduced-motion:reduce){.talk.rec{animation:none}}',
    '.how{font-size:12px;color:var(--mut);text-align:center;margin:8px 0 0}',
    '.foot{display:flex;justify-content:space-between;gap:8px;padding:8px 14px;border-top:1px solid var(--line);font-size:11px;color:var(--mut)}',
    '.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}'
  ].join('');

  var MIC = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v4"/></svg>';

  var Base = (typeof HTMLElement !== 'undefined') ? HTMLElement : function () {};

  class BonClayAgent extends Base {
    constructor() {
      super();
      this._s = { open: false, ack: false, session: null, sessionPromise: null, lastToken: null, turnError: null, state: 'idle', rec: null, abort: null, capLine: null };
      this._audio = null;
      this._queue = null;
    }

    static get observedAttributes() { return ['agent', 'api', 'label', 'position', 'color']; }

    connectedCallback() {
      if (this._root) return;
      this._root = this.attachShadow({ mode: 'open' });
      var color = safeColor(this.getAttribute('color'));
      this._root.innerHTML =
        '<style>' + CSS + '</style>' +
        '<button class="launcher" type="button" aria-haspopup="dialog" aria-expanded="false">' + MIC + '<span class="lt"></span></button>' +
        '<section class="panel" role="dialog" aria-modal="false" hidden>' +
          '<div class="head"><span class="title"></span><button class="x" type="button" aria-label="Close assistant">×</button></div>' +
          '<div class="body">' +
            '<div class="v-load" hidden><p class="status" role="status">Connecting...</p></div>' +
            '<div class="v-notice" hidden><p class="notice"></p><button class="primary start" type="button">Start</button></div>' +
            '<div class="v-talk" hidden>' +
              '<div class="caps" role="log" aria-live="polite" aria-label="Conversation captions"></div>' +
              '<p class="err" role="alert" hidden></p>' +
              '<p class="status" role="status" aria-live="polite"></p>' +
              '<button class="talk" type="button"></button>' +
              '<p class="how">Hold to talk, or tap once to start and tap again to send.</p>' +
            '</div>' +
            '<div class="v-fail" hidden><p class="err" role="alert"></p><button class="primary retry" type="button">Try again</button></div>' +
          '</div>' +
          '<div class="foot"><span class="ai">AI-generated voice</span><span>Powered by BonClay</span></div>' +
        '</section>';
      this._root.host.style.setProperty('--a', color);
      var q = (sel) => this._root.querySelector(sel);
      this._el = { launcher: q('.launcher'), lt: q('.lt'), panel: q('.panel'), title: q('.title'), close: q('.x'),
        load: q('.v-load'), notice: q('.v-notice'), noticeText: q('.notice'), start: q('.start'),
        talkView: q('.v-talk'), caps: q('.caps'), err: q('.v-talk .err'), status: q('.v-talk .status'), talk: q('.talk'),
        fail: q('.v-fail'), failText: q('.v-fail .err'), retry: q('.retry') };
      this._el.talk.setAttribute('aria-label', 'Hold to talk');
      this._el.talk.innerHTML = MIC;
      this._applyLabel();

      this._el.launcher.addEventListener('click', () => this._toggle());
      this._el.close.addEventListener('click', () => this._close(true));
      this._el.retry.addEventListener('click', () => this._begin());
      this._el.start.addEventListener('click', () => this._accept());
      this._root.addEventListener('keydown', (e) => { if (e.key === 'Escape' && this._s.open) { this._close(true); } });
      this._wireTalk();
    }

    disconnectedCallback() { this._close(false); }

    attributeChangedCallback() { if (this._root) { this._applyLabel(); } }

    /* ---- configuration */
    get _api() { return safeApi(this.getAttribute('api') || '') || DEFAULT_API; }
    get _key() { return (this.getAttribute('agent') || '').trim(); }
    _applyLabel() {
      var label = (this.getAttribute('label') || '').trim();
      var name = this._s.session && this._s.session.agent.name;
      this._el.lt.textContent = label;
      this._el.launcher.classList.toggle('text', !!label);
      this._el.launcher.setAttribute('aria-label', label || ('Talk to ' + (name || 'our AI assistant')));
    }

    /* ---- open / close */
    _toggle() { if (this._s.open) this._close(true); else this._open(); }

    _open() {
      this._s.open = true;
      this._el.panel.hidden = false;
      this._el.launcher.setAttribute('aria-expanded', 'true');
      this._begin();
    }

    _close(refocus) {
      this._stopEverything();
      this._s.open = false;
      if (!this._el) return;
      this._el.panel.hidden = true;
      this._el.launcher.setAttribute('aria-expanded', 'false');
      if (refocus) this._el.launcher.focus();
    }

    _views(name) {
      ['load', 'notice', 'talkView', 'fail'].forEach((v) => { this._el[v].hidden = (v !== name); });
    }

    async _begin() {
      if (!this._key) { return this._fail(friendlyError('WIDGET_FORBIDDEN')); }
      this._views('load');
      try {
        await this._ensureSession();
      } catch (e) {
        return this._fail(friendlyError(e && e.code ? e.code : 'network'));
      }
      this._el.title.textContent = this._s.session.agent.name;
      this._applyLabel();
      if (this._readAck()) { this._showTalk(); return; }
      var biz = this._s.session.agent.business_name || this._s.session.agent.name;
      this._el.noticeText.textContent = "You're talking to an AI assistant. Audio is processed to answer you; conversations may be stored by " + biz + '.';
      this._views('notice');
      this._el.start.focus();
    }

    _accept() {
      this._writeAck();
      this._showTalk();
      // Ask for the microphone now so a refusal shows up before the first question.
      this._probeMic();
    }

    _showTalk() {
      this._views('talkView');
      this._s.ack = true;
      this._setState('idle');
      if (!this._el.caps.children.length) {
        var g = this._s.session && this._s.session.agent.greeting;
        this._addCaption('hint', g ? g : 'Ask a question out loud.');
      }
      this._el.talk.focus();
    }

    _fail(message) {
      this._views('fail');
      this._el.failText.textContent = message;
      this._el.retry.focus();
    }

    _ackKey() { return 'bonclay-agent:ack:' + this._key; }
    _readAck() { try { return window.localStorage.getItem(this._ackKey()) === '1'; } catch (e) { return false; } }
    _writeAck() { try { window.localStorage.setItem(this._ackKey(), '1'); } catch (e) { /* private mode: asked again next time */ } }

    /* ---- session */
    _ensureSession() {
      var s = this._s;
      if (s.session && s.session.expiresAt - Date.now() > RENEW_BEFORE_MS) return Promise.resolve(s.session);
      if (s.sessionPromise) return s.sessionPromise;
      s.sessionPromise = this._newSession().then(
        (sess) => { s.session = sess; s.sessionPromise = null; return sess; },
        (err) => { s.sessionPromise = null; throw err; });
      return s.sessionPromise;
    }

    async _newSession() {
      var res;
      try {
        res = await fetch(this._api + '/api/widget/session', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: this._key, previous: (this._s.session && this._s.session.token) || this._s.lastToken || undefined })
        });
      } catch (e) { var n = new Error('network'); n.code = 'network'; throw n; }
      var body = await res.json().catch(function () { return {}; });
      if (!res.ok) { var err = new Error('session'); err.code = body.code || 'generic'; throw err; }
      return { token: body.token, expiresAt: Date.now() + (body.expires_in || 600) * 1000, agent: body.agent || { name: 'Assistant', business_name: '', greeting: '' } };
    }

    /* ---- recording controls */
    _wireTalk() {
      var b = this._el.talk;
      var down = (e) => { e.preventDefault(); this._press(); };
      var up = (e) => { e.preventDefault(); this._release(); };
      b.addEventListener('pointerdown', (e) => { if (e.button !== undefined && e.button > 0) return; try { b.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } down(e); });
      b.addEventListener('pointerup', up);
      b.addEventListener('pointercancel', up);
      b.addEventListener('contextmenu', (e) => e.preventDefault());
      b.addEventListener('keydown', (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) down(e); else if (e.key === ' ' || e.key === 'Enter') e.preventDefault(); });
      b.addEventListener('keyup', (e) => { if (e.key === ' ' || e.key === 'Enter') up(e); });
    }

    _unlockAudio() {
      if (!this._audio) { this._audio = new Audio(); }
      try { this._audio.src = SILENT_WAV; var p = this._audio.play(); if (p && p.catch) p.catch(function () {}); } catch (e) { /* ignore */ }
    }

    _press() {
      var s = this._s;
      if (s.state === 'thinking') return;
      if (s.rec && s.rec.latched) { this._finishRecording(); return; }
      if (s.rec) return;
      this._unlockAudio();
      this._stopPlayback();
      if (s.abort) { s.abort.abort(); s.abort = null; }
      this._showError('');
      s.rec = { startedAt: Date.now(), stopRequested: false, latched: false, chunks: [], recorder: null, stream: null, timer: null };
      this._setState('recording');
      this._startRecorder(s.rec);
    }

    _release() {
      var rec = this._s.rec;
      if (!rec || rec.latched || rec.released) return;
      rec.released = true;
      if (Date.now() - rec.startedAt < TAP_MS) {   // a tap: keep listening until the next tap
        rec.latched = true;
        this._setState('recording');
        return;
      }
      this._finishRecording();
    }

    _finishRecording() {
      var rec = this._s.rec;
      if (!rec) return;
      rec.stopRequested = true;
      if (rec.recorder && rec.recorder.state !== 'inactive') { try { rec.recorder.stop(); } catch (e) { this._abortRecording(); } }
      else if (!rec.recorder) { /* the microphone is still opening: _startRecorder stops it as soon as it is ready */ }
    }

    async _probeMic() {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
        var st = await navigator.mediaDevices.getUserMedia({ audio: true });
        st.getTracks().forEach(function (t) { t.stop(); });
      } catch (e) { this._showError(friendlyError('mic')); }
    }

    async _startRecorder(rec) {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
          throw Object.assign(new Error('nomic'), { code: 'nomic' });
        }
        rec.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (this._s.rec !== rec) { rec.stream.getTracks().forEach(function (t) { t.stop(); }); return; }
        var mime = pickMime();
        rec.mime = mime;
        rec.recorder = mime ? new MediaRecorder(rec.stream, { mimeType: mime }) : new MediaRecorder(rec.stream);
        rec.mime = rec.recorder.mimeType || mime;
        rec.recorder.ondataavailable = function (e) { if (e.data && e.data.size) rec.chunks.push(e.data); };
        rec.recorder.onstop = () => this._recorded(rec);
        rec.recorder.start();
        rec.timer = setTimeout(() => this._finishRecording(), MAX_RECORD_MS);
        if (rec.stopRequested) this._finishRecording();
      } catch (e) {
        this._abortRecording();
        this._showError(friendlyError(e && e.code === 'nomic' ? 'nomic' : 'mic'));
      }
    }

    _abortRecording() {
      var rec = this._s.rec;
      this._s.rec = null;
      if (rec) {
        clearTimeout(rec.timer);
        if (rec.recorder) { rec.recorder.ondataavailable = null; rec.recorder.onstop = null; try { if (rec.recorder.state !== 'inactive') rec.recorder.stop(); } catch (e) { /* ignore */ } }
        if (rec.stream) rec.stream.getTracks().forEach(function (t) { t.stop(); });
      }
      this._setState('idle');
    }

    _recorded(rec) {
      clearTimeout(rec.timer);
      if (rec.stream) rec.stream.getTracks().forEach(function (t) { t.stop(); });
      if (this._s.rec !== rec) return;
      this._s.rec = null;
      var ms = Date.now() - rec.startedAt;
      var blob = new Blob(rec.chunks, { type: rec.mime || 'audio/webm' });
      if (ms < MIN_RECORD_MS || !blob.size) {
        this._setState('idle');
        this._showError('Hold the button while you speak.');
        return;
      }
      this._sendTurn(blob, extFor(rec.mime));
    }

    /* ---- the turn */
    async _sendTurn(blob, ext, retried) {
      var s = this._s;
      this._setState('thinking');
      this._ensureQueue();
      var ctl = new AbortController();
      s.abort = ctl;
      s.capLine = null;
      try {
        var session = await this._ensureSession();
        var fd = new FormData();
        fd.append('audio', blob, 'recording.' + ext);
        fd.append('session', session.token);
        var res = await fetch(this._api + '/api/widget/turn/stream', { method: 'POST', body: fd, signal: ctl.signal });
        if (!res.ok) {
          var body = await res.json().catch(function () { return {}; });
          if (res.status === 401 && !retried) { s.lastToken = s.session && s.session.token; s.session = null; return this._sendTurn(blob, ext, true); }
          var err = new Error('turn'); err.code = body.code || 'generic'; throw err;
        }
        s.turnError = null;
        await this._readStream(res, ctl);
        if (s.abort !== ctl) return;
        if (s.turnError) { var te = new Error('turn'); te.code = s.turnError; throw te; }
        await this._queue.idle();
        if (s.abort === ctl) { s.abort = null; if (!s.rec) this._setState('idle'); }
      } catch (e) {
        if (ctl.signal.aborted) return;
        s.abort = null;
        this._queue.stop();
        this._setState('idle');
        this._showError(friendlyError(e && e.code ? e.code : 'network'));
      }
    }

    async _readStream(res, ctl) {
      var parser = new SSEParser();
      var handle = (ev) => this._onEvent(ev);
      if (!res.body || !res.body.getReader) { parser.push(await res.text()).forEach(handle); return; }
      var reader = res.body.getReader();
      var dec = new TextDecoder();
      for (;;) {
        var r = await reader.read();
        if (r.done) break;
        parser.push(dec.decode(r.value, { stream: true })).forEach(handle);
        if (ctl.signal.aborted) { try { reader.cancel(); } catch (e) { /* ignore */ } return; }
      }
    }

    _onEvent(ev) {
      var d = ev.data || {};
      if (ev.event === 'transcript') {
        if (d.user_text) this._addCaption('you', d.user_text);
        this._s.capLine = null;
      } else if (ev.event === 'sentence') {
        if (d.filler) { this._setStatus('One moment...'); }
        this._queue.add({ index: d.index, text: d.text || '', url: d.audio_url || '', filler: !!d.filler });
      } else if (ev.event === 'error') {
        this._s.turnError = d.code || 'generic';
      }
    }

    _ensureQueue() {
      if (this._queue) return;
      this._queue = new AudioQueue({
        fetchClip: (item) => this._fetchClip(item),
        play: (url) => this._play(url),
        onStart: (item) => { if (!item.filler) { this._addAgentText(item.text); } this._setState('speaking'); },
        onError: () => { /* the caption is still shown; a silent clip should not stop the reply */ }
      });
    }

    async _fetchClip(item) {
      var url = item.url;
      if (!url) return null;
      if (url.indexOf('data:') === 0) return url;
      var res = await fetch(this._api + url, { headers: { Authorization: 'Bearer ' + this._s.session.token } });
      if (!res.ok) throw new Error('clip');
      return URL.createObjectURL(await res.blob());
    }

    _play(url) {
      var a = this._audio || (this._audio = new Audio());
      return new Promise(function (resolve, reject) {
        var done = function (fn, v) { a.onended = a.onerror = null; if (url.indexOf('blob:') === 0) URL.revokeObjectURL(url); fn(v); };
        a.onended = function () { done(resolve); };
        a.onerror = function () { done(reject, new Error('audio')); };
        a.src = url;
        var p = a.play();
        if (p && p.catch) p.catch(function (e) { done(reject, e); });
      });
    }

    _stopPlayback() {
      if (this._queue) this._queue.stop();
      if (this._audio) { try { this._audio.pause(); } catch (e) { /* ignore */ } this._audio.onended = this._audio.onerror = null; }
    }

    _stopEverything() {
      if (this._s.abort) { this._s.abort.abort(); this._s.abort = null; }
      this._abortRecording();
      this._stopPlayback();
    }

    /* ---- UI helpers */
    _setState(state) {
      this._s.state = state;
      if (!this._el) return;
      var rec = this._s.rec;
      var latched = rec && rec.latched;
      var label = { idle: 'Hold to talk', recording: latched ? 'Tap to send' : 'Release to send', thinking: 'Thinking', speaking: 'Speaking. Hold to interrupt and talk' }[state];
      var text = { idle: 'Hold the button to talk', recording: latched ? 'Listening... tap to send' : 'Listening...', thinking: 'Thinking...', speaking: 'Speaking...' }[state];
      this._el.talk.classList.toggle('rec', state === 'recording');
      this._el.talk.disabled = state === 'thinking';
      this._el.talk.setAttribute('aria-label', label);
      this._setStatus(text);
    }

    _setStatus(text) { if (this._el) this._el.status.textContent = text; }

    _showError(text) {
      this._el.err.hidden = !text;
      this._el.err.textContent = text || '';
    }

    _addCaption(kind, text) {
      var p = document.createElement('p');
      if (kind) p.className = kind;
      p.textContent = text;
      var caps = this._el.caps;
      if (kind !== 'hint') { [].slice.call(caps.querySelectorAll('.hint')).forEach(function (n) { n.remove(); }); }
      caps.appendChild(p);
      while (caps.children.length > 8) caps.removeChild(caps.firstChild);
      caps.scrollTop = caps.scrollHeight;
      return p;
    }

    _addAgentText(text) {
      if (!text) return;
      var line = this._s.capLine;
      if (!line || !line.isConnected) { line = this._s.capLine = this._addCaption('', ''); }
      line.textContent = line.textContent ? line.textContent + ' ' + text : text;
      this._el.caps.scrollTop = this._el.caps.scrollHeight;
    }
  }

  if (typeof customElements !== 'undefined' && typeof HTMLElement !== 'undefined' && !customElements.get('bonclay-agent')) {
    customElements.define('bonclay-agent', BonClayAgent);
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SSEParser: SSEParser, AudioQueue: AudioQueue, friendlyError: friendlyError, extFor: extFor, safeColor: safeColor, safeApi: safeApi, BonClayAgent: BonClayAgent };
  }
})();
