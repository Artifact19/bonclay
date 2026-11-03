"use client";

import { useEffect, useRef, useState } from "react";
import { Headphones, Hand, Mic, PhoneOff, Volume2, VolumeX } from "lucide-react";
import { Badge, Button, Card } from "@/components/ui";
import { API_URL, api } from "@/lib/api";
import { MicStreamer, base64ToInt16, int16ToBase64, monitorUrl, type PcmPlayer, type Voice } from "@/lib/livePcm";
import type { LiveCall } from "./types";

interface Line { role: "caller" | "agent"; text: string }
type Phase = "connecting" | "listening" | "ended" | "error";

/** Listen in on one live call, read it as it goes, and take it over. Closing this panel while you have the call hands it
 *  back to the agent (the server does the same if the connection drops). */
export function LiveMonitor({ call, player, onClose }: { call: LiveCall; player: PcmPlayer; onClose: () => void }) {
  const [phase, setPhase] = useState<Phase>("connecting");
  const [lines, setLines] = useState<Line[]>([]);
  const [human, setHuman] = useState(false);
  const [mine, setMine] = useState(false);       // I am the person on the call
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const mic = useRef<MicStreamer | null>(null);
  const rate = useRef(8000);
  const mineRef = useRef(false);

  const stopMic = () => { mic.current?.stop(); mic.current = null; };

  useEffect(() => {
    let closed = false;
    let sock: WebSocket | null = null;
    (async () => {
      try {
        const { token } = await api.post<{ token: string }>(`/api/phone/exotel/live/${encodeURIComponent(call.conversation_id)}/monitor-token`, {});
        if (closed) return;
        sock = new WebSocket(monitorUrl(API_URL, token));
        ws.current = sock;
        sock.onmessage = (ev) => {
          let m: Record<string, unknown>;
          try { m = JSON.parse(String(ev.data)); } catch { return; }
          if (m.type === "hello") { rate.current = Number(m.rate) || 8000; setHuman(Boolean(m.human)); setPhase("listening"); }
          else if (m.type === "audio") player.play(m.from as Voice, base64ToInt16(String(m.pcm)), Number(m.rate) || rate.current);
          else if (m.type === "transcript") setLines((l) => [...l, { role: m.role as Line["role"], text: String(m.text) }].slice(-200));
          else if (m.type === "state") {
            setHuman(Boolean(m.human));
            if (!m.human) { mineRef.current = false; setMine(false); stopMic(); }
          } else if (m.type === "error") setError(String(m.message || "Something went wrong."));
          else if (m.type === "ended") { setPhase("ended"); mineRef.current = false; setMine(false); stopMic(); }
        };
        sock.onclose = () => { setPhase((p) => (p === "ended" ? p : p === "connecting" ? "error" : "ended")); stopMic(); };
        sock.onerror = () => setError("Could not listen in on this call.");
      } catch {
        if (!closed) { setPhase("error"); setError("This call is no longer live."); }
      }
    })();
    return () => { closed = true; stopMic(); sock?.close(); };
  }, [call.conversation_id, player]);

  const takeOver = async () => {
    setError(null);
    try {
      const streamer = new MicStreamer();
      await streamer.start(rate.current, (pcm) => {
        if (mineRef.current && ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify({ type: "audio", pcm: int16ToBase64(pcm) }));
      });
      mic.current = streamer;
      mineRef.current = true;
      setMine(true);
      ws.current?.send(JSON.stringify({ type: "takeover" }));
    } catch {
      stopMic();
      setError("Allow the microphone to talk to the caller.");
    }
  };
  const handBack = () => { ws.current?.send(JSON.stringify({ type: "release" })); };
  const toggleMute = () => { player.setMuted(!muted); setMuted(!muted); };

  return (
    <Card className="space-y-3 border-accent-primary/40 p-4" role="region" aria-label={`Listening in on ${call.agent_name}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Headphones className="size-4 text-text-secondary" aria-hidden />
          <span className="text-sm font-semibold text-text-primary">{call.agent_name}</span>
          <Badge tone={phase === "listening" ? "success" : phase === "ended" ? "neutral" : "warning"} dot>
            {phase === "connecting" ? "Connecting" : phase === "listening" ? (human ? (mine ? "You have the call" : "A person has the call") : "Listening") : phase === "ended" ? "Call ended" : "Not connected"}
          </Badge>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="ghost" onClick={toggleMute} aria-pressed={muted}>
            {muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}{muted ? "Unmute" : "Mute"}
          </Button>
          {phase === "listening" && !human && <Button size="sm" onClick={takeOver}><Mic className="size-4" />Take over</Button>}
          {phase === "listening" && mine && <Button size="sm" variant="outline" onClick={handBack}><Hand className="size-4" />Hand back to the agent</Button>}
          <Button size="sm" variant="ghost" onClick={onClose}><PhoneOff className="size-4" />Close</Button>
        </div>
      </div>
      {human && !mine && phase === "listening" && <p className="text-sm text-text-secondary">Someone else has taken this call over. You can keep listening.</p>}
      {mine && <p role="status" className="text-sm text-text-secondary">The agent is quiet. Speak to the caller. Closing this panel hands the call back.</p>}
      {error && <p role="alert" className="text-sm text-accent-error">{error}</p>}
      <ol role="log" aria-label="What is being said" className="max-h-56 space-y-1.5 overflow-y-auto rounded-lg border border-border p-3 text-sm">
        {lines.length === 0 && <li className="text-text-secondary">{phase === "listening" ? "Waiting for someone to speak…" : "…"}</li>}
        {lines.map((l, i) => (
          <li key={i}><span className="font-medium text-text-secondary">{l.role === "caller" ? "Caller" : "Agent"}: </span><span className="text-text-primary">{l.text}</span></li>
        ))}
      </ol>
    </Card>
  );
}
