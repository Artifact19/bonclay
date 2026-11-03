"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Pause, Play, Waves } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { useStats } from "@/hooks/useApi";
import { API_URL } from "@/lib/api";
import { AiGeneratedTag } from "@/components/audio/AiGeneratedTag";
import { clock, relativeTime, titleCase } from "./format";

const resolveUrl = (u: string) => (u.startsWith("/") ? `${API_URL}${u}` : u);

export function RecentActivity({ limit = 10, hideWhenEmpty = false }: { limit?: number; hideWhenEmpty?: boolean } = {}) {
  const { data, error, isLoading, mutate } = useStats();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      audioRef.current = null;
    };
  }, []);

  const toggle = useCallback(
    (id: string, url: string) => {
      const current = audioRef.current;
      if (current && playingId === id) {
        current.pause();
        setPlayingId(null);
        return;
      }
      current?.pause();
      const audio = new Audio(resolveUrl(url));
      audioRef.current = audio;
      audio.onended = () => setPlayingId((p) => (p === id ? null : p));
      audio.onerror = () => {
        setPlayingId((p) => (p === id ? null : p));
        toast.error("This clip could not be played.");
      };
      setPlayingId(id);
      audio.play().catch(() => {
        setPlayingId((p) => (p === id ? null : p));
      });
    },
    [playingId]
  );

  const recent = (data?.recent ?? []).slice(0, limit);

  if (hideWhenEmpty && !isLoading && !error && recent.length === 0) return null;

  return (
    <section aria-labelledby="recent-heading">
      <h2 id="recent-heading" className="mb-2 text-sm font-semibold text-text-primary">
        Recent audio clips
      </h2>
      {error && !data ? (
        <ErrorState title="Could not load activity" onRetry={() => mutate()} />
      ) : isLoading ? (
        <Card variant="solid" padded={false} className="divide-y divide-border">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 p-4">
              <Skeleton className="size-9 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-3.5 w-3/5" />
                <Skeleton className="h-3 w-2/5" />
              </div>
            </div>
          ))}
        </Card>
      ) : recent.length === 0 ? (
        <EmptyState
          icon={<Waves />}
          title="No audio clips yet"
          description="Make your first clip in the Studio and it will show up here with a play button."
          action={
            <Link href="/dashboard/studio">
              <Button size="sm">Open Studio</Button>
            </Link>
          }
        />
      ) : (
        <Card variant="solid" padded={false}>
          <ul className="divide-y divide-border">
            {recent.map((g) => {
              const playing = playingId === g.id;
              return (
                <li key={g.id} className="flex items-center gap-3 p-3 sm:p-4">
                  <button
                    type="button"
                    disabled={!g.audio_url}
                    onClick={() => g.audio_url && toggle(g.id, g.audio_url)}
                    aria-label={playing ? "Pause clip" : "Play clip"}
                    className="flex size-9 shrink-0 items-center justify-center rounded-full bg-accent-primary/15 text-accent-primary transition-colors duration-200 hover:bg-accent-primary hover:text-white disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-accent-primary/15 disabled:hover:text-accent-primary"
                  >
                    {playing ? <Pause className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
                  </button>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm text-text-primary">{g.text}</p>
                    <p className="mt-0.5 truncate text-xs text-text-secondary">
                      {[g.voice_name, relativeTime(g.created_at), clock(g.duration)].filter(Boolean).join(" · ")}
                    </p>
                  </div>
                  {g.audio_url && <AiGeneratedTag className="hidden shrink-0 sm:inline-flex" />}
                  {g.engine && <Badge className="hidden shrink-0 sm:inline-flex">{titleCase(g.engine)}</Badge>}
                </li>
              );
            })}
          </ul>
        </Card>
      )}
    </section>
  );
}
