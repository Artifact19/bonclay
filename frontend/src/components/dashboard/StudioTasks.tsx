"use client";

import { Languages, Mic, ShieldCheck, Type } from "lucide-react";
import { TaskCards, type Task } from "./TaskCards";

/** "What do you want to make?": the four things a creator comes to BonClay for. */
export const STUDIO_TASKS: Task[] = [
  { id: "speech", href: "/dashboard/studio", label: "Turn text into speech", hint: "Type or paste, pick a voice", Icon: Type },
  { id: "clone", href: "/dashboard/voices?new=1", label: "Clone my voice", hint: "Takes about 3 minutes", Icon: Mic },
  { id: "languages", href: "/dashboard/say-it-once", label: "Say it in several languages", hint: "One message, many languages", Icon: Languages },
  { id: "verify", href: "/verify", label: "Check if a clip is AI-made", hint: "Upload a clip to check it", Icon: ShieldCheck },
];

export function StudioTasks() {
  return <TaskCards tasks={STUDIO_TASKS} label="What do you want to make?" />;
}
