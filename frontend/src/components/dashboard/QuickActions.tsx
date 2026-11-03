"use client";

import { Bot, CalendarCheck, Mic, Phone } from "lucide-react";
import { useOnboarding } from "@/hooks/useApi";
import { phoneHref, type HomeOnboarding } from "./homeTypes";
import { TaskCards, type Task } from "./TaskCards";

/** "What do you want to do?": the four jobs a first-time owner comes for. */
export function QuickActions() {
  const { data } = useOnboarding();
  const home = data as HomeOnboarding | undefined;
  const waiting = home?.approvals_waiting ?? 0;

  const tasks: Task[] = [
    { id: "agent", href: "/dashboard/studio/agents", label: "Set up an agent", hint: "Answers your callers", Icon: Bot },
    { id: "voice", href: "/dashboard/voices?new=1", label: "Give your agent a voice", hint: "Clone yours in about 3 minutes", Icon: Mic },
    { id: "phone", href: phoneHref(home), label: "Connect my phone", hint: "Let people call you", Icon: Phone },
    {
      id: "calls",
      href: "/dashboard/calls",
      label: "See calls and approvals",
      hint: "What happened today",
      Icon: CalendarCheck,
      more: { href: "/dashboard/approvals", label: waiting > 0 ? `Approvals (${waiting} waiting)` : "Approvals" },
    },
  ];

  return <TaskCards tasks={tasks} label="What do you want to do?" />;
}
