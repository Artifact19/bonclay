import type { Agent, AgentTemplate } from "@/types/api";

/** Questions to try when the agent did not come from a template. Written for a caller of a clinic, college or shop. */
export const DEFAULT_QUESTIONS = {
  en: ["What are your opening hours?", "How can I book an appointment?", "How much does it cost?"],
  hi: ["आपका समय क्या है?", "मैं अपॉइंटमेंट कैसे बुक करूँ?", "इसकी फीस कितनी है?"],
  hinglish: ["Aapka timing kya hai?", "Appointment kaise book karun?", "Fees kitni hai?"],
} as const;

export type QuestionLanguage = keyof typeof DEFAULT_QUESTIONS;

const MAX = 4;

/** The template an agent came from: an explicit id when the agent carries one, else the template with the same name. */
export function templateForAgent(
  agent: (Pick<Agent, "name"> & { template_id?: string | null }) | null | undefined,
  templates: AgentTemplate[] | undefined
): AgentTemplate | null {
  if (!agent || !templates?.length) return null;
  if (agent.template_id) {
    const byId = templates.find((t) => t.id === agent.template_id);
    if (byId) return byId;
  }
  const name = agent.name.trim().toLowerCase();
  return templates.find((t) => t.name.trim().toLowerCase() === name) ?? null;
}

/**
 * Up to four questions to click: the template's own suggested questions when the agent came from one,
 * else plain defaults in the agent's languages (English, Hindi, and Hinglish when it speaks both).
 */
export function suggestedQuestionsFor(
  agent: (Pick<Agent, "name" | "supported_languages"> & { template_id?: string | null }) | null | undefined,
  templates: AgentTemplate[] | undefined
): string[] {
  if (!agent) return [];
  const own = templateForAgent(agent, templates)?.suggested_questions?.filter((q) => q.trim()) ?? [];
  if (own.length) return own.slice(0, MAX);
  const langs = agent.supported_languages ?? [];
  const hindi = langs.includes("hi");
  const english = langs.includes("en") || !langs.length;
  const out: string[] = [];
  if (english) out.push(...DEFAULT_QUESTIONS.en.slice(0, hindi ? 2 : 3));
  if (hindi) out.push(...DEFAULT_QUESTIONS.hinglish.slice(0, english ? 1 : 0), ...DEFAULT_QUESTIONS.hi.slice(0, english ? 1 : 3));
  if (!out.length) out.push(...DEFAULT_QUESTIONS.en);
  return out.slice(0, MAX);
}
