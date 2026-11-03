import type { OnboardingStatus } from "@/types/api";

/** One row of the setup checklist, from GET /api/onboarding (`setup`). */
export interface SetupStep {
  id: string;
  title: string;
  done: boolean;
  /** One line on why this step matters. */
  why: string;
  minutes: number;
  /** The button label, and where it goes. */
  cta: string;
  href: string;
}

/** GET /api/onboarding with the fields the Home page reads (the older `steps`, `completed`, `total` are still sent). */
export interface HomeOnboarding extends OnboardingStatus {
  setup?: SetupStep[];
  setup_completed?: number;
  setup_total?: number;
  agent_id?: string | null;
  has_agent?: boolean;
  phone_connected?: boolean;
  approvals_waiting?: number;
}

export const PHONE_HREF = "/dashboard/studio/agents";

/** The agents page, phone tab, for the user's first agent (the plain agents page when there is none yet). */
export function phoneHref(data?: HomeOnboarding | null): string {
  return data?.agent_id ? `${PHONE_HREF}?agent=${data.agent_id}&tab=phone` : PHONE_HREF;
}
