export type CampaignStatus = "draft" | "running" | "paused" | "done" | "stopped";
export type ContactStatus = "pending" | "calling" | "done" | "failed" | "dnc";

export interface CampaignWindow {
  days: number[]; // 0 = Monday ... 6 = Sunday
  start: string; // "HH:MM" local time
  end: string;
  timezone: string;
}

export interface CampaignContact {
  name: string;
  number: string; // masked by the server
  status: ContactStatus;
  attempts: number;
  last_status: string | null;
}

export interface Campaign {
  id: string;
  name: string;
  agent_id: string;
  status: CampaignStatus;
  created_at: number;
  finished_at: number | null;
  window: CampaignWindow;
  max_attempts: number;
  retry_minutes: number;
  concurrency: number;
  max_per_hour: number;
  paused_reason: string | null;
  counts: Record<ContactStatus, number>;
  total: number;
  contacts?: CampaignContact[];
  import?: { invalid: number; duplicates: number; over_limit: number };
}

export interface CampaignsResponse {
  campaigns: Campaign[];
}

export interface NewCampaignBody {
  name: string;
  agent_id: string;
  csv: string;
  caller_agreed: boolean;
  window: Partial<CampaignWindow>;
  max_attempts: number;
  retry_minutes: number;
  concurrency: number;
}
