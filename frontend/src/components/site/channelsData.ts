/** Copy and honest status for the "One agent, every channel" section. Statuses mirror STATUS.md and the docs. */

import { REPO_URL } from "./repo";

export const DOCS_BASE = `${REPO_URL}/blob/main/docs/`;

export type ChannelId = "phone" | "meetings" | "website" | "whatsapp";

export interface Channel {
  id: ChannelId;
  n: string;
  title: string;
  /** One line shown in the rail for every channel. */
  short: string;
  /** Plain-language description (rail when pinned, card header when stacked). */
  desc: string;
  /** Bold status label, then the detail that makes it honest. */
  status: string;
  statusDetail: string;
  /** "live" = exercised for real, "tests" = automated tests only, "beta" = built but not yet live-tested. */
  tone: "live" | "tests" | "beta";
  docPath: string;
  docLabel: string;
}

export const CHANNELS: readonly Channel[] = [
  {
    id: "phone",
    n: "01",
    title: "Phone calls",
    short: "Picks up, or calls out",
    desc: "Your agent answers a call on your Exotel number, or rings someone who agreed to it. It speaks first and says it is an AI, stops when the caller talks over it, and can book an appointment.",
    status: "Tested on a live call",
    statusDetail: "One outbound call on an Exotel trial account. Answering inbound calls is not yet tested.",
    tone: "live",
    docPath: "PHONE_AND_MEETINGS.md#phone-calls-exotel",
    docLabel: "Phone setup guide",
  },
  {
    id: "meetings",
    n: "02",
    title: "Meetings for an office",
    short: "Requests the owner approves",
    desc: "For a head of department or any busy office. The agent checks the calendar, offers other times when a slot is taken and files a pending request. The owner approves it in the dashboard, and only then does the event land in Google Calendar.",
    status: "Tested live",
    statusDetail: "A spoken request, the approval and the event on a real Google Calendar, then cancelling it.",
    tone: "live",
    docPath: "PHONE_AND_MEETINGS.md#hod-meeting-requests-template-hod-office-assistant",
    docLabel: "Meetings guide",
  },
  {
    id: "website",
    n: "03",
    title: "Your website",
    short: "One script tag",
    desc: "Paste one script tag and the agent is on your site. Visitors tap the microphone, talk, and read captions with a label that the voice is AI. Sessions are signed and expire, and it only works on the sites you list.",
    status: "Tested with automated tests",
    statusDetail: "Allowed sites, signed sessions, rate limits and revoked keys are covered. Not yet reported from a customer's site.",
    tone: "tests",
    docPath: "WIDGET.md",
    docLabel: "Website widget guide",
  },
  {
    id: "whatsapp",
    n: "04",
    title: "WhatsApp voice notes",
    short: "Beta, not yet live-tested",
    desc: "A customer sends a voice note or a message. The agent replies with a voice note and the same text, in the customer's language, from your documents. It says it is an AI and hands over to a person when asked.",
    status: "Beta: not yet live-tested",
    statusDetail: "Tested against a fake WhatsApp API. It needs a Meta test number before it can go live.",
    tone: "beta",
    docPath: "WHATSAPP.md",
    docLabel: "WhatsApp guide",
  },
];

/** Deterministic waveform heights (percent) so server and client markup match. */
export const WAVE_IN = [30, 55, 40, 75, 60, 90, 50, 35, 70, 85, 45, 30, 60, 80, 55, 40, 65, 50, 35, 25, 45, 70, 55, 30];
export const WAVE_OUT = [25, 45, 65, 50, 80, 60, 40, 70, 90, 55, 35, 60, 75, 50, 30, 45, 65, 85, 55, 40, 30, 55, 70, 45, 30, 20];
