import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ConsentNotice } from "@/components/voices/ConsentNotice";
import { ConsentRecordDialog, ConsentRecordPanel } from "@/components/voices/ConsentRecordPanel";
import { DeleteDialog } from "@/components/voices/DeleteDialog";
import { VoiceCard } from "@/components/voices/VoiceCard";
import { NOTICE_VERSION } from "@/components/voices/noticeText";
import { toView, type ConsentRecord } from "@/components/voices/voiceApi";
import { api } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  api: { get: jest.fn(), del: jest.fn() },
  ApiError: class ApiError extends Error {},
}));

jest.mock("@/components/voices/VoiceThumb", () => ({ VoiceThumb: () => null })); // canvas + ResizeObserver

const record: ConsentRecord = {
  record_version: 1,
  notice_version: NOTICE_VERSION,
  voice_id: "0123456789abcdef",
  voice_name: "Asha",
  consent_id: "c1",
  purpose: "Create an AI version of this voice for use in BonClay by Acme Clinic.",
  data_stored: ["voice sample", "consent recording", "consent transcript (the sentence you read)", "voice embedding (a numeric fingerprint of the sample)"],
  challenge_sentence: "I, Asha, consent to creating an AI version of my voice with BonClay on 15 October 2026. My code is mango 42.",
  challenge_code: "mango 42",
  challenge_id: "ch1",
  transcript_match_score: 0.95,
  transcript_verified: true,
  speaker_match_score: 0.81,
  speaker_match_threshold: 0.25,
  speaker_checked: true,
  language: "en",
  engine: "qwen3_0_6b",
  consented_at: "2026-10-15T10:00:00+00:00",
  retention_days: 365,
  retain_until: "2027-10-15T10:00:00+00:00",
  how_to_withdraw: 'Choose "Withdraw consent and delete" on the voice in the Voices page.',
  contact: "privacy@example.org",
  legacy: false,
};

const voice = toView({
  id: "0123456789abcdef",
  name: "Asha",
  display_name: "Asha",
  language: "en",
  engine: "qwen3_0_6b",
  created_at: "2026-10-15T10:00:00",
  consent_id: "c1",
  consent_verified: true,
});

beforeEach(() => {
  (api.get as jest.Mock).mockReset();
  (api.get as jest.Mock).mockResolvedValue(record);
});

describe("consent record panel", () => {
  it("shows every itemised field of the record", async () => {
    render(<ConsentRecordPanel voice={voice} />);
    expect(screen.getByRole("status")).toHaveTextContent(/Loading the consent record/);
    expect(await screen.findByTestId("consent-record")).toBeInTheDocument();
    expect((api.get as jest.Mock).mock.calls[0][0]).toBe("/api/voices/0123456789abcdef/consent-record");
    expect(screen.getByText(/for use in BonClay by Acme Clinic/)).toBeInTheDocument();
    for (const item of record.data_stored) expect(screen.getByText(item)).toBeInTheDocument();
    expect(screen.getByText(/My code is mango 42/)).toBeInTheDocument();
    expect(screen.getByText("mango 42")).toBeInTheDocument();
    expect(screen.getByText("95% (verified)")).toBeInTheDocument();
    expect(screen.getByText("81% (needed 25%)")).toBeInTheDocument();
    expect(screen.getByTestId("kept-until")).toHaveTextContent(/2027/);
    expect(screen.getByText(/365 days, or until you withdraw/)).toBeInTheDocument();
    expect(screen.getByText(NOTICE_VERSION)).toBeInTheDocument();
    expect(screen.getByText("privacy@example.org")).toBeInTheDocument();
    expect(screen.getByText(/not legal advice/i)).toBeInTheDocument();
  });

  it("downloads the record as JSON", async () => {
    const created: Blob[] = [];
    const createObjectURL = jest.fn((b: Blob) => {
      created.push(b);
      return "blob:record";
    });
    const revokeObjectURL = jest.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    const click = jest.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);

    render(<ConsentRecordPanel voice={voice} />);
    fireEvent.click(await screen.findByRole("button", { name: /Download as JSON/ }));
    expect(click).toHaveBeenCalledTimes(1);
    expect(created).toHaveLength(1);
    expect(created[0].type).toBe("application/json");
    const text = await new Promise<string>((resolve) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result));
      r.readAsText(created[0]);
    });
    expect(JSON.parse(text)).toMatchObject({ voice_id: record.voice_id, notice_version: NOTICE_VERSION, challenge_code: "mango 42" });
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:record");
    click.mockRestore();
  });

  it("says plainly when a voice was cloned before records existed", async () => {
    (api.get as jest.Mock).mockResolvedValue({
      ...record, legacy: true, notice_version: null, challenge_sentence: null, challenge_code: null,
      transcript_match_score: null, speaker_match_score: null, speaker_match_threshold: null, speaker_checked: false,
    });
    render(<ConsentRecordPanel voice={voice} />);
    expect(await screen.findByText(/cloned before itemised records were kept/)).toBeInTheDocument();
    expect(screen.getByText("Before notices were versioned")).toBeInTheDocument();
    expect(screen.getAllByText("Not checked").length).toBeGreaterThan(0);
  });

  it("shows an error when the record cannot be loaded", async () => {
    (api.get as jest.Mock).mockRejectedValue(new Error("Voice not found"));
    render(<ConsentRecordPanel voice={voice} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Voice not found");
  });

  it("offers the withdraw action from the record", async () => {
    const onWithdraw = jest.fn();
    render(<ConsentRecordDialog voice={voice} onClose={() => undefined} onWithdraw={onWithdraw} />);
    fireEvent.click(await screen.findByRole("button", { name: "Withdraw consent and delete" }));
    expect(onWithdraw).toHaveBeenCalledWith(voice);
  });
});

describe("withdraw consent and delete", () => {
  it("sits in the card's overflow menu and asks the parent to confirm", () => {
    const onDelete = jest.fn();
    const onShowConsent = jest.fn();
    render(<VoiceCard voice={voice} onGenerate={() => undefined} onDelete={onDelete} onShowConsent={onShowConsent} />);
    // Not a front-line button: it is behind the "..." menu.
    expect(screen.queryByRole("button", { name: "Withdraw consent and delete Asha" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "More actions for Asha" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Consent record" }));
    expect(onShowConsent).toHaveBeenCalledWith(voice);
    fireEvent.click(screen.getByRole("button", { name: "More actions for Asha" }));
    const del = screen.getByRole("menuitem", { name: "Withdraw consent and delete Asha" });
    expect(del).toHaveTextContent("Withdraw consent and delete");
    fireEvent.click(del);
    expect(onDelete).toHaveBeenCalledWith(voice);
    // the menu closes after a choice
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("shows status and language instead of the engine, with the engine under Details", () => {
    render(<VoiceCard voice={voice} onGenerate={() => undefined} onDelete={() => undefined} coverage={{ names: ["Hindi", "Marathi", "Tamil"], more: 19, codes: ["hi", "mr", "ta"] }} />);
    expect(screen.getByTestId("voice-status")).toHaveTextContent(/Consent recorded|Ready/);
    expect(screen.getByTestId("voice-languages")).toHaveTextContent("Also speaks Hindi, Marathi, Tamil and 19 more");
    const details = screen.getByText("Details").closest("details") as HTMLElement;
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveTextContent("Engine");
  });

  it("offers Preview and Use for an agent on every card", () => {
    const onGenerate = jest.fn();
    render(<VoiceCard voice={voice} onGenerate={onGenerate} onDelete={() => undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Preview Asha" }));
    expect(onGenerate).toHaveBeenCalledWith(voice);
    expect(screen.getByRole("link", { name: "Use for an agent" })).toHaveAttribute("href", `/dashboard/studio/agents?voice=${voice.id}`);
  });

  it("only deletes after the confirm dialog is confirmed, and explains what happens", async () => {
    const onConfirm = jest.fn().mockResolvedValue(undefined);
    const onCancel = jest.fn();
    render(<DeleteDialog voice={voice} onCancel={onCancel} onConfirm={onConfirm} />);
    expect(screen.getByText(/Withdraw consent and delete Asha\?/)).toBeInTheDocument();
    expect(screen.getByText(/receipts are kept and marked "voice withdrawn"/)).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Keep voice" }));
    expect(onCancel).toHaveBeenCalled();
    expect(onConfirm).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Withdraw consent and delete$/ }));
    await waitFor(() => expect(onConfirm).toHaveBeenCalledWith(voice));
  });
});

describe("consent notice", () => {
  it("covers what is collected, why, how long, how to withdraw and who to contact, with its version", () => {
    render(<ConsentNotice retentionDays={90} contact="privacy@example.org" />);
    const notice = screen.getByTestId("consent-notice");
    for (const title of ["What we collect", "Why", "How long we keep it", "How to withdraw", "Contact"]) {
      expect(notice).toHaveTextContent(`${title}:`);
    }
    expect(notice).toHaveTextContent("90 days from today");
    expect(notice).toHaveTextContent("privacy@example.org");
    expect(notice).toHaveTextContent(`Notice version ${NOTICE_VERSION}`);
  });

  it("falls back to 365 days and the studio operator when the server did not say", () => {
    render(<ConsentNotice />);
    expect(screen.getByTestId("consent-notice")).toHaveTextContent("365 days from today");
    expect(screen.getByTestId("consent-notice")).toHaveTextContent("the person who runs this BonClay studio");
  });
});
