import { render, screen } from "@testing-library/react";
import { VerifyResult } from "@/components/verify/VerifyResult";
import type { PublicReceipt, VerifyResponse } from "@/components/verify/types";

const receipt: PublicReceipt = {
  receipt_id: "5b1f7d3e-0000-4000-8000-000000000001",
  kind: "generation",
  created_at: "2026-10-14T10:30:00+00:00",
  engine: "qwen3_0_6b",
  voice_id: "v1",
  voice_name: "Asha",
  consent_id: "c1",
  consent_verified: true,
  signature_valid: true,
};

const base: VerifyResponse = {
  watermark: { detected: true, probability: 0.98 },
  match: { exact: true, receipt, signature_valid: true, candidates: [] },
  verdict: "made_with_bonclay",
};

describe("VerifyResult", () => {
  it("shows the verdict, confidence and receipt details for an exact match", () => {
    render(<VerifyResult result={base} />);
    expect(screen.getByRole("heading", { name: "Made with BonClay" })).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Watermark confidence" })).toHaveAttribute("aria-valuenow", "98");
    expect(screen.getByText("Consent verified")).toBeInTheDocument();
    expect(screen.getByText("Asha")).toBeInTheDocument();
    expect(screen.getByText("Qwen3-TTS")).toBeInTheDocument();
    expect(screen.getByText("Exact copy of the original file")).toBeInTheDocument();
    expect(screen.getByText(/Valid: the receipt has not been altered/)).toBeInTheDocument();
  });

  it("says so plainly when the signature does not verify", () => {
    render(<VerifyResult result={{ ...base, match: { ...base.match, receipt: { ...receipt, signature_valid: false } } }} />);
    expect(screen.getByText(/Invalid: the receipt does not verify/)).toBeInTheDocument();
  });

  it("does not claim consent for a preset voice", () => {
    const preset = { ...receipt, voice_id: null, consent_id: null, consent_verified: null, voice_name: "af_heart", engine: "kokoro" };
    render(<VerifyResult result={{ ...base, match: { ...base.match, exact: false, receipt: preset } }} />);
    expect(screen.queryByText("Consent verified")).not.toBeInTheDocument();
    expect(screen.getByText("Preset voice (no cloning)")).toBeInTheDocument();
    expect(screen.getByText("Found through the watermark")).toBeInTheDocument();
  });

  it("reports a clip with no watermark without implying it is genuine", () => {
    render(
      <VerifyResult
        result={{
          watermark: { detected: false, probability: 0.03 },
          match: { exact: false, receipt: null, signature_valid: false, candidates: [] },
          verdict: "not_detected",
        }}
      />,
    );
    expect(screen.getByRole("heading", { name: "Not detected" })).toBeInTheDocument();
    expect(screen.getByText(/does not show the clip is genuine/)).toBeInTheDocument();
    expect(screen.queryByTestId("receipt")).not.toBeInTheDocument();
  });

  it("lists candidates when several generations share the watermark message", () => {
    render(
      <VerifyResult
        result={{
          watermark: { detected: true, probability: 1 },
          match: { exact: false, receipt: null, signature_valid: false, candidates: [receipt, { ...receipt, receipt_id: "x2", voice_name: "Ravi" }] },
          verdict: "made_with_bonclay",
        }}
      />,
    );
    expect(screen.getByText(/matches 2 possible generations/)).toBeInTheDocument();
    expect(screen.getByText("Ravi")).toBeInTheDocument();
  });

  it("says the voice owner withdrew consent, while keeping the receipt", () => {
    const withdrawn = { ...receipt, voice_withdrawn_at: "2026-10-20T09:00:00+00:00", voice_removed_reason: "withdrawn" as const };
    render(<VerifyResult result={{ ...base, match: { ...base.match, receipt: withdrawn } }} />);
    expect(screen.getByTestId("voice-withdrawn")).toHaveTextContent("Voice withdrawn");
    expect(screen.getByText(/The voice owner withdrew consent on/)).toBeInTheDocument();
    expect(screen.getByText("Consent verified when made, now withdrawn")).toBeInTheDocument();
    expect(screen.getByText(/Valid: the receipt has not been altered/)).toBeInTheDocument();
  });

  it("does not say consent was withdrawn when the voice was only deleted at the end of its retention period", () => {
    const expired = { ...receipt, voice_withdrawn_at: "2027-10-20T09:00:00+00:00", voice_removed_reason: "retention_expired" as const };
    render(<VerifyResult result={{ ...base, match: { ...base.match, receipt: expired } }} />);
    expect(screen.getByText(/when its retention period ended/)).toBeInTheDocument();
    expect(screen.queryByText(/withdrew consent on/)).not.toBeInTheDocument();
  });

  it("shows no withdrawal note for a voice that still exists", () => {
    render(<VerifyResult result={base} />);
    expect(screen.queryByTestId("voice-withdrawn")).not.toBeInTheDocument();
  });
});
