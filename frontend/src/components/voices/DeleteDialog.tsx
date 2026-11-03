"use client";

import { useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui";
import { Modal } from "./Modal";
import type { VoiceView } from "./voiceApi";

interface Props {
  voice: VoiceView | null;
  onCancel: () => void;
  /** Should resolve once the delete has been requested; the dialog closes right after. */
  onConfirm: (voice: VoiceView) => Promise<void> | void;
}

export function DeleteDialog({ voice, onCancel, onConfirm }: Props) {
  const [busy, setBusy] = useState(false);

  const confirm = async () => {
    if (!voice) return;
    setBusy(true);
    try {
      await onConfirm(voice);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={!!voice}
      onClose={busy ? () => {} : onCancel}
      size="sm"
      title={voice ? `Withdraw consent and delete ${voice.name}?` : "Withdraw consent and delete"}
      description={
        voice
          ? `This deletes the voice, its sample, its embedding and its consent recording. It cannot be undone. Audio you already generated stays in your history, and its receipts are kept and marked "voice withdrawn".`
          : undefined
      }
    >
      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button variant="ghost" onClick={onCancel} disabled={busy} data-autofocus>
          Keep voice
        </Button>
        <Button className="!border-transparent !bg-[var(--color-accent-error)]" onClick={confirm} disabled={busy}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          Withdraw consent and delete
        </Button>
      </div>
    </Modal>
  );
}
