"use client";

import { useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui";
import { Modal } from "@/components/voices/Modal";

interface Props {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  keepLabel?: string;
  onCancel: () => void;
  /** Resolves once the action finished; the dialog stays busy until then. */
  onConfirm: () => Promise<void> | void;
}

export function ConfirmDialog({ open, title, description, confirmLabel, keepLabel = "Cancel", onCancel, onConfirm }: Props) {
  const [busy, setBusy] = useState(false);

  const confirm = async () => {
    setBusy(true);
    try {
      await onConfirm();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={busy ? () => {} : onCancel} size="sm" title={title} description={description}>
      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button variant="ghost" onClick={onCancel} disabled={busy} data-autofocus>
          {keepLabel}
        </Button>
        <Button className="!border-transparent !bg-[var(--color-accent-error)]" onClick={confirm} disabled={busy}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
          {confirmLabel}
        </Button>
      </div>
    </Modal>
  );
}
