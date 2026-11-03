"use client";

import { useMemo, useRef, useState, type FormEvent } from "react";
import {
  BookOpen,
  Bot,
  GraduationCap,
  Landmark,
  Loader2,
  School,
  Sprout,
  Stethoscope,
  TriangleAlert,
  UploadCloud,
  X,
  type LucideIcon,
  FileQuestion,
} from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useAgentTemplates, useAgents } from "@/hooks/useApi";
import { Badge, Button, Card, Skeleton } from "@/components/ui";
import { Modal } from "@/components/voices/Modal";
import { errMessage } from "@/components/dashboard/format";
import type { Agent, AgentTemplate } from "@/types/api";

const ICONS: Record<string, LucideIcon> = {
  clinic_receptionist: Stethoscope,
  emi_reminder: Landmark,
  farmer_helpline: Sprout,
  college_admissions: GraduationCap,
  hindi_tutor: BookOpen,
  school_fee_reminder: School,
  document_qa: FileQuestion,
};

/** Languages offered in the form (the template's own languages are always offered too). */
const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  mr: "Marathi",
  ta: "Tamil",
  te: "Telugu",
  bn: "Bengali",
  gu: "Gujarati",
  kn: "Kannada",
  ml: "Malayalam",
  pa: "Punjabi",
  ur: "Urdu",
  or: "Odia",
};
const MORE_LANGUAGES = Object.keys(LANGUAGE_NAMES);
export const languageName = (code: string) =>
  LANGUAGE_NAMES[code] ?? code.toUpperCase();

/** What the server adds to GET /api/agent-templates for the form (older servers may not send it). */
export type TemplateInfo = AgentTemplate & {
  audience?: string;
  org_label?: string;
  org_placeholder?: string;
  personalise?: {
    name_pattern: string;
    sample_orgs: string[];
    sample_city?: string | null;
    strip: string[];
  };
};

async function downloadSamplePdf(t: TemplateInfo) {
  if (!t.sample_pdf) return;
  try {
    const blob = await api.blob(t.sample_pdf);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      t.sample_pdf
        .split("/")
        .pop()
        ?.replace(/^sample/, t.id) || `${t.id}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    toast.error(errMessage(e, "Could not download the sample PDF."));
  }
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Swap the template's sample organisation for the user's, the same way the server does it. */
export function personaFor(
  t: TemplateInfo,
  org: string,
): { name: string; greeting: string } {
  const p = t.personalise;
  if (!p) return { name: t.name, greeting: t.greeting };
  let greeting = t.greeting;
  for (const phrase of p.strip) greeting = greeting.split(phrase).join("");
  const names = [...new Set(p.sample_orgs)]
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);
  if (names.length)
    greeting = greeting.replace(
      new RegExp(names.map(escapeRe).join("|"), "g"),
      () => org,
    );
  return {
    name: p.name_pattern.split("{org}").join(org).slice(0, 120),
    greeting,
  };
}

const DOC_OK = /\.(pdf|txt|md)$/i;
const fieldClass =
  "min-h-11 w-full rounded-lg border border-border bg-bg-primary px-3 py-2.5 text-sm text-text-primary placeholder:text-text-secondary/60 focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary";
const labelClass = "mb-1.5 block text-sm font-medium text-text-primary";

interface Props {
  /** Called with the new agent and the template it came from, after the agents list is refreshed. */
  onCreated: (agent: Agent, template: AgentTemplate) => void;
  /** A narrow vertical list of rows (for a side rail) instead of the card grid. */
  compact?: boolean;
}

interface FormProps {
  template: TemplateInfo;
  existing: Agent[];
  onCancel: () => void;
  onCreated: Props["onCreated"];
  onBusy: (busy: boolean) => void;
  refresh: () => Promise<unknown>;
}

function TemplateForm({
  template: t,
  existing,
  onCancel,
  onCreated,
  onBusy,
  refresh,
}: FormProps) {
  const [org, setOrg] = useState("");
  const [city, setCity] = useState("");
  const [details, setDetails] = useState("");
  const [langs, setLangs] = useState<string[]>(t.languages);
  const [file, setFile] = useState<File | null>(null);
  const [withSample, setWithSample] = useState(true);
  const [busy, setBusy] = useState(false);
  const [dupe, setDupe] = useState<Agent | null>(null);
  const [touched, setTouched] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const orgLabel = t.org_label ?? "Organisation name";
  const orgClean = org.trim();
  const choices = useMemo(
    () => [...new Set([...t.languages, ...MORE_LANGUAGES])],
    [t.languages],
  );
  const preview = personaFor(t, orgClean || `[${orgLabel}]`);
  const orgMissing = touched && !orgClean;
  const langsMissing = langs.length === 0;

  const toggleLang = (code: string) =>
    setLangs((cur) =>
      cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code],
    );

  const create = async () => {
    setBusy(true);
    onBusy(true);
    try {
      const created = await api.post<Agent>("/api/agents/from-template", {
        template_id: t.id,
        org_name: orgClean,
        city: city.trim() || undefined,
        details: details.trim() || undefined,
        languages: langs,
        include_sample: file ? false : withSample,
      });
      await refresh();
      toast.success(
        file || !withSample
          ? `${created.name} created.`
          : `${created.name} created with a sample document you can replace.`,
      );
      if (file) {
        try {
          const fd = new FormData();
          fd.append("agent_id", created.id);
          fd.append("file", file);
          await api.upload(`/api/knowledge/upload`, fd);
          toast.success(`${file.name} added to your agent's documents.`);
          await refresh();
        } catch (e) {
          toast.error(
            errMessage(
              e,
              `Your agent was created, but ${file.name} could not be uploaded. Add it on the agent's page.`,
            ),
          );
        }
      }
      onCreated(created, t);
    } catch (e) {
      toast.error(
        errMessage(e, "Could not create the agent from this template."),
      );
      setBusy(false);
      onBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (!orgClean || langsMissing || busy) return;
    const sameName = existing.find(
      (a) => a.name.trim().toLowerCase() === preview.name.trim().toLowerCase(),
    );
    if (sameName && !dupe) {
      setDupe(sameName);
      return;
    }
    void create();
  };

  const pickFile = (f: File | undefined) => {
    if (!f) return;
    if (!DOC_OK.test(f.name)) {
      toast.error(`${f.name}: only PDF, TXT and MD files are supported.`);
      return;
    }
    setFile(f);
  };

  if (dupe) {
    return (
      <div className="space-y-4">
        <div
          role="alert"
          className="flex gap-3 rounded-lg border border-accent-warning/40 bg-accent-warning/10 p-3 text-sm text-text-primary"
        >
          <TriangleAlert
            className="mt-0.5 size-4 shrink-0 text-accent-warning"
            aria-hidden
          />
          <p>You already have an agent called {dupe.name}. Open it instead?</p>
        </div>
        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <Button variant="ghost" onClick={() => setDupe(null)} disabled={busy}>
            Go back
          </Button>
          <Button
            variant="outline"
            onClick={() => void create()}
            loading={busy}
          >
            Create another one
          </Button>
          <Button
            onClick={() => onCreated(dupe, t)}
            disabled={busy}
            data-autofocus
          >
            Open {dupe.name.length > 28 ? "it" : dupe.name}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="space-y-4" noValidate>
      <div>
        <label htmlFor="tpl-org" className={labelClass}>
          {orgLabel} <span className="text-accent-error">*</span>
        </label>
        <input
          id="tpl-org"
          className={fieldClass}
          value={org}
          onChange={(e) => setOrg(e.target.value)}
          onBlur={() => setTouched(true)}
          maxLength={80}
          placeholder={t.org_placeholder}
          aria-required="true"
          aria-invalid={orgMissing}
          aria-describedby={orgMissing ? "tpl-org-err" : undefined}
          data-autofocus
        />
        {orgMissing && (
          <p id="tpl-org-err" className="mt-1.5 text-xs text-accent-error">
            Type the name callers will hear.
          </p>
        )}
      </div>

      <div>
        <label htmlFor="tpl-city" className={labelClass}>
          City or area{" "}
          <span className="font-normal text-text-secondary">(optional)</span>
        </label>
        <input
          id="tpl-city"
          className={fieldClass}
          value={city}
          onChange={(e) => setCity(e.target.value)}
          maxLength={60}
          placeholder="e.g. Nashik"
        />
      </div>

      <div>
        <label htmlFor="tpl-details" className={labelClass}>
          Opening hours, or what callers usually ask{" "}
          <span className="font-normal text-text-secondary">(optional)</span>
        </label>
        <textarea
          id="tpl-details"
          className={`${fieldClass} h-20 resize-y`}
          value={details}
          onChange={(e) => setDetails(e.target.value)}
          maxLength={1000}
          placeholder="e.g. Open Monday to Saturday, 9 to 6. People mostly ask about fees and booking."
        />
      </div>

      <fieldset>
        <legend className={labelClass}>Languages your agent speaks</legend>
        <div className="flex flex-wrap gap-2">
          {choices.map((code) => {
            const on = langs.includes(code);
            return (
              <label
                key={code}
                className={`inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-lg border px-3 text-sm focus-within:outline-2 focus-within:outline-offset-1 focus-within:outline-accent-primary ${
                  on
                    ? "border-accent-primary bg-accent-primary/10 text-text-primary"
                    : "border-border text-text-secondary"
                }`}
              >
                <input
                  type="checkbox"
                  className="sr-only"
                  checked={on}
                  onChange={() => toggleLang(code)}
                />
                {languageName(code)}
              </label>
            );
          })}
        </div>
        {touched && langsMissing && (
          <p className="mt-1.5 text-xs text-accent-error">
            Pick at least one language.
          </p>
        )}
      </fieldset>

      <div>
        <span className={labelClass}>
          Your own document{" "}
          <span className="font-normal text-text-secondary">
            (optional, PDF, TXT or MD)
          </span>
        </span>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
          className="sr-only"
          aria-label="Choose your own document"
          onChange={(e) => {
            pickFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        {file ? (
          <div className="flex min-h-11 items-center justify-between gap-2 rounded-lg border border-border bg-bg-elevated px-3 text-sm">
            <span className="truncate">{file.name}</span>
            <button
              type="button"
              onClick={() => setFile(null)}
              className="flex size-9 items-center justify-center rounded-md text-text-secondary hover:text-text-primary"
              aria-label={`Remove ${file.name}`}
            >
              <X className="size-4" aria-hidden />
            </button>
          </div>
        ) : (
          <Button
            type="button"
            variant="outline"
            size="sm"
            leftIcon={<UploadCloud className="size-4" aria-hidden />}
            onClick={() => fileRef.current?.click()}
          >
            Upload a document now
          </Button>
        )}
        {!file && (
          <label className="mt-3 flex min-h-11 cursor-pointer items-start gap-2 text-sm text-text-secondary">
            <input
              type="checkbox"
              className="mt-1 size-4"
              checked={withSample}
              onChange={(e) => setWithSample(e.target.checked)}
            />
            <span>
              Or add later. Meanwhile, include a sample document so I can try
              the agent now. Its details are made up and it is labelled
              &quot;Sample&quot;.
            </span>
          </label>
        )}
        {file && (
          <p className="mt-1.5 text-xs text-text-secondary">
            Your document is added right after the agent is created. No sample
            is added.
          </p>
        )}
      </div>

      <p
        className="rounded-lg bg-bg-elevated p-3 text-sm text-text-secondary"
        aria-live="polite"
      >
        <span className="font-medium text-text-primary">
          Your agent will greet callers as:{" "}
        </span>
        <span>&ldquo;{preview.greeting}&rdquo;</span>
      </p>

      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button
          type="button"
          variant="ghost"
          onClick={onCancel}
          disabled={busy}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          loading={busy}
          leftIcon={
            busy ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : undefined
          }
        >
          Create agent
        </Button>
      </div>
    </form>
  );
}

/** "Start from a template": ready-made agents. Choosing one asks for the organisation's own details first. */
export function TemplateGallery({ onCreated, compact = false }: Props) {
  const templates = useAgentTemplates();
  const agents = useAgents();
  const [chosen, setChosen] = useState<TemplateInfo | null>(null);
  const [busy, setBusy] = useState(false);

  const close = () => {
    if (!busy) setChosen(null);
  };

  if (templates.error) return null; // templates are a convenience; the create form still works
  return (
    <section aria-labelledby="templates-heading" className="space-y-3">
      <div>
        <h2
          id="templates-heading"
          className={
            compact
              ? "text-xs font-medium uppercase tracking-wide text-text-muted"
              : "font-display text-lg font-semibold text-text-primary"
          }
        >
          Start from a template
        </h2>
        {!compact && (
          <p className="text-sm text-text-secondary">
            Pick a starting point, then tell us your organisation&apos;s name.
            The examples below use made-up names, and your agent gets yours.
          </p>
        )}
      </div>
      {compact ? (
        templates.isLoading ? (
          <div className="space-y-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-10 w-full rounded-lg" />
            ))}
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {((templates.data ?? []) as TemplateInfo[]).map((t) => {
              const Icon = ICONS[t.id] ?? Bot;
              return (
                <li key={t.id} className="flex items-center gap-3 py-2.5">
                  <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent-primary/10 text-accent-primary">
                    <Icon className="size-4" aria-hidden />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-text-primary">
                      {t.name}
                    </p>
                    <p className="line-clamp-1 text-xs text-text-secondary">
                      {t.audience ?? t.role}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setChosen(t)}
                    aria-label={`Use template ${t.name}`}
                  >
                    Use
                  </Button>
                </li>
              );
            })}
          </ul>
        )
      ) : templates.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-36 w-full rounded-xl" />
          ))}
        </div>
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {((templates.data ?? []) as TemplateInfo[]).map((t) => {
            const Icon = ICONS[t.id] ?? Bot;
            return (
              <li key={t.id}>
                <Card
                  variant="solid"
                  className="flex h-full flex-col gap-3 rounded-xl"
                >
                  <div className="flex items-start gap-3">
                    <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-accent-primary/10 text-accent-primary">
                      <Icon className="size-4" aria-hidden />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <h3 className="font-display text-sm font-semibold text-text-primary">
                          {t.name}
                        </h3>
                        <Badge tone="neutral">Sample</Badge>
                      </div>
                      <p className="mt-1 text-xs text-text-secondary">
                        {t.audience ?? t.role}
                      </p>
                    </div>
                  </div>
                  <p className="text-xs text-text-secondary">
                    <span className="sr-only">Languages: </span>
                    Speaks {t.languages.map(languageName).join(", ")}
                  </p>
                  {t.id === "document_qa" && t.suggested_questions?.length ? (
                    <div className="text-xs text-text-secondary">
                      <p className="font-medium text-text-primary">
                        Questions you can ask
                      </p>
                      <ul className="mt-1 list-disc space-y-0.5 pl-4">
                        {t.suggested_questions.slice(0, 4).map((q) => (
                          <li key={q}>{q}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  <div className="mt-auto flex items-center justify-end gap-2">
                    {t.sample_pdf ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => void downloadSamplePdf(t)}
                        aria-label={`Download sample PDF for ${t.name}`}
                      >
                        Sample PDF
                      </Button>
                    ) : null}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setChosen(t)}
                      aria-label={`Use template ${t.name}`}
                    >
                      Use template
                    </Button>
                  </div>
                </Card>
              </li>
            );
          })}
        </ul>
      )}
      <Modal
        open={!!chosen}
        onClose={close}
        title="Set up your agent"
        description={
          chosen
            ? `Starting from the sample "${chosen.name}". Tell us who it will answer calls for.`
            : undefined
        }
        dismissible={!busy}
      >
        {chosen && (
          <TemplateForm
            key={chosen.id}
            template={chosen}
            existing={agents.data ?? []}
            onCancel={close}
            onCreated={(a, t) => {
              setChosen(null);
              setBusy(false);
              onCreated(a, t);
            }}
            onBusy={setBusy}
            refresh={() => agents.mutate()}
          />
        )}
      </Modal>
    </section>
  );
}
