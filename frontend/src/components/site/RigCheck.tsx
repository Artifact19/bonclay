"use client";

import { useState } from "react";
import { ArrowUpRight, ChevronDown } from "lucide-react";
import {
  CAPABILITIES,
  CHANNELS_NOTE,
  GLOSSARY,
  LEGEND,
  LEVEL_SYMBOL,
  LEVEL_WORD,
  MACHINES,
  MACHINE_GROUPS,
  MINIMUM_LINE,
  OPTIMIZATION_URL,
  README_SETUP_URL,
  RIG_SECTION_CAVEAT,
  SOURCE_HINT,
  SOURCE_LABEL,
  TRIAL_REPORT_URL,
  type Figure,
  type Level,
  type Machine,
  type Source,
  type Tier,
} from "./rigCheckData";
import { REPO_PUBLIC } from "./repo";

/* ------------------------------------------------------------------ small pieces */

const TIER_STYLE: Record<Tier, string> = {
  "Full studio": "bg-[var(--s-marigold)] text-[var(--s-marigold-ink)] border-transparent",
  "Runs well": "bg-[var(--s-ink)] text-white border-transparent",
  "Runs with Groq": "bg-[var(--s-paper)] text-[var(--s-ink)] border-[var(--s-ink)]",
  Basic: "bg-transparent text-[var(--s-ink-2)] border-dashed border-[var(--s-mute)]",
};

const SOURCE_STYLE: Record<Source, string> = {
  measured: "bg-[var(--s-ink)] text-white border-transparent",
  estimated: "bg-transparent text-[#8A5300] border-[#C98A2B]",
  untested: "bg-transparent text-[var(--s-mute)] border-dashed border-[var(--s-mute)]",
};

const LEVEL_STYLE: Record<Level, string> = {
  yes: "text-[var(--s-ink)] font-semibold",
  partial: "text-[#8A5300] text-[1.35em] font-bold leading-none",
  no: "text-[var(--s-mute)]",
};

function SourceTag({ source }: { source: Source }) {
  return (
    <span
      data-source={source}
      title={SOURCE_HINT[source]}
      className={`inline-flex shrink-0 items-center rounded-full border px-2 py-[1px] s-label font-medium leading-[1.5] ${SOURCE_STYLE[source]}`}
    >
      {SOURCE_LABEL[source]}
    </span>
  );
}

function LevelMark({ level }: { level: Level }) {
  return (
    <>
      <span aria-hidden className={LEVEL_STYLE[level]}>
        {LEVEL_SYMBOL[level]}
      </span>
      <span className="sr-only">{LEVEL_WORD[level]}</span>
    </>
  );
}

function FigureList({ figures }: { figures: Figure[] }) {
  return (
    <dl className="divide-y divide-[var(--s-line)] border-y border-[var(--s-line)]">
      {figures.map((f) => (
        <div key={f.label} className="grid gap-x-4 gap-y-1 py-3 sm:grid-cols-[minmax(0,13rem)_minmax(0,1fr)]">
          <dt className="s-small text-[var(--s-mute)]">{f.label}</dt>
          <dd className="min-w-0">
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1 s-body font-medium text-[var(--s-ink)]">
              <span>{f.value}</span>
              <SourceTag source={f.source} />
            </p>
            {f.note && <p className="mt-1 s-small leading-snug text-[var(--s-mute)]">{f.note}</p>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------ picker */

function MachinePicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  return (
    <fieldset className="min-w-0">
      <legend className="s-label font-medium text-[var(--s-mute)]">Your machine</legend>
      <div className="mt-3 space-y-4">
        {MACHINE_GROUPS.map((group) => (
          <div key={group}>
            <p className="mb-2 s-small font-medium text-[var(--s-ink)]">{group}</p>
            <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
              {MACHINES.filter((m) => m.group === group).map((m) => (
                <li key={m.id}>
                  <label
                    htmlFor={`rig-${m.id}`}
                    className="flex min-h-[3rem] cursor-pointer items-start gap-3 rounded-[var(--s-r-inner)] border border-[var(--s-line)] bg-[var(--s-paper)] px-4 py-2.5 transition-colors hover:border-[var(--s-bar)] has-[:checked]:border-[var(--s-ink)] has-[:checked]:shadow-[inset_0_0_0_1px_var(--s-ink)] has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--s-ink)]"
                  >
                    <input
                      id={`rig-${m.id}`}
                      type="radio"
                      name="rig-machine"
                      value={m.id}
                      checked={value === m.id}
                      onChange={() => onChange(m.id)}
                      className="peer sr-only"
                    />
                    <span
                      aria-hidden
                      className="mt-[3px] grid h-[1.125rem] w-[1.125rem] shrink-0 place-items-center rounded-full border border-[var(--s-mute)] peer-checked:border-[var(--s-ink)] peer-checked:[&>span]:scale-100"
                    >
                      <span className="h-[0.625rem] w-[0.625rem] scale-0 rounded-full bg-[var(--s-ink)] transition-transform duration-150" />
                    </span>
                    <span className="min-w-0">
                      <span className="block s-body font-medium leading-snug text-[var(--s-ink)]">{m.label}</span>
                      <span className="block s-small leading-snug text-[var(--s-mute)]">{m.hint}</span>
                    </span>
                  </label>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </fieldset>
  );
}

function GroqToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label
      htmlFor="rig-groq"
      className="mt-5 flex min-h-[3rem] cursor-pointer items-start gap-3 rounded-[var(--s-r-inner)] border border-[var(--s-line)] bg-[var(--s-paper)] px-4 py-3 has-[:checked]:border-[var(--s-ink)] has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--s-ink)]"
    >
      <input id="rig-groq" type="checkbox" role="switch" checked={on} onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
      <span
        aria-hidden
        className="relative mt-[2px] h-[1.375rem] w-[2.375rem] shrink-0 rounded-full bg-[var(--s-bar)] transition-colors duration-200 peer-checked:bg-[var(--s-ink)] after:absolute after:left-[3px] after:top-[3px] after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-transform after:duration-200 peer-checked:after:translate-x-4"
      />
      <span className="min-w-0">
        <span className="block s-body font-medium leading-snug text-[var(--s-ink)]">With a free Groq key (cloud speech + thinking)</span>
        <span className="block s-small leading-snug text-[var(--s-mute)]">Groq listens and thinks online. Voices still run on your machine.</span>
      </span>
    </label>
  );
}

/* ------------------------------------------------------------------ short summary (always visible) */

const BUCKETS: { level: Level; title: string; empty: string }[] = [
  { level: "yes", title: "Works well", empty: "Nothing on this machine." },
  { level: "partial", title: "Slow or needs extra setup", empty: "Nothing." },
  { level: "no", title: "Won't fit", empty: "Nothing: it all fits." },
];

function ResultSummary({ machine, groq }: { machine: Machine; groq: boolean }) {
  const profile = groq ? machine.groq : machine.local;
  const mode = groq ? "with a Groq key" : "local models only";
  return (
    <article className="min-w-0 rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)]" data-testid="rig-result" aria-label={`Result for ${machine.label}, ${mode}`}>
      <div aria-live="polite" aria-atomic="true" className="p-5 sm:p-7" data-testid="rig-live">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <span
            data-testid="rig-tier"
            className={`inline-flex items-center rounded-full border px-4 py-1.5 s-body font-medium ${TIER_STYLE[profile.tier]}`}
          >
            {profile.tier}
          </span>
          <p className="s-body text-[var(--s-ink-2)]">
            {machine.label}, {mode}
          </p>
        </div>
        <p className="mt-3 max-w-[62ch] s-body leading-relaxed text-[var(--s-ink)]">{profile.summary}</p>

        <p className="mt-5 s-label font-medium text-[var(--s-mute)]">On your machine</p>
        <div className="mt-2 grid gap-3 sm:grid-cols-3" data-testid="rig-buckets">
          {BUCKETS.map((b) => {
            const caps = CAPABILITIES.filter((cap) => profile.caps[cap.id].level === b.level);
            return (
              <section key={b.level} data-bucket={b.level} aria-label={b.title} className="rounded-[var(--s-r-inner)] bg-[var(--s-ground)] p-4">
                <h3 className="flex items-center gap-2 s-body font-medium text-[var(--s-ink)]">
                  <span aria-hidden className={`grid h-6 w-6 place-items-center rounded-full border border-[var(--s-line)] bg-[var(--s-paper)] s-small leading-none ${LEVEL_STYLE[b.level]}`}>
                    {LEVEL_SYMBOL[b.level]}
                  </span>
                  {b.title}
                </h3>
                {caps.length ? (
                  <ul className="mt-2 space-y-1 s-small leading-snug text-[var(--s-ink-2)]">
                    {caps.map((cap) => (
                      <li key={cap.id} data-cap={cap.id}>
                        {cap.label}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-2 s-small text-[var(--s-mute)]">{b.empty}</p>
                )}
              </section>
            );
          })}
        </div>

        <p className="mt-4 flex flex-wrap items-start gap-x-2 gap-y-1 s-small leading-snug text-[var(--s-mute)]">
          <SourceTag source={machine.evidence} />
          <span className="min-w-0 flex-1 basis-[16rem]">{machine.evidenceNote}</span>
        </p>
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ every feature (behind the disclosure) */

function FeatureDetails({ machine, groq }: { machine: Machine; groq: boolean }) {
  const profile = groq ? machine.groq : machine.local;
  const mode = groq ? "with a Groq key" : "local models only";
  return (
    <div className="border-t border-[var(--s-line)] p-5 sm:p-8">
      <p className="s-body text-[var(--s-ink-2)]">
        Details for {machine.label}, {mode}.
      </p>
      <div className="mt-6 grid gap-x-10 gap-y-8 xl:grid-cols-2">
        <section aria-labelledby="rig-works">
          <h3 id="rig-works" className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">
            What works
          </h3>
          <ul className="mt-3 divide-y divide-[var(--s-line)] border-y border-[var(--s-line)]">
            {CAPABILITIES.map((cap) => {
              const cell = profile.caps[cap.id];
              return (
                <li key={cap.id} className="flex items-start gap-3 py-2.5" data-cap={cap.id} data-level={cell.level}>
                  <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-[var(--s-line)] s-small leading-none">
                    <LevelMark level={cell.level} />
                  </span>
                  <span className="min-w-0">
                    <span className="block s-body text-[var(--s-ink)]">{cap.label}</span>
                    {cell.note && <span className="block s-small leading-snug text-[var(--s-mute)]">{cell.note}</span>}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>

        <section aria-labelledby="rig-advice">
          <h3 id="rig-advice" className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">
            Switch off, or use the cloud for
          </h3>
          <ul className="mt-3 space-y-3 s-body leading-relaxed text-[var(--s-ink-2)]" data-testid="rig-advice">
            {profile.advice.map((a) => (
              <li key={a} className="flex gap-3">
                <span aria-hidden className="mt-[0.7em] h-1 w-1 shrink-0 rounded-full bg-[var(--s-ink)]" />
                <span className="min-w-0">{a}</span>
              </li>
            ))}
          </ul>
          <p className="mt-5 rounded-[var(--s-r-inner)] bg-[var(--s-ground)] p-4 s-small leading-relaxed text-[var(--s-ink-2)]" data-testid="rig-channels">
            {CHANNELS_NOTE}
          </p>
        </section>

        <section aria-labelledby="rig-speed" className="xl:col-span-2">
          <h3 id="rig-speed" className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">
            Expected speed and memory
          </h3>
          <div className="mt-3">
            <FigureList figures={profile.speed} />
          </div>
        </section>

        <section aria-labelledby="rig-disk" className="xl:col-span-2">
          <h3 id="rig-disk" className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">
            Disk space
          </h3>
          <div className="mt-3">
            <FigureList figures={machine.footprint} />
          </div>
        </section>
      </div>

      <CompareTable selected={machine.id} groq={groq} />

      <section aria-labelledby="rig-glossary" className="mt-10">
        <h3 id="rig-glossary" className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">
          Words used here, in plain language
        </h3>
        <dl className="mt-3 grid gap-x-10 border-b border-[var(--s-line)] md:grid-cols-2" data-testid="rig-glossary">
          {GLOSSARY.map((g) => (
            <div key={g.term} className="border-t border-[var(--s-line)] py-3">
              <dt className="s-body font-medium text-[var(--s-ink)]">{g.term}</dt>
              <dd className="mt-0.5 s-small leading-snug text-[var(--s-ink-2)]">{g.text}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ comparison table */

function CompareTable({ selected, groq }: { selected: string; groq: boolean }) {
  return (
    <div className="mt-10" data-testid="rig-compare">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h3 className="s-item font-medium tracking-[-0.01em] text-[var(--s-ink)]">All machines side by side</h3>
        <p className="s-small text-[var(--s-mute)]">Showing {groq ? "with a free Groq key" : "local models only"}. Your pick is outlined.</p>
      </div>
      <div
        className="relative mt-4 overflow-x-auto rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)]"
        role="region"
        aria-label="Comparison of machines and capabilities"
        tabIndex={0}
      >
        <table className="w-full min-w-[34rem] border-collapse text-left" data-testid="rig-table">
          <caption className="sr-only">
            Which capabilities each machine supports, {groq ? "with a free Groq key" : "with local models only"}.
          </caption>
          <thead>
            <tr className="border-b border-[var(--s-line)] s-label font-medium text-[var(--s-mute)]">
              <th scope="col" className="sticky left-0 z-10 bg-[var(--s-paper)] px-4 py-3 font-medium">
                Machine
              </th>
              {CAPABILITIES.map((cap) => (
                <th key={cap.id} scope="col" className="px-2 py-3 text-center align-bottom font-medium leading-tight">
                  {cap.short}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MACHINES.map((m) => {
              const profile = groq ? m.groq : m.local;
              const on = m.id === selected;
              return (
                <tr key={m.id} data-selected={on} className={`border-b border-[var(--s-line)] last:border-b-0 ${on ? "bg-[var(--s-ground)]" : ""}`}>
                  <th
                    scope="row"
                    className={`sticky left-0 z-10 px-4 py-3 s-small font-medium text-[var(--s-ink)] ${on ? "bg-[var(--s-ground)] shadow-[inset_3px_0_0_var(--s-ink)]" : "bg-[var(--s-paper)]"}`}
                  >
                    <span className="block">{m.tableName}</span>
                    <span className="mt-0.5 flex items-center gap-1.5">
                      <SourceTag source={m.evidence} />
                    </span>
                    <span className="sr-only">Tier: {profile.tier}</span>
                  </th>
                  {CAPABILITIES.map((cap) => (
                    <td key={cap.id} className="px-2 py-3 text-center s-body">
                      <LevelMark level={profile.caps[cap.id].level} />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-2 s-small text-[var(--s-ink-2)]" aria-label="Legend">
        {LEGEND.map((l) => (
          <li key={l.level} className="flex items-baseline gap-2">
            <span aria-hidden className={`s-body ${LEVEL_STYLE[l.level]}`}>
              {LEVEL_SYMBOL[l.level]}
            </span>
            <span>{l.text}</span>
          </li>
        ))}
      </ul>
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-2 s-small text-[var(--s-ink-2)]" aria-label="Evidence tags">
        {(Object.keys(SOURCE_LABEL) as Source[]).map((s) => (
          <li key={s} className="flex items-center gap-2">
            <SourceTag source={s} />
            <span>{SOURCE_HINT[s].replace(/\.$/, "").toLowerCase()}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ section */

/** `asPage`: the component is the main content of its own page (/run-it), which already shows the page heading and an
 *  intro line, so the section's own kicker, h2 and lede are left out and only the caveat that belongs to the picker stays. */
export function RigCheck({ asPage = false }: { asPage?: boolean } = {}) {
  const [machineId, setMachineId] = useState("mac16");
  const [groq, setGroq] = useState(false);
  const machine = MACHINES.find((m) => m.id === machineId) ?? MACHINES[0];

  return (
    <section
      id="rig-check"
      className={`s-container ${asPage ? "mt-12 md:mt-16" : "mt-24 md:mt-32"}`}
      {...(asPage ? { "aria-label": "Rig Check" } : { "aria-labelledby": "rig-check-title" })}
    >
      {asPage ? (
        <p data-reveal className="max-w-[60ch] s-body leading-snug text-[var(--s-mute)]">
          {RIG_SECTION_CAVEAT}
        </p>
      ) : (
        <div data-reveal>
          <h2 id="rig-check-title" className="h2 max-w-[20ch]">
            Will BonClay run on your laptop?
          </h2>
          <p className="lede mt-4">
            Pick your machine. We say what works well, what is slow and what will not fit, and how sure we are. {RIG_SECTION_CAVEAT}
          </p>
        </div>
      )}

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:items-start lg:gap-10">
        <div className="min-w-0" data-reveal>
          <MachinePicker value={machineId} onChange={setMachineId} />
          <GroqToggle on={groq} onChange={setGroq} />
        </div>
        <div className="min-w-0" data-reveal>
          <ResultSummary machine={machine} groq={groq} />
        </div>
      </div>

      <details
        className="group mt-6 overflow-hidden rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)]"
        data-testid="rig-details"
        data-reveal
      >
        <summary className="flex min-h-[3.5rem] cursor-pointer list-none items-center justify-between gap-4 px-5 py-3 s-body font-medium text-[var(--s-ink)] focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[var(--s-ink)] sm:px-8 [&::-webkit-details-marker]:hidden">
          <span className="min-w-0">
            See every feature
            <span className="block s-small font-normal leading-snug text-[var(--s-mute)]">
              Notes per feature, speed, disk space, all machines side by side, and a plain-language glossary
            </span>
          </span>
          <ChevronDown className="h-5 w-5 shrink-0 transition-transform duration-200 group-open:rotate-180 motion-reduce:transition-none" aria-hidden />
        </summary>
        <FeatureDetails machine={machine} groq={groq} />
      </details>

      <div className="mt-6 rounded-[var(--s-r-card)] bg-[var(--s-marigold)] p-6 text-[var(--s-marigold-ink)] sm:p-8" data-reveal>
        <p className="s-lede font-medium leading-snug text-[var(--s-marigold-ink)]" data-testid="rig-minimum">
          {MINIMUM_LINE}
        </p>
        <p className="mt-3 flex flex-wrap gap-x-6 gap-y-0 s-body">
          {REPO_PUBLIC ? (
            <>
              <a href={README_SETUP_URL} className="inline-flex min-h-[2.75rem] items-center gap-1 underline underline-offset-4">
                Setup steps in the README <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
              </a>
              <a href={OPTIMIZATION_URL} className="inline-flex min-h-[2.75rem] items-center gap-1 underline underline-offset-4">
                docs/OPTIMIZATION_RESEARCH.md <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
              </a>
              <a href={TRIAL_REPORT_URL} className="inline-flex min-h-[2.75rem] items-center gap-1 underline underline-offset-4">
                docs/TRIAL_REPORT.md (the measured numbers) <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
              </a>
            </>
          ) : (
            <span className="inline-flex min-h-[2.75rem] items-center">Setup steps, the optimisation research and the trial report with the measured numbers are in the project&apos;s docs, shared on request.</span>
          )}
        </p>
      </div>
    </section>
  );
}
