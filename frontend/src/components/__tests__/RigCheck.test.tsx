import { fireEvent, render, screen, within } from "@testing-library/react";
import { RigCheck } from "@/components/site/RigCheck";
import { CAPABILITIES, MACHINES, MINIMUM_LINE, OPTIMIZATION_URL, README_SETUP_URL } from "@/components/site/rigCheckData";

const live = () => screen.getByTestId("rig-live");
const tier = () => screen.getByTestId("rig-tier").textContent;
const groqSwitch = () => screen.getByRole("switch", { name: /free Groq key/i });

describe("Rig Check section", () => {
  it("renders the heading, the picker, the toggle, a result and the table", () => {
    render(<RigCheck />);
    expect(screen.getByRole("heading", { level: 2, name: "Will BonClay run on your laptop?" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Your machine" })).toBeTruthy();
    expect(screen.getAllByRole("radio")).toHaveLength(MACHINES.length);
    expect(groqSwitch()).toBeTruthy();
    expect(screen.getByTestId("rig-result")).toBeTruthy();
    expect(screen.getByTestId("rig-table")).toBeTruthy();
    expect(screen.getByTestId("rig-minimum").textContent).toBe(MINIMUM_LINE);
  });

  it("does not link to the private repository: the docs are said to be shared on request", () => {
    render(<RigCheck />);
    expect(screen.queryByRole("link", { name: /Setup steps in the README/ })).toBeNull();
    expect(screen.queryByRole("link", { name: /OPTIMIZATION_RESEARCH/ })).toBeNull();
    expect(screen.getByText(/in the project's docs, shared on request/)).toBeInTheDocument();
    // the addresses stay on record for when the repository is public (see components/site/repo.ts)
    expect(README_SETUP_URL).toContain("github.com/Artifact19/bonclay/blob/main/README.md");
    expect(OPTIMIZATION_URL).toContain("github.com/Artifact19/bonclay/blob/main/docs/OPTIMIZATION_RESEARCH.md");
  });

  it("announces the result politely and updates it when each machine is selected", () => {
    render(<RigCheck />);
    expect(live().getAttribute("aria-live")).toBe("polite");
    for (const m of MACHINES) {
      fireEvent.click(screen.getByRole("radio", { name: new RegExp(m.label.replace(/[()]/g, "\\$&"), "i") }));
      expect(live().textContent).toContain(m.label);
      expect(live().textContent).toContain(m.local.summary);
      expect(tier()).toBe(m.local.tier);
    }
  });

  it("shows the measured M4 numbers for the 16 GB Mac, tagged measured", () => {
    render(<RigCheck />);
    const card = screen.getByTestId("rig-details");
    for (const text of ["3.1 s", "1.0 s", "2.4 s", "about 35 s per sentence", "Your voice in Indian languages", "about 5 to 7 s per sentence", "10 to 12 GB"]) {
      expect(within(card).getAllByText(text, { exact: false }).length).toBeGreaterThan(0);
    }
    expect(card.querySelectorAll('[data-source="measured"]').length).toBeGreaterThan(3);
  });

  it("changes the recommendations when the Groq toggle is flipped", () => {
    render(<RigCheck />);
    fireEvent.click(screen.getByRole("radio", { name: /NVIDIA GPU, 4 GB/i }));
    const before = screen.getByTestId("rig-advice").textContent;
    expect(tier()).toBe("Basic");
    expect(groqSwitch()).toHaveProperty("checked", false);

    fireEvent.click(groqSwitch());
    expect(groqSwitch()).toHaveProperty("checked", true);
    expect(tier()).toBe("Runs with Groq");
    expect(live().textContent).toContain("with a Groq key");
    expect(screen.getByTestId("rig-advice").textContent).not.toBe(before);

    fireEvent.click(screen.getByRole("radio", { name: /Apple Silicon Mac, 16 GB/i }));
    fireEvent.click(groqSwitch());
    expect(live().textContent).toContain("local models only");
    fireEvent.click(groqSwitch());
    expect(screen.getAllByText("2.8 s", { exact: false }).length).toBeGreaterThan(0); // Groq-answered reply, measured
  });

  it("is honest about hardware it has not run: 4 GB NVIDIA is untested and estimated", () => {
    render(<RigCheck />);
    fireEvent.click(screen.getByRole("radio", { name: /NVIDIA GPU, 4 GB/i }));
    const card = screen.getByTestId("rig-details");
    expect(screen.getByTestId("rig-result").querySelectorAll('[data-source="untested"]').length).toBeGreaterThan(0);
    expect(card.querySelectorAll('[data-source="untested"]').length).toBeGreaterThan(0);
    expect(card.querySelectorAll('[data-source="estimated"]').length).toBeGreaterThan(0);
    const speed = screen.getByRole("region", { name: /Expected speed/ });
    expect(speed.querySelectorAll('[data-source="measured"]').length).toBe(0); // no Mac timings leak onto the PC card
    expect(within(screen.getByTestId("rig-result")).getByText(/Windows setup script|bonclay\.ps1/, { exact: false })).toBeTruthy();
    expect(within(card).getByText(/Use a free Groq key for thinking/)).toBeTruthy();
  });

  it("tags every figure with measured, estimated or untested, and explains the tags", () => {
    render(<RigCheck />);
    expect(screen.getByLabelText("Evidence tags").textContent).toMatch(/Measured.*Estimated.*Untested/);
    for (const m of MACHINES) {
      for (const p of [m.local, m.groq]) {
        for (const f of [...p.speed, ...m.footprint]) expect(["measured", "estimated", "untested"]).toContain(f.source);
      }
    }
  });

  it("renders a comparison table with every machine, every capability and a legend", () => {
    render(<RigCheck />);
    const table = screen.getByTestId("rig-table");
    expect(within(table).getAllByRole("columnheader")).toHaveLength(CAPABILITIES.length + 1);
    expect(within(table).getAllByRole("row")).toHaveLength(MACHINES.length + 1);
    for (const m of MACHINES) expect(within(table).getByRole("rowheader", { name: new RegExp(m.tableName) })).toBeTruthy();
    const legend = screen.getByLabelText("Legend");
    expect(legend.textContent).toMatch(/✓/);
    expect(legend.textContent).toMatch(/~/);
    expect(legend.textContent).toMatch(/✗/);
    // Symbols are paired with words for screen readers, so colour is never the only signal.
    expect(within(table).getAllByText("Works with limits").length).toBeGreaterThan(0);
  });

  it("makes the table follow the Groq toggle and outlines the picked machine", () => {
    render(<RigCheck />);
    const rows = () => within(screen.getByTestId("rig-table")).getAllByRole("row").slice(1);
    const cpuRow = () => rows()[MACHINES.findIndex((m) => m.id === "cpu")];
    const liveCell = () => within(cpuRow()).getAllByRole("cell")[0].textContent;
    expect(liveCell()).toContain("Not practical"); // CPU only, no Groq: nothing to think with
    fireEvent.click(groqSwitch());
    expect(liveCell()).toContain("Works");
    expect(liveCell()).not.toContain("Not practical");
    fireEvent.click(screen.getByRole("radio", { name: /Any laptop, no GPU/i }));
    expect(cpuRow().getAttribute("data-selected")).toBe("true");
  });
});

describe("Rig Check compact layout", () => {
  it("is a section with an id, labelled by its heading, and marks its blocks for the shared reveal", () => {
    const { container } = render(<RigCheck />);
    const section = container.querySelector("section#rig-check");
    expect(section?.getAttribute("aria-labelledby")).toBe("rig-check-title");
    expect(container.querySelectorAll("#rig-check [data-reveal]").length).toBeGreaterThanOrEqual(4);
  });

  it("keeps the full capability table behind a closed 'See every feature' disclosure", () => {
    render(<RigCheck />);
    const details = screen.getByTestId("rig-details") as HTMLDetailsElement;
    expect(details.tagName).toBe("DETAILS");
    expect(details.open).toBe(false);
    expect(within(details).getByText("See every feature")).toBeTruthy();
    expect(details.contains(screen.getByTestId("rig-table"))).toBe(true);
    expect(screen.getByTestId("rig-result").contains(screen.getByTestId("rig-table"))).toBe(false);
  });

  it("summarises what works well, what is slow and what will not fit, inside the live region", () => {
    render(<RigCheck />);
    const buckets = within(screen.getByTestId("rig-live")).getByTestId("rig-buckets");
    const names = (level: string) => Array.from(buckets.querySelectorAll(`[data-bucket="${level}"] [data-cap]`)).map((e) => e.getAttribute("data-cap"));
    expect(names("yes")).toEqual(["live", "clone", "hindiClone", "offline", "handsFree"]); // Mac 16 GB, local models only
    expect(names("partial")).toEqual(["indic"]);
    expect(names("no")).toEqual([]);
    fireEvent.click(screen.getByRole("radio", { name: /Any laptop, no GPU/i }));
    expect(names("no")).toEqual(["live", "indic", "offline", "handsFree"]);
    expect(within(buckets).getByText("Won't fit")).toBeTruthy();
  });

  it("covers phone calls, meetings, DhVaani and plain-word explanations", () => {
    render(<RigCheck />);
    const details = screen.getByTestId("rig-details");
    expect(within(details).getByTestId("rig-channels").textContent).toMatch(/Phone calls.*Exotel.*Google Calendar/);
    const glossary = within(details).getByTestId("rig-glossary").textContent ?? "";
    for (const term of ["Groq", "Kokoro", "DhVaani", "Indic-Speak", "VRAM", "Swapping"]) expect(glossary).toContain(term);
    expect(glossary).toContain("ARTPARK-IISc");
  });

  it("gives every link a 44 px target", () => {
    render(<RigCheck />);
    for (const a of screen.queryAllByRole("link")) expect(a.className).toContain("min-h-[2.75rem]");
  });
});

describe("Rig Check data", () => {
  it("covers every machine in the brief and gives each a result with and without Groq", () => {
    expect(MACHINES.map((m) => m.id)).toEqual(["mac8", "mac16", "mac24", "nv4", "nv8", "nv12", "cpu"]);
    for (const m of MACHINES) {
      for (const p of [m.local, m.groq]) {
        expect(Object.keys(p.caps).sort()).toEqual(CAPABILITIES.map((c) => c.id).sort());
        expect(p.advice.length).toBeGreaterThan(0);
        expect(p.speed.length).toBeGreaterThan(0);
      }
    }
  });

  it("only the Apple M4 16 GB profile carries measured speed figures", () => {
    for (const m of MACHINES) {
      const measured = [...m.local.speed, ...m.groq.speed].some((f) => f.source === "measured" && /s\b/.test(f.value) && !/GB/.test(f.value));
      if (m.id !== "mac16") expect(measured).toBe(false);
    }
  });
});
