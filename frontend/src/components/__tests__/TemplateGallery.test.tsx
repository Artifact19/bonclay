import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { toast } from "sonner";
import { TemplateGallery, personaFor, type TemplateInfo } from "@/components/agents/TemplateGallery";
import { api } from "@/lib/api";
import { useAgentTemplates, useAgents } from "@/hooks/useApi";
import type { Agent } from "@/types/api";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { post: jest.fn(), upload: jest.fn() }, ApiError: class extends Error {} }));
jest.mock("@/hooks/useApi", () => ({ useAgentTemplates: jest.fn(), useAgents: jest.fn() }));

const post = api.post as jest.Mock;
const upload = api.upload as jest.Mock;

const CLINIC: TemplateInfo = {
  id: "clinic_receptionist",
  name: "Sunrise Family Clinic Receptionist",
  role: "Front-desk receptionist for a family clinic",
  audience: "For a clinic that wants calls answered about timings and fees.",
  org_label: "Clinic name",
  org_placeholder: "e.g. Sunrise Family Clinic",
  languages: ["en", "hi"],
  greeting: "Hello, thank you for calling Sunrise Family Clinic. How can I help you today?",
  document: "clinic_receptionist.md",
  suggested_questions: ["What are the timings?"],
  personalise: { name_pattern: "{org} Receptionist", sample_orgs: ["Sunrise Family Clinic"], sample_city: "Pune", strip: [] },
};

const agent = (id: string, name: string) => ({ id, name, document_count: 0 }) as unknown as Agent;
const mutate = jest.fn(async () => undefined);

function setup(existing: Agent[] = []) {
  (useAgentTemplates as jest.Mock).mockReturnValue({ data: [CLINIC], isLoading: false, error: undefined });
  (useAgents as jest.Mock).mockReturnValue({ data: existing, mutate });
  const onCreated = jest.fn();
  render(<TemplateGallery onCreated={onCreated} />);
  return onCreated;
}

const open = () => fireEvent.click(screen.getByRole("button", { name: /use template/i }));
const type = (label: RegExp, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

beforeEach(() => {
  jest.clearAllMocks();
  post.mockImplementation(async (_p: string, body: { org_name: string }) => agent("new1", `${body.org_name} Receptionist`));
});

describe("template cards", () => {
  it("show a Sample tag, who it is for and the language names", () => {
    setup();
    expect(screen.getByText("Sample")).toBeTruthy();
    expect(screen.getByText(/calls answered about timings and fees/)).toBeTruthy();
    expect(screen.getByText(/Speaks English, Hindi/)).toBeTruthy();
    expect(screen.queryByText("en")).toBeNull(); // no raw codes
  });

  it("do not create anything on click, they open the form", () => {
    setup();
    open();
    expect(screen.getByRole("dialog", { name: "Set up your agent" })).toBeTruthy();
    expect(post).not.toHaveBeenCalled();
  });
});

describe("the form", () => {
  it("pre-fills languages by name and previews the greeting with the typed name", () => {
    setup();
    open();
    expect((screen.getByRole("checkbox", { name: "English" }) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole("checkbox", { name: "Hindi" }) as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole("checkbox", { name: "Tamil" }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText(/Your agent will greet callers as:/).parentElement?.textContent).toContain("[Clinic name]");
    type(/Clinic name/, "Lotus Care");
    const line = screen.getByText(/Your agent will greet callers as:/).parentElement?.textContent ?? "";
    expect(line).toContain("thank you for calling Lotus Care.");
    expect(line).not.toContain("Sunrise");
  });

  it("needs an organisation name and does not call the API without one", () => {
    setup();
    open();
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    expect(screen.getByText("Type the name callers will hear.")).toBeTruthy();
    expect(post).not.toHaveBeenCalled();
  });

  it("sends the overrides, then calls onCreated with the new agent and the template", async () => {
    const onCreated = setup();
    open();
    type(/Clinic name/, "  Lotus Care ");
    type(/City or area/, "Nashik");
    type(/Opening hours/, "9 to 6");
    fireEvent.click(screen.getByRole("checkbox", { name: "Tamil" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Hindi" }));
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));
    expect(post).toHaveBeenCalledWith("/api/agents/from-template", {
      template_id: "clinic_receptionist",
      org_name: "Lotus Care",
      city: "Nashik",
      details: "9 to 6",
      languages: ["en", "ta"],
      include_sample: true,
    });
    expect(mutate).toHaveBeenCalled();
    expect(onCreated.mock.calls[0][0].id).toBe("new1");
    expect(onCreated.mock.calls[0][1]).toBe(CLINIC);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("uploads the user's document after creating and then leaves the sample out", async () => {
    upload.mockResolvedValue({});
    const onCreated = setup();
    open();
    type(/Clinic name/, "Lotus Care");
    const file = new File(["hours"], "fees.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByLabelText("Choose your own document"), { target: { files: [file] } });
    expect(screen.getByText("fees.pdf")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(post.mock.calls[0][1].include_sample).toBe(false);
    const fd = upload.mock.calls[0][1] as FormData;
    expect(upload.mock.calls[0][0]).toBe("/api/knowledge/upload");
    expect(fd.get("agent_id")).toBe("new1");
    expect((fd.get("file") as File).name).toBe("fees.pdf");
  });

  it("still lands on the agent if the upload fails, and says so", async () => {
    upload.mockRejectedValue(new Error("too big"));
    const onCreated = setup();
    open();
    type(/Clinic name/, "Lotus Care");
    fireEvent.change(screen.getByLabelText("Choose your own document"), {
      target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(toast.error).toHaveBeenCalled();
  });

  it("can leave the sample document out", async () => {
    const onCreated = setup();
    open();
    type(/Clinic name/, "Lotus Care");
    fireEvent.click(screen.getByRole("checkbox", { name: /include a sample document/i }));
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(post.mock.calls[0][1].include_sample).toBe(false);
  });

  it("stays open and reports a failure", async () => {
    post.mockRejectedValue(new Error("boom"));
    const onCreated = setup();
    open();
    type(/Clinic name/, "Lotus Care");
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect(onCreated).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog")).toBeTruthy();
  });
});

describe("duplicates", () => {
  it("warns before creating a second agent with the same name, and can open the existing one", async () => {
    const existing = agent("old1", "Lotus Care Receptionist");
    const onCreated = setup([existing]);
    open();
    type(/Clinic name/, "lotus care");
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    const alert = screen.getByRole("alert");
    expect(alert.textContent).toBe("You already have an agent called Lotus Care Receptionist. Open it instead?");
    expect(post).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: /^Open/ }));
    expect(onCreated).toHaveBeenCalledWith(existing, CLINIC);
    expect(post).not.toHaveBeenCalled();
  });

  it("creates anyway when asked to", async () => {
    const onCreated = setup([agent("old1", "Lotus Care Receptionist")]);
    open();
    type(/Clinic name/, "Lotus Care");
    fireEvent.click(screen.getByRole("button", { name: "Create agent" }));
    fireEvent.click(screen.getByRole("button", { name: "Create another one" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(post).toHaveBeenCalledTimes(1);
  });
});

describe("personaFor", () => {
  it("swaps sample names in one pass and applies the name pattern", () => {
    const hod: TemplateInfo = {
      ...CLINIC,
      name: "HOD Office Assistant: Prof. Jayashri Prasad",
      greeting: "this is the office of Professor Jayashri Prasad, Head",
      personalise: {
        name_pattern: "HOD Office Assistant: {org}",
        sample_orgs: ["Professor Jayashri Prasad", "Prof. Jayashri Prasad", "Jayashri Prasad"],
        strip: [],
      },
    };
    expect(personaFor(hod, "Prof. Jayashri Rao")).toEqual({
      name: "HOD Office Assistant: Prof. Jayashri Rao",
      greeting: "this is the office of Prof. Jayashri Rao, Head",
    });
  });
  it("leaves a template without rules alone", () => {
    expect(personaFor({ ...CLINIC, personalise: undefined }, "X").name).toBe(CLINIC.name);
  });
});
