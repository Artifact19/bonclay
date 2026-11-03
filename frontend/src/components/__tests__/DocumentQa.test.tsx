import { render, screen } from "@testing-library/react";
import { TemplateGallery, type TemplateInfo } from "@/components/agents/TemplateGallery";
import { KnowledgePanel } from "@/components/agents/KnowledgePanel";
import { useAgentDocuments, useAgentTemplates, useAgents, useExampleQuestions } from "@/hooks/useApi";

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn() } }));
jest.mock("@/lib/api", () => ({ api: { post: jest.fn(), upload: jest.fn(), blob: jest.fn() }, ApiError: class extends Error {} }));
jest.mock("@/hooks/useApi", () => ({
  useAgentTemplates: jest.fn(),
  useAgents: jest.fn(),
  useAgentDocuments: jest.fn(),
  useExampleQuestions: jest.fn(),
}));

const QA: TemplateInfo = {
  id: "document_qa",
  name: "Document Q&A Assistant",
  role: "Answers from your PDF",
  audience: "For anyone who wants to ask questions of a PDF.",
  languages: ["en"],
  greeting: "Hello",
  document: "document_qa.md",
  sample_pdf: "/api/agent-templates/document_qa/sample.pdf",
  suggested_questions: ["How much is a student membership?", "What is the late fee for a book?"],
};

test("the Q&A template card lists questions to ask and offers the sample PDF", () => {
  (useAgentTemplates as jest.Mock).mockReturnValue({ data: [QA], isLoading: false, error: undefined });
  (useAgents as jest.Mock).mockReturnValue({ data: [], mutate: jest.fn() });
  render(<TemplateGallery onCreated={jest.fn()} />);
  expect(screen.getByText("Questions you can ask")).toBeInTheDocument();
  expect(screen.getByText("How much is a student membership?")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /download sample pdf/i })).toBeInTheDocument();
});

test("the documents panel shows questions written from the documents", () => {
  (useAgentDocuments as jest.Mock).mockReturnValue({
    data: [{ id: "d1", filename: "handbook.pdf", chunk_count: 4, created_at: new Date().toISOString() }],
    isLoading: false,
    mutate: jest.fn(),
  });
  (useAgents as jest.Mock).mockReturnValue({ data: [], mutate: jest.fn() });
  (useExampleQuestions as jest.Mock).mockReturnValue({ data: ["When is the library open on Sunday?"] });
  render(<KnowledgePanel agentId="a1" />);
  expect(useExampleQuestions).toHaveBeenCalledWith("a1", 1);
  expect(screen.getByText("Questions you can ask")).toBeInTheDocument();
  expect(screen.getByText("When is the library open on Sunday?")).toBeInTheDocument();
});
