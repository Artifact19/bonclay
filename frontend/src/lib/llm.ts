/** Shown wherever no language model is configured (health badge, Agent Studio, Playground). */
export const LLM_FIX_TEXT = "Add GROQ_API_KEY to backend/.env, or on a Mac run: bash backend/scripts/setup_mlx.sh";

const PROVIDER_LABEL: Record<string, string> = { groq: "Groq", openai: "OpenAI", mlx: "Local" };

/** "mlx-community/Llama-3.2-3B-Instruct-4bit" -> "Llama 3.2 3B". */
export function shortModelName(model?: string | null): string {
  if (!model) return "";
  const base = model.split("/").pop() ?? model;
  return base
    .replace(/-(instruct|chat|\d+bit|bf16|fp16|mlx)\b/gi, "")
    .split(/[-_]/)
    .filter(Boolean)
    .join(" ");
}

/** Short label for a provider; local models include their short model name. */
export function llmLabel(provider?: string | null, model?: string | null): string {
  if (!provider) return "No language model";
  const name = PROVIDER_LABEL[provider] ?? provider;
  if (provider === "mlx") {
    const short = shortModelName(model);
    return short ? `${name} (${short})` : name;
  }
  return name;
}
