/** Discover ghc-api models, including token limits, when pi starts. */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface CatalogModel {
  id: string;
  display_name?: string;
  model_type?: string | null;
  context_window?: number | null;
  max_output_tokens?: number | null;
  input?: string[] | null;
  reasoning?: boolean | null;
  supported_endpoints?: string[] | null;
}

function selectApi(model: CatalogModel) {
  const endpoints = new Set(
    (model.supported_endpoints ?? []).map((path) => path.replace(/^\/v1\//, "/")),
  );
  if (endpoints.has("/responses")) return "openai-responses" as const;
  if (endpoints.has("/chat/completions")) return "openai-completions" as const;
  if (endpoints.has("/messages")) return "anthropic-messages" as const;
  return null;
}

export default async function (pi: ExtensionAPI) {
  const baseUrl = (process.env.GHC_API_BASE_URL ?? "http://localhost:8313/v1").replace(/\/+$/, "");
  const apiKey = process.env.GHC_API_KEY;
  const response = await fetch(`${baseUrl}/models`, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    throw new Error(`ghc-api model discovery failed: HTTP ${response.status}`);
  }
  const catalog = (await response.json()) as { data?: CatalogModel[] };
  if (!Array.isArray(catalog.data)) {
    throw new Error("ghc-api returned an invalid model catalog");
  }

  const models = catalog.data.flatMap((model) => {
    if (model.model_type === "embeddings" || model.model_type === "decision") return [];
    const api = selectApi(model);
    const contextWindow = model.context_window;
    const maxTokens = model.max_output_tokens;
    // Skip entries whose budgets are unknown rather than registering guessed limits.
    if (!api || typeof contextWindow !== "number" || typeof maxTokens !== "number" ||
        !Number.isSafeInteger(contextWindow) || !Number.isSafeInteger(maxTokens) ||
        contextWindow <= 0 || maxTokens <= 0) return [];
    const input = (model.input ?? ["text"]).filter(
      (kind): kind is "text" | "image" => kind === "text" || kind === "image",
    );
    return [{
      id: model.id,
      name: model.display_name ?? model.id,
      api,
      baseUrl: api === "anthropic-messages" ? baseUrl.replace(/\/v1$/, "") : baseUrl,
      contextWindow,
      maxTokens,
      input: input.length ? input : ["text" as const],
      reasoning: model.reasoning === true,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    }];
  });
  if (!models.length) {
    throw new Error("ghc-api advertised no chat models with known context and output limits");
  }

  pi.registerProvider("ghc-api", {
    baseUrl,
    apiKey: apiKey || "ghc-api-local",
    models,
  });
}
