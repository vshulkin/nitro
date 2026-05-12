import { exit } from "node:process";
import { spawn } from "node:child_process";

import {
  streamText,
  type ToolSet,
  type ModelMessage,
  type SystemModelMessage,
} from "ai";
import type { LanguageModelV3 } from "@ai-sdk/provider";
import {
  createOpenAICompatible,
  OpenAICompatibleProviderOptions,
} from "@ai-sdk/openai-compatible";
import { createOpenAI, OpenAIResponsesProviderOptions } from "@ai-sdk/openai";
import { AnthropicProviderOptions, createAnthropic } from "@ai-sdk/anthropic";

import {
  getDefaultProvider,
  type ProviderInfoWithName,
  type ProviderInfo,
} from "./provider";
import { type ReasoningEffort } from "./settings";

function addCacheControl<T extends ModelMessage>(
  message: T,
  providerName: string,
): T {
  return {
    ...message,
    providerOptions: {
      ...message.providerOptions,
      [providerName]: {
        cacheControl: { type: "ephemeral" },
      },
    } satisfies AnthropicProviderOptions,
  };
}

// Mark the last tool with a cache breakpoint so system prompt + all tool
// definitions are cached together. The combined size (~1800+ tokens) clears
// Anthropic's 1024-token minimum; the system prompt alone does not.
function cacheLastTool(tools: ToolSet, providerName: string): ToolSet {
  const keys = Object.keys(tools);
  if (keys.length === 0) return tools;
  const lastKey = keys[keys.length - 1]!;
  const lastTool = tools[lastKey]!;
  return {
    ...tools,
    [lastKey]: {
      ...lastTool,
      providerOptions: {
        ...lastTool.providerOptions,
        [providerName]: {
          cacheControl: { type: "ephemeral" },
        },
      },
    },
  } as unknown as ToolSet;
}

function createSystemMessage(systemPrompt: string): SystemModelMessage {
  const systemMessage: SystemModelMessage = {
    role: "system",
    content: systemPrompt,
  };
  return systemMessage;
}

export interface GenerationOptions {
  maxOutputTokens?: number;
  reasoningEffort?: ReasoningEffort;
}

function promptToString(prompt: unknown[]): string {
  const parts: string[] = [];
  for (const msg of prompt as Array<{ role: string; content: unknown }>) {
    if (msg.role === "system") {
      parts.push(msg.content as string);
    } else if (msg.role === "user") {
      const content = msg.content as Array<{ type: string; text?: string }>;
      const text = content
        .filter(p => p.type === "text")
        .map(p => p.text ?? "")
        .join("");
      parts.push(`Human: ${text}`);
    } else if (msg.role === "assistant") {
      const content = msg.content as Array<{ type: string; text?: string }>;
      const text = content
        .filter(p => p.type === "text")
        .map(p => p.text ?? "")
        .join("");
      if (text) parts.push(`Assistant: ${text}`);
    } else if (msg.role === "tool") {
      const content = msg.content as Array<{ result: unknown }>;
      const text = content
        .map(p =>
          typeof p.result === "string" ? p.result : JSON.stringify(p.result),
        )
        .join("\n");
      parts.push(`Tool Result: ${text}`);
    }
  }
  return parts.join("\n\n");
}

function createClaudeCodeModel(modelId: string): LanguageModelV3 {
  return {
    specificationVersion: "v3",
    provider: "claude-code",
    modelId,
    supportedUrls: {},

    doGenerate() {
      return Promise.reject(
        new Error("claude-code provider does not support doGenerate"),
      );
    },

    doStream(options) {
      const prompt = promptToString(options.prompt as unknown[]);

      const stream = new ReadableStream({
        start(controller) {
          const proc = spawn("claude", ["--print", "--output-format", "json"], {
            stdio: ["pipe", "pipe", "pipe"],
          });

          let output = "";
          proc.stdout.on("data", (chunk: Buffer) => {
            output += chunk.toString();
          });
          proc.stderr.on("data", (chunk: Buffer) => {
            process.stderr.write(`[claude-code] ${chunk.toString()}`);
          });

          proc.on("error", (err: Error) => {
            controller.enqueue({ type: "error", error: err });
            controller.close();
          });

          proc.on("close", (code: number | null) => {
            if (code !== 0) {
              controller.enqueue({
                type: "error",
                error: new Error(`claude exited with code ${code}`),
              });
              controller.close();
              return;
            }

            let text = output.trim();
            let inputCacheRead = 0;
            let inputCacheWrite = 0;
            let inputTotal: number;
            let outputTotal: number;

            try {
              const json = JSON.parse(text) as {
                result?: string;
                usage?: {
                  input_tokens?: number;
                  output_tokens?: number;
                  cache_read_input_tokens?: number;
                  cache_creation_input_tokens?: number;
                };
              };
              text = json.result ?? text;
              const u = json.usage ?? {};
              inputCacheRead = u.cache_read_input_tokens ?? 0;
              inputCacheWrite = u.cache_creation_input_tokens ?? 0;
              inputTotal =
                (u.input_tokens ?? 0) + inputCacheRead + inputCacheWrite;
              outputTotal = u.output_tokens ?? 0;
            } catch {
              inputTotal = Math.ceil(prompt.length / 4);
              outputTotal = Math.ceil(text.length / 4);
            }

            const id = "text-0";
            controller.enqueue({ type: "stream-start", warnings: [] });
            controller.enqueue({ type: "text-start", id });
            controller.enqueue({ type: "text-delta", id, delta: text });
            controller.enqueue({ type: "text-end", id });
            controller.enqueue({
              type: "finish",
              finishReason: "stop",
              usage: {
                inputTokens: {
                  total: inputTotal,
                  noCache: inputTotal - inputCacheRead - inputCacheWrite,
                  cacheRead: inputCacheRead,
                  cacheWrite: inputCacheWrite,
                },
                outputTokens: { total: outputTotal },
              },
            });
            controller.close();
          });

          proc.stdin.write(prompt);
          proc.stdin.end();
        },
      });

      return Promise.resolve({ stream });
    },
  };
}

function createClient(name: string, provider: ProviderInfo) {
  switch (provider.apiType) {
    case "openai-responses":
      return createOpenAI({
        name,
        baseURL: provider.baseURL,
        apiKey: provider.apiKey,
      });
    case "anthropic":
      return createAnthropic({
        name,
        baseURL: provider.baseURL,
        apiKey: provider.apiKey,
      });
    case "claude-code":
      return (_modelId: string) => createClaudeCodeModel(_modelId);
    case "openai-compatible":
    default:
      return createOpenAICompatible({
        name,
        baseURL: provider.baseURL,
        apiKey: provider.apiKey,
      });
  }
}

function getProviderOptions(
  provider: ProviderInfoWithName,
  reasoningEffort?: ReasoningEffort,
) {
  if (
    !reasoningEffort
    || reasoningEffort === "none"
    || provider.apiType === "claude-code"
  ) {
    return {};
  }
  switch (provider.apiType) {
    case "openai-compatible": {
      const result: OpenAICompatibleProviderOptions = {
        reasoningEffort: reasoningEffort,
      };
      return result;
    }
    case "openai-responses": {
      const result: OpenAIResponsesProviderOptions = {
        reasoningEffort: reasoningEffort,
      };
      return result;
    }
    case "anthropic": {
      const effortMap: Record<
        Exclude<ReasoningEffort, "none">,
        "low" | "high" | "medium" | "max"
      > = {
        low: "low",
        med: "medium",
        high: "high",
      };
      const result: AnthropicProviderOptions = {
        thinking: { type: "adaptive" },
        effort: effortMap[reasoningEffort],
      };
      return result;
    }
  }
}

export function generateCompletion(
  provider: ProviderInfoWithName,
  messages: ModelMessage[],
  systemPrompt: string,
  options?: GenerationOptions,
  tools?: ToolSet,
) {
  const client = createClient(provider.name, provider);

  const providerOptions = getProviderOptions(
    provider,
    options?.reasoningEffort,
  );

  const isAnthropic = provider.apiType === "anthropic";

  const systemMessage = createSystemMessage(systemPrompt);

  const cachedMessages =
    isAnthropic && messages.length > 0
      ? [
          ...messages.slice(0, -1),
          addCacheControl(messages[messages.length - 1]!, provider.name),
        ]
      : messages;

  // Cache the last tool definition so system prompt + all tools are cached
  // together (~1800+ tokens, above Anthropic's 1024-token minimum). The system
  // prompt alone is ~800 tokens and would be silently ignored.
  const finalTools =
    isAnthropic && tools ? cacheLastTool(tools, provider.name) : tools;

  return streamText({
    model: client(provider.model),
    messages: cachedMessages,
    system: systemMessage,
    maxOutputTokens: options?.maxOutputTokens,
    tools: finalTools,
    providerOptions: {
      [provider.name]: providerOptions,
    },
  });
}

export function getDefaultChatProvider(): ProviderInfoWithName | null {
  return getDefaultProvider() ?? null;
}

export function transformInput(input: string): string {
  const trimmed = input.trim();

  switch (trimmed) {
    case "/exit":
      exit(0);
      break;
    default:
      return input;
  }
}
