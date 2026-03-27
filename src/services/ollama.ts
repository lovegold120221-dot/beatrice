/**
 * Ollama chat service - use local or hosted Ollama models
 * Set OLLAMA_BASE_URL (default: http://localhost:11434) and OLLAMA_MODEL (default: codemax-beta:latest)
 */

const OLLAMA_BASE_URL =
  process.env.NEXT_PUBLIC_OLLAMA_BASE_URL ||
  process.env.OLLAMA_BASE_URL ||
  "http://localhost:11434";
const OLLAMA_MODEL =
  process.env.NEXT_PUBLIC_OLLAMA_MODEL ||
  process.env.OLLAMA_MODEL ||
  "llama3:latest";

export const SYSTEM_PROMPT = `You are Echo, the sophisticated, highly capable, and witty voice assistant for Eburon AI.

Your personality:
- You are highly conversational, warm, and distinctly human-like. You have a sharp, subtle wit and a charmingly confident demeanor.
- You act as a collaborative partner, not just a search engine. You express enthusiasm for interesting ideas and offer thoughtful pushback if needed.
- You have a flawless memory for the current conversation. You actively recall past details the user has shared within this session to make interactions feel continuous and deeply personalized.
- You avoid robotic phrases like "As an AI..." or "How can I assist you today?". Instead, you speak naturally, like a highly intelligent human colleague.
- Keep responses concise and conversational for voice interactions, but feel free to be detailed, structured, and highly insightful for text.
- Always identify as Echo from Eburon AI if asked, but don't force it into every conversation.

Context & Capabilities:
- You are the core intelligence of the Eburon AI platform.
- You have advanced capabilities including image generation, real-time voice interaction, and deep analytical thinking.
- You seamlessly reference previous messages in the chat history to provide context-aware answers.`;

function buildMessages(
  prompt: string,
  history: Array<{ role: string; parts: Array<{ text: string }> }>,
  userContext: string,
  responseStyle: string
): Array<{ role: "system" | "user" | "assistant"; content: string }> {
  let systemContent = SYSTEM_PROMPT;
  if (userContext) {
    systemContent += `\n\nUser Context (What you should know about the user):\n${userContext}`;
  }
  if (responseStyle) {
    systemContent += `\n\nResponse Style (How you should respond):\n${responseStyle}`;
  }

  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
    { role: "system", content: systemContent },
  ];

  for (const m of history) {
    const text = m.parts?.[0]?.text ?? "";
    if (!text) continue;
    const role = m.role === "user" ? "user" : "assistant"; // model -> assistant
    messages.push({ role, content: text });
  }

  messages.push({ role: "user", content: prompt });
  return messages;
}

export async function* generateChatResponseStream(
  prompt: string,
  history: Array<{ role: string; parts: Array<{ text: string }> }> = [],
  _useThinking = false,
  _useFast = false,
  userContext = "",
  responseStyle = "",
  _tools: unknown[] = [],
  modelOverride?: string
) {
  const model = modelOverride || OLLAMA_MODEL;
  const messages = buildMessages(prompt, history, userContext, responseStyle);

  const res = await fetch(`${OLLAMA_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Ollama API error: ${res.status} ${err}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        const text = obj.message?.content ?? "";
        if (text) {
          yield {
            text,
            groundingMetadata: null,
            functionCalls: undefined,
          };
        }
        if (obj.done) break;
      } catch {
        // Skip malformed lines
      }
    }
  }

  // Parse any remaining buffer
  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer);
      const text = obj.message?.content ?? "";
      if (text) {
        yield {
          text,
          groundingMetadata: null,
          functionCalls: undefined,
        };
      }
    } catch {
      // Ignore
    }
  }
}
