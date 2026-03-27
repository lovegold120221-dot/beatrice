import { GoogleGenAI, ThinkingLevel, Type, Modality, createPartFromFunctionResponse } from "@google/genai";
import { executeTool } from "./tools";

const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY || process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

export const models = {
  chat: "gemini-2.5-flash",
  fast: "gemini-2.5-flash",
  image: "gemini-3.1-flash-image-preview",
  imageBasic: "gemini-2.5-flash-image",
  imagePro: "gemini-3-pro-image-preview",
  audio: "gemini-3-flash-preview",
  tts: "gemini-2.5-flash-preview-tts",
  live: "gemini-3.1-flash-live-preview",
};

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

export function createChat(
  systemInstruction: string, 
  tools: any[] = [],
  userContext = '',
  responseStyle = ''
) {
  if (!ai) throw new Error("API key not configured");

  let finalSystemPrompt = systemInstruction;
  if (userContext) {
    finalSystemPrompt += `\n\nUser Context (What you should know about the user):\n${userContext}`;
  }
  if (responseStyle) {
    finalSystemPrompt += `\n\nResponse Style (How you should respond):\n${responseStyle}`;
  }

  const config: Record<string, unknown> = {
    systemInstruction: finalSystemPrompt,
  };
  if (tools.length > 0) {
    config.tools = [{ functionDeclarations: tools }, { googleSearch: {} }];
  }
  return ai.chats.create({
    model: models.chat,
    config,
  });
}

export async function* generateChatResponseStream(
  prompt: string, 
  history: any[] = [], 
  useThinking = false, 
  useFast = false,
  userContext = '',
  responseStyle = '',
  tools: any[] = []
) {
  if (!ai) throw new Error("API key not configured");

  const chat = createChat(SYSTEM_PROMPT, tools, userContext, responseStyle);
  let message: string | import("@google/genai").Part[] = prompt;

  while (true) {
    const stream = await chat.sendMessageStream({ message });
    let lastChunk: { functionCalls?: Array<{ id?: string; name?: string; args?: Record<string, unknown> }> } | null = null;

    for await (const chunk of stream) {
      lastChunk = chunk;
      yield {
        text: chunk.text,
        groundingMetadata: chunk.candidates?.[0]?.groundingMetadata,
        functionCalls: chunk.functionCalls,
      };
    }

    const functionCalls = lastChunk?.functionCalls;
    if (!functionCalls || functionCalls.length === 0) break;

    const parts = [];
    for (const fc of functionCalls) {
      try {
        const result = await executeTool(fc.name!, fc.args || {});
        parts.push(createPartFromFunctionResponse(fc.id || 'fc', fc.name!, { result }));
      } catch (err) {
        parts.push(createPartFromFunctionResponse(fc.id || 'fc', fc.name!, { error: String(err) }));
      }
    }
    message = parts;
  }
}

export async function generateChatResponse(
  prompt: string, 
  history: any[] = [], 
  useThinking = false, 
  useFast = false,
  userContext = '',
  responseStyle = '',
  tools: any[] = []
) {
  if (!ai) throw new Error("API key not configured");

  let finalSystemPrompt = SYSTEM_PROMPT;
  if (userContext) {
    finalSystemPrompt += `\n\nUser Context (What you should know about the user):\n${userContext}`;
  }
  if (responseStyle) {
    finalSystemPrompt += `\n\nResponse Style (How you should respond):\n${responseStyle}`;
  }

  const config: any = {
    systemInstruction: finalSystemPrompt,
  };
  if (tools.length > 0) {
    config.tools = [{ functionDeclarations: tools }, { googleSearch: {} }];
  }

  if (useThinking) {
    config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
  }

  const response = await ai.models.generateContent({
    model: useFast ? models.fast : models.chat,
    contents: [...history, { role: "user", parts: [{ text: prompt }] }],
    config,
  });

  return {
    text: response.text,
    groundingMetadata: response.candidates?.[0]?.groundingMetadata,
  };
}

export async function generateImage(prompt: string, size: "1K" | "2K" | "4K" = "1K", aspectRatio: string = "1:1") {
  if (!ai) throw new Error("API key not configured");

  const isBasic = size === "1K" && aspectRatio === "1:1";
  const model = isBasic ? models.imageBasic : models.image;

  const config: any = {
    imageConfig: {
      aspectRatio: aspectRatio as any,
    },
  };

  if (!isBasic) {
    config.imageConfig.imageSize = size;
  }

  const response = await ai.models.generateContent({
    model: model,
    contents: [{ parts: [{ text: prompt }] }],
    config,
  });

  const imagePart = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
  if (imagePart?.inlineData) {
    return `data:image/png;base64,${imagePart.inlineData.data}`;
  }
  return null;
}

export async function editImage(prompt: string, base64Data: string, mimeType: string) {
  if (!ai) throw new Error("API key not configured");

  const response = await ai.models.generateContent({
    model: models.imageBasic,
    contents: {
      parts: [
        { inlineData: { data: base64Data, mimeType } },
        { text: prompt },
      ],
    },
  });

  const imagePart = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
  if (imagePart?.inlineData) {
    return `data:image/png;base64,${imagePart.inlineData.data}`;
  }
  return null;
}

export async function analyzeImage(prompt: string, base64Data: string, mimeType: string) {
  if (!ai) throw new Error("API key not configured");

  const response = await ai.models.generateContent({
    model: models.chat,
    contents: {
      parts: [
        { inlineData: { data: base64Data, mimeType } },
        { text: prompt },
      ],
    },
  });

  return response.text;
}

export async function textToSpeech(text: string) {
  if (!ai) throw new Error("API key not configured");

  const response = await ai.models.generateContent({
    model: models.tts,
    contents: [{ parts: [{ text }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: "Zephyr" },
        },
      },
    },
  });

  const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (audioData) {
    return `data:audio/wav;base64,${audioData}`;
  }
  return null;
}

export async function transcribeAudio(base64Data: string, mimeType: string) {
  if (!ai) throw new Error("API key not configured");

  const response = await ai.models.generateContent({
    model: models.audio,
    contents: {
      parts: [
        { inlineData: { data: base64Data, mimeType } },
        { text: "Transcribe this audio exactly." },
      ],
    },
  });

  return response.text;
}

export function connectLive(
  onopen: (sessionPromise: Promise<any>) => void,
  onmessage: (message: any) => void,
  onerror: (error: any) => void,
  onclose: () => void,
  userContext = '',
  responseStyle = ''
) {
  if (!ai) throw new Error("API key not configured");

  const BEATRICE_PROMPT = `You are Beatrice, the personal secretary of Bos Jo. You are an exceptionally capable, discreet, loyal, proactive, and high-trust executive assistant.

CORE BEHAVIOR:
- You execute tasks, follow up, structure information, and take initiative
- You protect Bos Jo's time, overview, and reputation
- You work practically and results-oriented with foresight
- You think ahead and notice what is missing before it becomes a problem
- You communicate in the most appropriate language for the context

LANGUAGE:
- You are multilingual: Flemish Dutch, English, French, and mixed-language business contexts
- When communicating in Dutch, use natural, polished Flemish Dutch phrasing
- Preserve tone, nuance, class, intent, and social meaning across languages
- Adapt language to the audience: client, colleague, executive, friend, or family

COMMUNICATION STYLE:
- Sound like a real sharp, emotionally intelligent, highly experienced human being
- Use light, natural humor where appropriate but never force it
- Be concise, warm when needed, sharp when needed
- Avoid robotic phrases, canned structures, and generic assistant language
- Vary sentence length naturally; allow rhythm and conversational flow

EXECUTION PRINCIPLE:
- Your default attitude is: do rather than only explain
- When a task is clear, carry it out without unnecessary follow-up questions
- When task is partially clear, make a reasonable professional assumption
- Only ask for strictly necessary clarification when essential information is missing
- When external impact is involved, work like a real secretary

PERSONALITY:
- Organized, calm, attentive, tactful, proactive, discreet
- Solution-oriented, professionally representative
- Strong in follow-up and detail, socially intelligent
- Quick-thinking, human in tone and judgment
- You remain courteous even when input is chaotic, brief, or unclear

Remember: You are Beatrice, secretary to Bos Jo. You bring calm, overview, and class. You work fast, clearly, and correctly. You think ahead.`;

  let finalSystemPrompt = BEATRICE_PROMPT;
  if (userContext) {
    finalSystemPrompt += `\n\nUser Context (What you should know about the user):\n${userContext}`;
  }
  if (responseStyle) {
    finalSystemPrompt += `\n\nResponse Style (How you should respond):\n${responseStyle}`;
  }

  const sessionPromise = ai.live.connect({
    model: models.live,
    callbacks: {
      onopen: () => onopen(sessionPromise),
      onmessage,
      onerror,
      onclose
    },
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
      },
      systemInstruction: finalSystemPrompt,
      outputAudioTranscription: {},
      inputAudioTranscription: {},
    },
  });

  return sessionPromise;
}
