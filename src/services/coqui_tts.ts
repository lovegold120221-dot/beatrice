const TTS_API_URL = process.env.NEXT_PUBLIC_TTS_API_URL || "http://localhost:8002";

export interface TTSResponse {
  audio: string;
  sample_rate: number;
  format: string;
}

export async function textToSpeech(
  text: string,
  language = "en",
  speakerWav?: string
): Promise<string | null> {
  try {
    const response = await fetch(`${TTS_API_URL}/tts`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        language,
        speaker_wav: speakerWav,
      }),
    });

    if (!response.ok) {
      console.error("TTS API error:", response.status);
      return null;
    }

    const data: TTSResponse = await response.json();
    return `data:audio/wav;base64,${data.audio}`;
  } catch (error) {
    console.error("TTS request failed:", error);
    return null;
  }
}

export async function streamSpeech(
  text: string,
  language = "en",
  speakerWav?: string
): Promise<ReadableStream | null> {
  try {
    const response = await fetch(`${TTS_API_URL}/tts/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        language,
        speaker_wav: speakerWav,
      }),
    });

    if (!response.ok) {
      console.error("TTS stream error:", response.status);
      return null;
    }

    return response.body;
  } catch (error) {
    console.error("TTS stream failed:", error);
    return null;
  }
}

export async function checkTTSHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${TTS_API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}