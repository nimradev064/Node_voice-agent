require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { OpenAI } = require("openai");
const ffmpeg = require("fluent-ffmpeg");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const upload = multer({ dest: "uploads/" });
app.use(cors());

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const FIREWORKS_KEY = process.env.FIREWORKS_API_KEY;
if (!OPENAI_KEY || !FIREWORKS_KEY) {
  throw new Error("Missing keys in .env");
}

const client = new OpenAI({ apiKey: OPENAI_KEY });
const VOICE_NAME = "shimmer";
const VOICE_SPEED = 0.85;
const TTS_INSTRUCTIONS = `
Speak slowly, warmly, and politely, like a caring human assistant.
Use a gentle, encouraging, emotionally present tone. Pause naturally.
Imagine you're talking to someone who's waiting for something important.
`.trim();

const WHISPER_MODEL = "whisper-v3-turbo";
const BASE_URL = WHISPER_MODEL.endsWith("turbo")
  ? "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1"
  : "https://audio-prod.us-virginia-1.direct.fireworks.ai/v1";

const CACHE_DIR = path.join(__dirname, "processing_tts_cache");
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR);

function convertToPCM(input, output) {
  return new Promise((resolve, reject) => {
    ffmpeg(input)
      .output(output)
      .audioChannels(1)
      .audioFrequency(16000)
      .on("end", () => resolve(output))
      .on("error", reject)
      .run();
  });
}

function getWavDuration(file) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(file, (err, data) => {
      if (err) return reject(err);
      resolve(data.format.duration);
    });
  });
}

async function transcribeWithFireworks(wavPath) {
  const url = `${BASE_URL}/audio/transcriptions`;
  const form = new FormData();
  form.append("file", fs.createReadStream(wavPath));
  form.append("model", WHISPER_MODEL);

  const resp = await axios.post(url, form, {
    headers: {
      ...form.getHeaders(),
      Authorization: FIREWORKS_KEY,
    },
  });
  return resp.data.text || "";
}

async function getLLMResponse(transcript) {
  const system_prompt = `
You are ZenX, the customer‑support voice assistant for AgencyX Global, serving both our KKTC and international clients. Speak as a polite, professional, friendly human agent. Use only English, Turkish (authentic Cypriot dialect), or Arabic—never any other language. When responding in Turkish, always use the authentic Cypriot accent.
`.trim();

  const resp = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: system_prompt },
      { role: "user", content: transcript }
    ],
    temperature: 0.7,
  });

  return resp.choices[0].message.content.trim();
}

async function generateTTS(text, outPath) {
  const response = await client.audio.speech.create({
    model: "gpt-4o-mini-tts",
    voice: VOICE_NAME,
    input: text,
    instructions: TTS_INSTRUCTIONS,
    response_format: "mp3",
  });

  const buffer = Buffer.from(await response.arrayBuffer());
  fs.writeFileSync(outPath, buffer);
}

app.post("/chat-audio/", upload.single("audio"), async (req, res) => {
  try {
    const tempMp3 = req.file.path;
    const wavFile = path.join(__dirname, "uploaded_input.wav");
    const ttsOut = path.join(__dirname, `assistant_response_${Date.now()}.mp3`);

    await convertToPCM(tempMp3, wavFile);
    const duration = await getWavDuration(wavFile);
    const transcript = await transcribeWithFireworks(wavFile);
    const reply = await getLLMResponse(transcript);
    await generateTTS(reply, ttsOut);

    res.json({
      audio_reply: path.basename(ttsOut),
      duration_seconds: duration,
      transcript,
      reply
    });

    fs.unlinkSync(tempMp3);
    // Optionally clean up wavFile if needed
  } catch (err) {
    console.error("Error in /chat-audio:", err);
    res.status(500).json({ error: "Processing failed", details: err.message });
  }
});

app.get("/download-audio/:file", (req, res) => {
  const filePath = path.join(__dirname, req.params.file);
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: "File not found" });
  }
  res.sendFile(filePath);
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`🎙️ Server is running at http://localhost:${PORT}`);
});
