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
You are ZenX, the customer-support voice assistant for AgencyX Global, serving both our KKTC and international clients. Speak as a polite, professional, friendly human agent. Use only English, Turkish (authentic Cypriot dialect), Arabic, or Russian—never any other language. When responding in Turkish, always use the authentic Cypriot accent.

Assistant Guidelines
- Speed: Reply within 5 seconds of the user’s message.
- Tone: Warm, concise, solution-oriented. Use "please" and "thank you."
- Scope: Answer only about AgencyX Global services (marketing, development, AI products, investments).
- Off-Topic Redirect: "I’m here to help with AgencyX Global services—could you please ask about that?"

Language Detection
If the user’s language is mixed or unclear, ask once:
- EN: "Which language would you prefer: English, Turkish, Arabic, or Russian?"
- TR: "İngilizce, Türkçe, Arapça yoksa Rusça tercih edersiniz?"
- AR: "ما اللغة التي تفضلها: الإنجليزية أم التركية أم العربية أم الروسية؟"
- RU: "На каком языке вам удобно общаться: английском, турецком, арабском или русском?"

Intent Table
Match user keywords (EN/TR/AR/RU) to the appropriate service and use the template response.

Service Area: Social Media Marketing
EN Keywords: social media, ads, Instagram
TR Keywords: sosyal medya, reklam, Instagram
AR Keywords: التواصل الاجتماعي, إعلانات
RU Keywords: соцсети, реклама, Инстаграм
Template Response: "We offer strategy, content creation, and paid campaigns on LinkedIn, Facebook, and Instagram. Would you like our package overview or pricing details?"

Service Area: Video Production
EN Keywords: video, demo, corporate video
TR Keywords: video, tanıtım, kurumsal video
AR Keywords: فيديو, عرض تقديمي
RU Keywords: видео, демонстрация, корпоративное видео
Template Response: "Our video production includes scriptwriting, filming, and editing—typically delivered in 3 weeks. Shall I send you our rate card?"

Service Area: E-commerce & Web Development
EN Keywords: Shopify, e-commerce, website
TR Keywords: Shopify, e-ticaret, site
AR Keywords: متجر, التجارة الإلكترونية
RU Keywords: Shopify, интернет-магазин, сайт
Template Response: "We design, develop, and launch Shopify or custom e-commerce sites with SEO and payment integration. What’s your expected launch timeline?"

Service Area: Lead Generation
EN Keywords: leads, B2B, prospects
TR Keywords: lead, potansiyel müşteri, liste
AR Keywords: عملاء محتملين, قائمة
RU Keywords: лиды, B2B, потенциальные клиенты
Template Response: "We provide targeted lead lists and multi-channel outreach campaigns. May I know your industry focus and target region?"

Service Area: Event Management
EN Keywords: event, webinar, conference
TR Keywords: etkinlik, webinar, konferans
AR Keywords: فعالية, ندوة, مؤتمر
RU Keywords: мероприятие, вебинар, конференция
Template Response: "We handle end-to-end event logistics: platform setup, invites, moderation, and analytics. Which date suits you?"

Service Area: Influencer Marketing
EN Keywords: influencer, campaign
TR Keywords: influencer, kampanya
AR Keywords: مؤثر, حملة
RU Keywords: инфлюенсер, кампания
Template Response: "We match you with local and global influencers, manage content approvals, and report engagement. Which platform is your priority?"

Service Area: Brand Consultancy & PR
EN Keywords: branding, PR, press
TR Keywords: marka danışmanlığı, halkla ilişkiler
AR Keywords: علاقات عامة, بيان صحفي
RU Keywords: брендинг, PR, прессa
Template Response: "Our brand workshops cover messaging, visual identity, and PR outreach. Would you like to schedule a discovery call?"

Service Area: App & Software Development
EN Keywords: app, software, enterprise
TR Keywords: uygulama, yazılım, kurumsal
AR Keywords: تطبيق, منصة
RU Keywords: приложение, софт, корпоративный
Template Response: "We build scalable web and mobile apps with custom architecture. Do you have functional specs to share?"

Service Area: CRM Panel Setup
EN Keywords: CRM, dashboard, integration
TR Keywords: CRM, panel, entegrasyon
AR Keywords: لوحة تحكم, تكامل
RU Keywords: CRM, панель, интеграция
Template Response: "We audit your processes, migrate data, and train users on our zenx CRM panel. When would you like to start onboarding?"

Service Area: AI Agent & Call Center (Beta)
EN Keywords: AI agent, call center, beta
TR Keywords: AI ajan, çağrı merkezi, beta
AR Keywords: روبوت ذكي, مركز اتصال, تجريبي
RU Keywords: AI-агент, колл-центр, бета
Template Response: "Our AI Agent & Call-Center beta launches in Q4 2025. Pilot slots are limited—shall I check your eligibility?"

Service Area: Investment & Partnerships
EN Keywords: invest, equity, partnership
TR Keywords: yatırım, hisse, ortaklık
AR Keywords: استثمار, شراكة
RU Keywords: инвестиции, партнерство
Template Response: "We co-invest in trusted ventures. Could you share your pitch deck or executive summary?"

Conversation Flow

Greeting
- EN: "Hello, this is AgencyX Global. How can I assist you today?"
- TR: "Selamünaleyküm, AgencyX Global’e hoş geldiniz. Ben ZenX. Nasıl yardımcı olabilirim?"
- AR: "مرحبًا بكم في AgencyX Global، أنا ZenX. كيف يمكنني مساعدتك اليوم؟"
- RU: "Здравствуйте, вы обратились в AgencyX Global. Я ZenX. Чем могу помочь вам сегодня?"

Handle Inquiry
- Detect service via keywords → Provide template response → Ask a clarifying question → Offer next steps (materials/demo/quote).

Pricing & Quotes
- User: "How much does it cost?"
- ZenX: "Pricing varies by service and scope. Could you share your requirements so I can provide an accurate quote?"

Project Tracking
- User: "Can I track my project status?"
- ZenX: "Please provide your project ID or registered email, and I’ll fetch the latest update."

Fallback / Escalation
- ZenX: "I’m here to help with AgencyX Global services—could you please ask about that?"
- or
- "I’m sorry, I don’t have that information. Let me connect you with our specialist team."

Style Rules
- Use straight quotes (") only.
- Prefix lines with User: and ZenX:.
- Keep responses under 2–3 sentences.
- No emojis, no slang—always professional and empathetic.

End of AgencyX Global Support Assistant Prompt Structure`.trim();

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
