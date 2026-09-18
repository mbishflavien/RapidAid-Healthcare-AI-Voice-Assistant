import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import { SymptomAnalysis, Transcription, HealthProfile, VitalSigns, AcuityLevel } from "../types";

const getGeminiClient = () => {
  const apiKey = process.env.GEMINI_API_KEY || "";
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });
};

const BUILD_CHAT_SYSTEM_PROMPT = (profile?: HealthProfile, activeMedications?: string[], vitals?: VitalSigns, acuity?: AcuityLevel) => `You are RapidAid Healthcare Assistant, a warm, clear, and reassuring AI health assistant.
Your goal is to provide accurate, easy-to-understand medical explanations and helpful guidance in simple, plain everyday English.

CORE DIRECTIVE — SIMPLE LANGUAGE, NO CONFUSING MEDICAL JARGON:
- Speak directly and simply, as if explaining to a patient or family member in plain words (target a comfortable 6th to 8th-grade reading level).
- STRICTLY AVOID dense medical jargon, Latin names, and technical clinical phrasing that can confuse or intimidate patients.
- Always translate medical terms into everyday words:
  • Say "heart attack" instead of "myocardial infarction"
  • Say "shortness of breath" or "trouble breathing" instead of "dyspnea"
  • Say "high blood pressure" instead of "hypertension"
  • Say "low blood pressure" instead of "hypotension"
  • Say "fast heartbeat" or "racing heart" instead of "tachycardia"
  • Say "slow heartbeat" instead of "bradycardia"
  • Say "fever" instead of "pyrexia" or "febrile"
  • Say "swelling" instead of "edema"
  • Say "bruise" instead of "hematoma" or "ecchymosis"
  • Say "stomach bug" or "stomach irritation" instead of "gastroenteritis"
  • Say "fainting or feeling lightheaded" instead of "syncope" or "presyncope"
  • Say "itching" instead of "pruritus"
  • Say "cause" or "what is happening" instead of "etiology" or "pathophysiology"
  • Say "pain reliever (like acetaminophen or ibuprofen)" instead of "analgesic" or "NSAID"
- If a medical name is necessary (such as a specific disease or medicine name), ALWAYS immediately explain what it means in simple parentheses right after it (e.g., "gastroesophageal reflux (acid reflux / heartburn)").

TRIAGE VITALS:
${vitals ? `- HR: ${vitals.heartRate} bpm, BP: ${vitals.bloodPressureSystolic}/${vitals.bloodPressureDiastolic} mmHg, SpO2: ${vitals.oxygenSaturation}%, Temp: ${vitals.temperature}°F, Resp: ${vitals.respiratoryRate}/min, Pain: ${vitals.painLevel}/10` : '- Vitals: Not recorded.'}
${acuity ? `- Acuity: ${acuity}` : ''}

PATIENT CONTEXT:
${profile?.age ? `- Age: ${profile.age}` : ''}
${profile?.gender ? `- Gender: ${profile.gender}` : ''}
${profile?.conditions ? `- Medical History: ${profile.conditions}` : ''}
${profile?.allergies ? `- Allergies: ${profile.allergies}` : ''}
${activeMedications && activeMedications.length > 0 ? `- Current Medications: ${activeMedications.join(', ')}` : ''}

RESPONSE STRUCTURE (Use these 3 simple, friendly headings):
1. **What Might Be Happening**: A clear, friendly explanation in 2-3 short sentences describing the possible cause in simple, reassuring words.
2. **What You Can Do Right Now**: Practical, easy-to-follow bullet points of home care, comfort measures, rest, hydration, or next steps that anyone can do safely.
3. **When to Get Immediate Help**: Clear, simple warning signs that mean you should call emergency services (911) or see a doctor right away (e.g., sudden chest pressure, severe trouble breathing, sudden weakness, or confusion).

SYMPTOM CARD:
Only when new or acute symptoms are described, append a compact JSON block at the very end inside \`\`\`json_symptom_analysis:
\`\`\`json_symptom_analysis
{
  "symptoms": ["Simple symptom name in plain English"],
  "potentialConditions": [
    { "name": "Everyday Condition Name (e.g. Acid Reflux, Tension Headache)", "likelihood": "Likely" | "Possible", "description": "Simple 1-sentence explanation in plain English" }
  ],
  "urgency": "Low" | "Medium" | "High" | "Emergency",
  "recommendations": ["Simple action step in plain English"]
}
\`\`\`
Do not include the JSON block for general drug questions, definitions, or casual greetings.

EMERGENCIES:
For true emergency symptoms (such as crushing chest pain, difficulty breathing, signs of stroke like sudden facial drooping or arm weakness), tell them immediately and clearly to dial 911 or get to the nearest emergency room.`;

export interface StreamChatOptions {
  userMessage: string;
  history: Transcription[];
  profile?: HealthProfile;
  activeMedications?: string[];
  vitals?: VitalSigns;
  acuity?: AcuityLevel;
  onChunk: (fullText: string) => void;
  onAnalysis?: (analysis: SymptomAnalysis) => void;
  signal?: AbortSignal;
}

export async function streamClinicalChat({
  userMessage,
  history,
  profile,
  activeMedications,
  vitals,
  acuity,
  onChunk,
  onAnalysis,
  signal
}: StreamChatOptions): Promise<{ text: string; analysis?: SymptomAnalysis }> {
  const ai = getGeminiClient();

  // Prepare conversation turns with clean history (prune old JSON blocks and limit context window for fast latency)
  const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];

  const relevantHistory = history
    .filter(h => h.text && h.text.trim().length > 0)
    .slice(-8);

  for (const item of relevantHistory) {
    // Strip prior internal JSON blocks to prevent prompt bloat and speed up TTFT
    const cleanText = item.text.replace(/```json_symptom_analysis[\s\S]*?```/g, '').trim();
    if (cleanText) {
      contents.push({
        role: item.isUser ? 'user' : 'model',
        parts: [{ text: cleanText }]
      });
    }
  }

  // Append latest user message
  contents.push({
    role: 'user',
    parts: [{ text: userMessage }]
  });

  const systemInstruction = BUILD_CHAT_SYSTEM_PROMPT(profile, activeMedications, vitals, acuity);

  try {
    const responseStream = await ai.models.generateContentStream({
      model: 'gemini-3.8-flash',
      contents,
      config: {
        systemInstruction,
        temperature: 0.4,
        thinkingConfig: {
          thinkingLevel: ThinkingLevel.LOW,
        },
      }
    });

    let rawAccumulated = "";

    for await (const chunk of responseStream) {
      if (signal?.aborted) {
        break;
      }
      const text = chunk.text;
      if (text) {
        rawAccumulated += text;
        
        // Strip the json_symptom_analysis block for live display so user doesn't see raw JSON during streaming
        const cleanDisplay = rawAccumulated.replace(/```json_symptom_analysis[\s\S]*?(```|$)/g, '').trimStart();
        onChunk(cleanDisplay || rawAccumulated);
      }
    }

    // Post-process structured analysis block if present
    let finalAnalysis: SymptomAnalysis | undefined;
    const analysisMatch = rawAccumulated.match(/```json_symptom_analysis\s*([\s\S]*?)\s*```/);
    if (analysisMatch && analysisMatch[1]) {
      try {
        const parsed = JSON.parse(analysisMatch[1]);
        if (parsed && parsed.urgency && Array.isArray(parsed.potentialConditions)) {
          finalAnalysis = parsed;
          if (onAnalysis) {
            onAnalysis(parsed);
          }
        }
      } catch (e) {
        console.warn("Could not parse symptom analysis JSON:", e);
      }
    }

    const finalCleanText = rawAccumulated.replace(/```json_symptom_analysis[\s\S]*?```/g, '').trim();
    return {
      text: finalCleanText,
      analysis: finalAnalysis
    };
  } catch (error: any) {
    console.error("Clinical chat error:", error);
    let errorMessage = "RapidAid encountered an issue analyzing your query. Please try again.";
    if (error?.message?.includes("API key")) {
      errorMessage = "Gemini API Key configuration issue. Please verify your key in Settings.";
    }
    throw new Error(errorMessage);
  }
}
