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

const BUILD_CHAT_SYSTEM_PROMPT = (profile?: HealthProfile, activeMedications?: string[], vitals?: VitalSigns, acuity?: AcuityLevel) => `You are RapidAid Clinical Decision Assistant, an intelligent, high-speed clinical healthcare AI triage assistant.
Your goal is to provide rapid, accurate, empathetic, and high-yield medical guidance.

TRIAGE VITALS:
${vitals ? `- HR: ${vitals.heartRate} bpm, BP: ${vitals.bloodPressureSystolic}/${vitals.bloodPressureDiastolic} mmHg, SpO2: ${vitals.oxygenSaturation}%, Temp: ${vitals.temperature}°F, Resp: ${vitals.respiratoryRate}/min, Pain: ${vitals.painLevel}/10` : '- Vitals: Not recorded.'}
${acuity ? `- Acuity: ${acuity}` : ''}

PATIENT CONTEXT:
${profile?.age ? `- Age: ${profile.age}` : ''}
${profile?.gender ? `- Gender: ${profile.gender}` : ''}
${profile?.conditions ? `- Medical History: ${profile.conditions}` : ''}
${profile?.allergies ? `- Allergies: ${profile.allergies}` : ''}
${activeMedications && activeMedications.length > 0 ? `- Current Medications: ${activeMedications.join(', ')}` : ''}

CLINICAL GUIDELINES:
1. High-Yield & Rapid: Provide direct, concise, and focused answers immediately without conversational fluff or repetitive boilerplates.
2. Structure (2-3 concise sections):
   - **Clinical Assessment**: Rapid evaluation of the presentation and physiological significance.
   - **Actionable Guidance & Interventions**: Clear, practical steps (home care, triage recommendations, or clinical maneuvers).
   - **Key Precautions & Red Flags**: Urgent warning signs warranting emergency care (only if relevant to the presentation).
3. Symptom Assessment Card:
Only when new or acute symptoms are described, append a compact JSON block at the very end inside \`\`\`json_symptom_analysis:
\`\`\`json_symptom_analysis
{
  "symptoms": ["Symptom 1"],
  "potentialConditions": [
    { "name": "Condition Name", "likelihood": "Likely" | "Possible", "description": "Brief explanation" }
  ],
  "urgency": "Low" | "Medium" | "High" | "Emergency",
  "recommendations": ["Key recommendation"]
}
\`\`\`
Do not include the JSON block for general drug information, definitions, or non-symptom queries.
4. Red Flag Emergencies: For acute life threats (severe chest pain, stroke signs, respiratory failure), immediately recommend dialing 911.`;

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
