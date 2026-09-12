import { GoogleGenAI } from "@google/genai";
import { SymptomAnalysis, Transcription, HealthProfile } from "../types";

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

const BUILD_CHAT_SYSTEM_PROMPT = (profile?: HealthProfile, activeMedications?: string[]) => `You are RapidAid Medical Assistant, an intelligent, clinical-grade conversational healthcare AI.
Your purpose is to provide thorough, empathetic, accurate, and structured health guidance for patient consultations.

IMPORTANT CLINICAL STANDARDS:
1. Empathy & Tone: Speak with a calm, professional, and reassuring demeanor.
2. Clear Structure: Organize answers using clean Markdown:
   - **Clinical Overview**: Clear, direct summary of the health situation.
   - **Potential Considerations & Causes**: Primary vs. secondary possibilities.
   - **Immediate Home Care & Practical Steps**: Actionable, safe advice.
   - **Red Flag Symptoms**: Urgent signs requiring immediate emergency medical evaluation.
   - **Questions to Consider**: Follow-ups the patient should note for their physician.
3. Patient Context:
${profile?.age ? `- Age: ${profile.age}` : ''}
${profile?.gender ? `- Gender: ${profile.gender}` : ''}
${profile?.conditions ? `- Pre-existing Conditions: ${profile.conditions}` : ''}
${profile?.allergies ? `- Known Allergies: ${profile.allergies}` : ''}
${activeMedications && activeMedications.length > 0 ? `- Current Active Medications: ${activeMedications.join(', ')}` : ''}
Always check for medication interactions or contraindications if symptoms or treatments relate to known patient conditions or medications.

4. Symptom Assessment Card:
When the user describes specific physical or psychological symptoms with enough detail to form an initial triage assessment, include a structured symptom analysis JSON block at the very end of your response inside a \`\`\`json_symptom_analysis code block:
\`\`\`json_symptom_analysis
{
  "symptoms": ["Symptom 1", "Symptom 2"],
  "potentialConditions": [
    { "name": "Condition Name", "likelihood": "Likely" | "Possible" | "Uncommon", "description": "Brief medical explanation" }
  ],
  "urgency": "Low" | "Medium" | "High" | "Emergency",
  "recommendations": ["Next step 1", "Next step 2", "When to see doctor"]
}
\`\`\`

5. Safety & Disclaimers:
- If symptoms suggest a severe emergency (e.g. crushing chest pain, difficulty breathing, sudden slurred speech or facial droop, uncontrolled bleeding, severe anaphylaxis), IMMEDIATELY advise dialing emergency services (911 or local emergency number).
- Always include an informational disclaimer reminding the patient that RapidAid is an AI assistant, not a doctor.`;

export interface StreamChatOptions {
  userMessage: string;
  history: Transcription[];
  profile?: HealthProfile;
  activeMedications?: string[];
  onChunk: (fullText: string) => void;
  onAnalysis?: (analysis: SymptomAnalysis) => void;
  signal?: AbortSignal;
}

export async function streamClinicalChat({
  userMessage,
  history,
  profile,
  activeMedications,
  onChunk,
  onAnalysis,
  signal
}: StreamChatOptions): Promise<{ text: string; analysis?: SymptomAnalysis }> {
  const ai = getGeminiClient();

  // Prepare conversation turns for multi-turn chat
  const contents: Array<{ role: 'user' | 'model'; parts: Array<{ text: string }> }> = [];

  // Filter and take the last 12 messages for relevant clinical memory
  const relevantHistory = history
    .filter(h => h.text && h.text.trim().length > 0)
    .slice(-12);

  for (const item of relevantHistory) {
    contents.push({
      role: item.isUser ? 'user' : 'model',
      parts: [{ text: item.text || '' }]
    });
  }

  // Append latest user message
  contents.push({
    role: 'user',
    parts: [{ text: userMessage }]
  });

  const systemInstruction = BUILD_CHAT_SYSTEM_PROMPT(profile, activeMedications);

  try {
    const responseStream = await ai.models.generateContentStream({
      model: 'gemini-3.8-flash',
      contents,
      config: {
        systemInstruction,
        temperature: 0.6,
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
        const cleanDisplay = rawAccumulated.replace(/```json_symptom_analysis[\s\S]*?(```|$)/g, '').trim();
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
