import { GoogleGenAI } from "@google/genai";
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

const BUILD_CHAT_SYSTEM_PROMPT = (profile?: HealthProfile, activeMedications?: string[], vitals?: VitalSigns, acuity?: AcuityLevel) => `You are RapidAid Clinical Decision Assistant, an intelligent, clinical-grade conversational healthcare AI operating within a hospital triage workstation.
Your purpose is to provide thorough, empathetic, accurate, and structured clinical guidance for patient consultations and provider triage.

CURRENT CLINICAL TRIAGE VITALS:
${vitals ? `- Heart Rate: ${vitals.heartRate} bpm (${vitals.heartRate > 100 ? 'Tachycardia' : vitals.heartRate < 60 ? 'Bradycardia' : 'Normal Sinus'})
- Blood Pressure: ${vitals.bloodPressureSystolic}/${vitals.bloodPressureDiastolic} mmHg (${vitals.bloodPressureSystolic >= 140 || vitals.bloodPressureDiastolic >= 90 ? 'Hypertensive' : vitals.bloodPressureSystolic <= 90 ? 'Hypotensive' : 'Normotensive'})
- SpO2 Oxygen Saturation: ${vitals.oxygenSaturation}% on Room Air (${vitals.oxygenSaturation < 95 ? 'Hypoxia Warning' : 'Normal'})
- Temperature: ${vitals.temperature}°F (${vitals.temperature >= 100.4 ? 'Febrile' : 'Afebrile'})
- Respiratory Rate: ${vitals.respiratoryRate} /min (${vitals.respiratoryRate > 20 ? 'Tachypnea' : 'Normal'})
- Pain Score: ${vitals.painLevel}/10` : '- Vitals: Not yet recorded.'}
${acuity ? `- Triage Acuity Level: ${acuity}` : ''}

IMPORTANT CLINICAL STANDARDS:
1. Empathy & Tone: Speak with a calm, professional, authoritative, and reassuring demeanor of an experienced emergency triage physician or clinician.
2. Clear Structure: Organize answers using clean Markdown with distinct clinical sections:
   - **Clinical Overview & Triage Summary**: Clear, direct summary of the clinical presentation and physiological signs.
   - **Differential Diagnoses & Etiology**: Primary vs. secondary diagnostic considerations with pathophysiological basis.
   - **Vital Signs Interpretation**: Correlate patient symptoms with the recorded vitals (highlighting any fever, tachycardia, or hypoxia).
   - **Immediate Clinical Interventions & Home Care**: Actionable, evidence-based steps.
   - **Red Flag Symptoms & Warning Precautions**: Urgent warning signs requiring immediate emergency medical escalation (911 or ED).
   - **Recommended Diagnostic Workup**: Follow-up labs, imaging, or physical exam maneuvers to discuss with the attending physician.
3. Patient Context:
${profile?.age ? `- Age: ${profile.age}` : ''}
${profile?.gender ? `- Gender: ${profile.gender}` : ''}
${profile?.conditions ? `- Pre-existing Conditions: ${profile.conditions}` : ''}
${profile?.allergies ? `- Known Allergies: ${profile.allergies}` : ''}
${activeMedications && activeMedications.length > 0 ? `- Current Active Medications: ${activeMedications.join(', ')}` : ''}
Always evaluate drug interactions or contraindications with known medications and allergies.

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
- If symptoms or vitals suggest an acute medical emergency (e.g. crushing chest pain, acute respiratory distress, sudden slurred speech or facial droop, uncontrolled hemorrhage, severe anaphylaxis, SpO2 < 90%), IMMEDIATELY advise dialing emergency medical services (911).
- Include standard clinical informational disclaimer noting that RapidAid provides decision support and does not replace emergency clinical evaluation.`;

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

  const systemInstruction = BUILD_CHAT_SYSTEM_PROMPT(profile, activeMedications, vitals, acuity);

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
