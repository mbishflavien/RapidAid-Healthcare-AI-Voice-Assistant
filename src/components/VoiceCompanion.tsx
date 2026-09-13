import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Activity, 
  AlertCircle, 
  X, 
  ChevronRight, 
  Phone, 
  ShieldAlert, 
  CheckCircle2, 
  Sparkles,
  Radio
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Transcription, HealthProfile } from '../types';
import { Medication, addMedication, deleteMedication } from '../lib/medications';

interface VoiceCompanionProps {
  isOpen: boolean;
  onClose: () => void;
  onTranscription: (transcription: Transcription) => void;
  user?: any;
  patientProfile?: HealthProfile;
  medications: Medication[];
  onMedicationsUpdated?: () => void;
}

const MODEL = "gemini-3.1-flash-live-preview";
const SAMPLE_RATE = 16000;
const VOICES = ['Puck', 'Aoede', 'Fenrir', 'Charon', 'Kore'];

export const VoiceCompanion: React.FC<VoiceCompanionProps> = ({
  isOpen,
  onClose,
  onTranscription,
  user,
  patientProfile,
  medications,
  onMedicationsUpdated,
}) => {
  const [isActive, setIsActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeakerMuted, setIsSpeakerMuted] = useState(false);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedVoice, setSelectedVoice] = useState('Puck');
  const [showVoicePicker, setShowVoicePicker] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState('Auto-Detect');
  const [userVolume, setUserVolume] = useState(0);
  const [aiVolume, setAiVolume] = useState(0);
  const [liveCaption, setLiveCaption] = useState<{ text: string; isUser: boolean } | null>(null);

  // WebAudio refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef<boolean>(false);
  const nextStartTimeRef = useRef<number>(0);
  const medicationsRef = useRef<Medication[]>(medications);

  useEffect(() => {
    medicationsRef.current = medications;
  }, [medications]);

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    nextStartTimeRef.current = 0;
    setUserVolume(0);
    setAiVolume(0);
  }, []);

  const playNextChunk = useCallback(() => {
    if (isSpeakerMuted) return;
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    if (!audioContextRef.current || audioContextRef.current.state === 'closed') return;

    isPlayingRef.current = true;
    const chunk = audioQueueRef.current.shift()!;
    const audioBuffer = audioContextRef.current.createBuffer(1, chunk.length, 24000);
    const channelData = audioBuffer.getChannelData(0);

    for (let i = 0; i < chunk.length; i++) {
      channelData[i] = chunk[i] / 32768.0;
    }

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);

    const currentTime = audioContextRef.current.currentTime;
    const startTime = Math.max(currentTime, nextStartTimeRef.current);
    source.start(startTime);
    nextStartTimeRef.current = startTime + audioBuffer.duration;

    source.onended = () => {
      playNextChunk();
    };
  }, [isSpeakerMuted]);

  const endSession = useCallback(() => {
    if (sessionRef.current) {
      try {
        sessionRef.current.close();
      } catch (e) {
        console.warn("Session close error:", e);
      }
      sessionRef.current = null;
    }
    setIsActive(false);
    setStatus('idle');
    stopAudio();
  }, [stopAudio]);

  const startVoiceSession = async () => {
    try {
      setStatus('connecting');
      setErrorMessage(null);

      // Check API Key
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("Missing Gemini API Key. Please verify settings.");
      }

      // 1. Audio context
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioCtx({ sampleRate: SAMPLE_RATE });
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // 2. Microphone stream
      let micStream: MediaStream | null = null;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });
        streamRef.current = micStream;
      } catch (micErr) {
        console.error("Mic access denied:", micErr);
        throw new Error("Microphone permission denied. Please allow microphone access.");
      }

      const ai = new GoogleGenAI({ apiKey });

      const patientContext = patientProfile ? `
Patient Context:
- Age: ${patientProfile.age || 'Not specified'}
- Allergies: ${patientProfile.allergies || 'None listed'}
- Conditions: ${patientProfile.conditions || 'None listed'}
- Active Meds: ${medicationsRef.current.map(m => m.name).join(', ') || 'None'}
` : '';

      const systemInstruction = `You are RapidAid Voice Healthcare Companion, an empathetic real-time clinical voice assistant.
Speak clearly, with natural medical cadence, reassurance, and empathy.
Keep spoken responses concise and conversational (2-4 sentences max per spoken turn) so the patient can converse naturally without long pauses.
${patientContext}

SAFETY RULES:
1. Always state: "I am an AI medical assistant, not a doctor. In an emergency call 911."
2. If patient mentions chest pain, severe bleeding, difficulty breathing, stroke symptoms, immediately tell them to dial 911 and call the 'callEmergencyServices' tool.
3. Use 'reportLanguage' when the patient speaks in another language.
4. Use 'displaySymptomAnalysis' if comprehensive symptoms are reported.
5. Use 'addMedicationReminder' or 'listMedications' when managing meds.`;

      const session = await ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: selectedVoice
              }
            }
          },
          systemInstruction: {
            parts: [{ text: systemInstruction }]
          },
          tools: [{
            functionDeclarations: [
              {
                name: "reportLanguage",
                description: "Reports the language detected from the user.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    language: { type: Type.STRING, description: "Language name (e.g. English, Spanish, Mandarin)" }
                  },
                  required: ["language"]
                }
              },
              {
                name: "callEmergencyServices",
                description: "Triggers emergency services alert for critical health events.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    reason: { type: Type.STRING, description: "Emergency justification" }
                  },
                  required: ["reason"]
                }
              },
              {
                name: "displaySymptomAnalysis",
                description: "Displays structured symptom evaluation.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    analysis: {
                      type: Type.OBJECT,
                      properties: {
                        symptoms: { type: Type.ARRAY, items: { type: Type.STRING } },
                        potentialConditions: {
                          type: Type.ARRAY,
                          items: {
                            type: Type.OBJECT,
                            properties: {
                              name: { type: Type.STRING },
                              likelihood: { type: Type.STRING },
                              description: { type: Type.STRING }
                            },
                            required: ["name", "likelihood", "description"]
                          }
                        },
                        urgency: { type: Type.STRING, enum: ["Low", "Medium", "High", "Emergency"] },
                        recommendations: { type: Type.ARRAY, items: { type: Type.STRING } }
                      },
                      required: ["symptoms", "potentialConditions", "urgency", "recommendations"]
                    }
                  },
                  required: ["analysis"]
                }
              },
              {
                name: "addMedicationReminder",
                description: "Registers a medication reminder for the user.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING },
                    dosage: { type: Type.STRING },
                    frequency: { type: Type.STRING },
                    times: { type: Type.ARRAY, items: { type: Type.STRING } }
                  },
                  required: ["name", "dosage", "frequency", "times"]
                }
              },
              {
                name: "listMedications",
                description: "Lists patient's current medication schedule.",
                parameters: { type: Type.OBJECT, properties: {} }
              },
              {
                name: "removeMedicationReminder",
                description: "Removes a medication reminder.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    name: { type: Type.STRING }
                  },
                  required: ["name"]
                }
              }
            ]
          }]
        },
        callbacks: {
          onopen: () => {
            setIsActive(true);
            setStatus('active');
          },
          onmessage: async (message: LiveServerMessage) => {
              // 1. Audio data from model
              const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
              if (base64Audio) {
                const binary = atob(base64Audio);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                  bytes[i] = binary.charCodeAt(i);
                }
                const pcmData = new Int16Array(bytes.buffer);

                // Calculate volume RMS
                let sum = 0;
                for (let i = 0; i < pcmData.length; i++) {
                  const val = pcmData[i] / 32767;
                  sum += val * val;
                }
                const rms = Math.sqrt(sum / pcmData.length);
                setAiVolume(rms);

                audioQueueRef.current.push(pcmData);
                playNextChunk();
              }

              // 2. Interruption
              if (message.serverContent?.interrupted) {
                audioQueueRef.current = [];
                nextStartTimeRef.current = 0;
              }

              // 3. User & Model text turns
              if (message.serverContent) {
                const { userContent, modelTurn } = message.serverContent as any;
                if (userContent?.parts) {
                  userContent.parts.forEach((part: any) => {
                    if (part.text) {
                      setLiveCaption({ text: part.text, isUser: true });
                      onTranscription({
                        text: part.text,
                        isUser: true,
                        timestamp: Date.now(),
                        fromVoice: true
                      });
                    }
                  });
                }
                if (modelTurn?.parts) {
                  modelTurn.parts.forEach((part: any) => {
                    if (part.text) {
                      setLiveCaption({ text: part.text, isUser: false });
                      onTranscription({
                        text: part.text,
                        isUser: false,
                        timestamp: Date.now(),
                        fromVoice: true
                      });
                    }
                  });
                }
              }

              // 4. Tool calls
              const toolCall = message.toolCall;
              if (toolCall) {
                for (const fc of toolCall.functionCalls) {
                  if (fc.name === "reportLanguage") {
                    const lang = (fc.args as any).language;
                    if (lang) setDetectedLanguage(lang);
                    sessionRef.current?.sendToolResponse({
                      functionResponses: [{
                        name: fc.name,
                        id: fc.id,
                        response: { output: `Language reported: ${lang}` }
                      }]
                    });
                  } else if (fc.name === "callEmergencyServices") {
                    window.location.href = "tel:911";
                    setErrorMessage("Emergency 911 dispatch signaled.");
                    sessionRef.current?.sendToolResponse({
                      functionResponses: [{
                        name: fc.name,
                        id: fc.id,
                        response: { output: "Emergency dialer opened." }
                      }]
                    });
                  } else if (fc.name === "displaySymptomAnalysis") {
                    const analysis = (fc.args as any).analysis;
                    onTranscription({
                      analysis,
                      isUser: false,
                      timestamp: Date.now(),
                      fromVoice: true
                    });
                    sessionRef.current?.sendToolResponse({
                      functionResponses: [{
                        name: fc.name,
                        id: fc.id,
                        response: { output: "Symptom analysis rendered in consultation record." }
                      }]
                    });
                  } else if (fc.name === "addMedicationReminder") {
                    if (user) {
                      const { name, dosage, frequency, times } = fc.args as any;
                      await addMedication(user.uid, {
                        userId: user.uid,
                        name,
                        dosage,
                        frequency,
                        times
                      });
                      if (onMedicationsUpdated) onMedicationsUpdated();
                      sessionRef.current?.sendToolResponse({
                        functionResponses: [{
                          name: fc.name,
                          id: fc.id,
                          response: { output: `Medication ${name} saved.` }
                        }]
                      });
                    }
                  } else if (fc.name === "listMedications") {
                    const medsList = medicationsRef.current.map(m => `- ${m.name}: ${m.dosage} (${m.frequency}) at ${m.times.join(', ')}`).join('\n');
                    sessionRef.current?.sendToolResponse({
                      functionResponses: [{
                        name: fc.name,
                        id: fc.id,
                        response: { output: medsList || "No active medications." }
                      }]
                    });
                  } else if (fc.name === "removeMedicationReminder") {
                    if (user) {
                      const { name } = fc.args as any;
                      const med = medicationsRef.current.find(m => m.name.toLowerCase() === name.toLowerCase());
                      if (med) {
                        await deleteMedication(user.uid, med.id);
                        if (onMedicationsUpdated) onMedicationsUpdated();
                      }
                      sessionRef.current?.sendToolResponse({
                        functionResponses: [{
                          name: fc.name,
                          id: fc.id,
                          response: { output: `Medication ${name} removed.` }
                        }]
                      });
                    }
                  }
                }
              }
            },
            onclose: () => {
              setIsActive(false);
              setStatus('idle');
              stopAudio();
            },
            onerror: (err: any) => {
              console.error("Live API Error:", err);
              setStatus('error');
              setErrorMessage("Voice link interrupted. Please try again.");
              stopAudio();
            }
          }
        });

      sessionRef.current = session;

      // Audio script processor
      if (streamRef.current && audioContextRef.current) {
        const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
        const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (isMuted) return;
          const inputData = e.inputBuffer.getChannelData(0);

          let sum = 0;
          for (let i = 0; i < inputData.length; i++) {
            sum += inputData[i] * inputData[i];
          }
          const rms = Math.sqrt(sum / inputData.length);
          setUserVolume(rms);

          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
          }

          const base64Data = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
          sessionRef.current?.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        };

        source.connect(processor);
        processor.connect(audioContextRef.current.destination);
      }
    } catch (err: any) {
      console.error("Failed to start voice companion:", err);
      setStatus('error');
      setErrorMessage(err?.message || "Could not start voice assistant.");
      stopAudio();
    }
  };

  useEffect(() => {
    return () => {
      endSession();
    };
  }, [endSession]);

  useEffect(() => {
    if (liveCaption) {
      const timer = setTimeout(() => setLiveCaption(null), 6000);
      return () => clearTimeout(timer);
    }
  }, [liveCaption]);

  if (!isOpen) return null;

  return (
    <div className="w-full lg:w-96 border-l border-slate-200 bg-white flex flex-col h-full shrink-0 z-30 shadow-xl lg:shadow-none">
      {/* Telehealth Audio Station Header */}
      <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/90">
        <div className="flex items-center gap-3">
          <div className={`w-9 h-9 rounded-xl flex items-center justify-center transition-all ${
            isActive ? 'bg-teal-700 text-white shadow-xs' : 'bg-slate-200 text-slate-600'
          }`}>
            <Radio className={`w-4 h-4 ${isActive ? 'animate-pulse' : ''}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold text-slate-900 tracking-tight">Telehealth Audio Station</h3>
              <span className={`w-2 h-2 rounded-full ${
                isActive ? 'bg-emerald-500 shadow-[0_0_6px_#10b981]' : 'bg-slate-300'
              }`} />
            </div>
            <p className="text-[10px] text-slate-500 font-mono font-medium">
              {status === 'active' ? 'FULL-DUPLEX 16kHz • LIVE' : status === 'connecting' ? 'CONNECTING PROTOCOL...' : 'STANDBY • READY'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowVoicePicker(!showVoicePicker)}
            className="px-2.5 py-1 rounded-md bg-white border border-slate-200 text-[11px] font-semibold text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-1 shadow-xs"
            title="Select Clinical Voice Persona"
          >
            <span>{selectedVoice}</span>
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-700 hover:bg-slate-200/50 transition-colors"
            title="Collapse Audio Panel"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Voice Persona Dropdown */}
      <AnimatePresence>
        {showVoicePicker && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-b border-slate-200 bg-white p-3 space-y-1.5 overflow-hidden"
          >
            <div className="flex items-center justify-between px-1 mb-1">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Clinical Persona</p>
              <span className="text-[10px] text-slate-400 font-mono">Gemini Live Audio</span>
            </div>
            <div className="grid grid-cols-1 gap-1">
              {[
                { name: 'Aoede', role: 'Triage Specialist (Calm & Precise)' },
                { name: 'Puck', role: 'General Practice (Direct & Attentive)' },
                { name: 'Fenrir', role: 'Emergency Protocols (Fast & Authoritative)' },
                { name: 'Charon', role: 'Trauma & Critical Care (Steady & Reassuring)' },
                { name: 'Kore', role: 'Pediatric & Family Consultation (Gentle)' }
              ].map(item => (
                <button
                  key={item.name}
                  onClick={() => {
                    setSelectedVoice(item.name);
                    setShowVoicePicker(false);
                  }}
                  className={`px-3 py-2 rounded-lg text-xs font-medium text-left flex items-center justify-between transition-colors ${
                    selectedVoice === item.name
                      ? 'bg-teal-50 text-teal-900 border border-teal-200'
                      : 'bg-slate-50 text-slate-700 hover:bg-slate-100 border border-transparent'
                  }`}
                >
                  <div>
                    <span className="font-bold">{item.name}</span>
                    <span className="text-[10px] text-slate-500 block">{item.role}</span>
                  </div>
                  {selectedVoice === item.name && <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Audio Visualizer / Monitor */}
      <div className="p-5 flex-1 flex flex-col justify-between overflow-y-auto custom-scrollbar">
        <div className="space-y-4">
          {/* Main Visualizer Stage (Oscilloscope & Vitals Style) */}
          <div className="relative p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center min-h-[220px] text-center overflow-hidden medical-ecg-bg">
            {/* Top Telemetry Overlay */}
            <div className="absolute top-2.5 inset-x-3 flex items-center justify-between text-[10px] font-mono text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-emerald-400 animate-ping' : 'bg-slate-600'}`} />
                {isActive ? 'CHANNEL: 16.0 kHz PCM' : 'CHANNEL: INACTIVE'}
              </span>
              <span>{isActive ? 'LATENCY: <120ms' : 'OFFLINE'}</span>
            </div>

            {isActive ? (
              <>
                <div className="relative my-4">
                  {/* Concentric Telemetry Ripple */}
                  <motion.div 
                    animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.3, 0.1] }}
                    transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                    className="absolute -inset-4 rounded-full bg-teal-500 blur-md"
                  />
                  <div className="w-20 h-20 rounded-full bg-slate-800 border border-teal-400/40 shadow-xl flex items-center justify-center relative z-10">
                    <Activity className="w-8 h-8 text-teal-400" />
                  </div>
                </div>

                {/* Animated Clinical Audio Spectrum */}
                <div className="flex items-end justify-center gap-1 h-10 w-full px-2">
                  {Array.from({ length: 22 }).map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{
                        height: Math.max(4, Math.min(38, (aiVolume > 0.01 ? aiVolume * 55 : userVolume * 55) * (Math.sin(i / 1.8) + 1.2)))
                      }}
                      transition={{ duration: 0.08 }}
                      className={`w-1.5 rounded-xs ${
                        aiVolume > 0.01 ? 'bg-teal-400' : 'bg-emerald-400'
                      }`}
                    />
                  ))}
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <p className="text-xs font-mono font-medium text-slate-200">
                    {aiVolume > 0.01 ? "CLINICAL AI SPEAKING..." : isMuted ? "MIC MUTED" : "LISTENING TO PATIENT..."}
                  </p>
                </div>
              </>
            ) : (
              <div className="py-4 flex flex-col items-center">
                <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center mb-3 text-slate-400">
                  <Mic className="w-7 h-7 text-teal-400" />
                </div>
                <h4 className="text-sm font-bold text-white tracking-tight">Telehealth Audio Channel</h4>
                <p className="text-xs text-slate-400 font-medium max-w-[210px] mt-1 leading-relaxed">
                  Start hands-free voice consultation to speak directly with the clinical decision model.
                </p>
              </div>
            )}

            {/* Language & Protocol Tag */}
            <div className="mt-3 inline-flex items-center gap-2 px-2.5 py-0.5 rounded bg-slate-800/90 border border-slate-700 text-[10px] font-mono text-slate-300">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400" />
              <span>LANG: {detectedLanguage.toUpperCase()}</span>
              <span className="text-slate-500">•</span>
              <span>ENC: AES-256</span>
            </div>
          </div>

          {/* Clinical Dictation Tape (Live Subtitle Feed) */}
          {liveCaption && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3.5 rounded-xl bg-slate-900 text-white border border-slate-800 shadow-sm text-xs leading-relaxed"
            >
              <div className="flex items-center justify-between pb-1 mb-1 border-b border-slate-800">
                <span className="text-[9px] uppercase tracking-wider font-mono font-bold text-teal-400">
                  {liveCaption.isUser ? "PATIENT DICTATION" : "CLINICAL AI ADVICE"}
                </span>
                <span className="text-[9px] font-mono text-slate-400">
                  {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
              </div>
              <p className="font-mono text-[11px] text-slate-200">"{liveCaption.text}"</p>
            </motion.div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-500" />
              <div className="flex-1">
                <span className="font-bold block">Audio Protocol Alert</span>
                <span className="font-medium text-[11px]">{errorMessage}</span>
              </div>
              <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-600">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Clinical Audio Protocol Guidelines */}
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-600 text-xs space-y-1.5">
            <span className="font-bold text-slate-800 flex items-center gap-1.5 text-[11px]">
              <Sparkles className="w-3.5 h-3.5 text-teal-600" />
              Clinical Voice Capabilities:
            </span>
            <ul className="text-[11px] text-slate-500 space-y-1 list-disc pl-4 font-medium">
              <li>Natural hands-free reporting of ongoing symptoms & vitals</li>
              <li>Medication dosage reminders and intake logging</li>
              <li>Dictated advice automatically registers into patient chart notes</li>
            </ul>
          </div>
        </div>

        {/* Telehealth Audio Controls */}
        <div className="pt-4 border-t border-slate-200 space-y-2.5">
          <div className="flex items-center gap-2">
            {/* Mic Mute */}
            <button
              onClick={() => setIsMuted(!isMuted)}
              disabled={!isActive}
              className={`p-2.5 rounded-xl border transition-all flex items-center justify-center ${
                isMuted
                  ? 'bg-red-50 border-red-200 text-red-600'
                  : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
              } disabled:opacity-40 disabled:pointer-events-none`}
              title={isMuted ? "Unmute Microphone" : "Mute Microphone"}
            >
              {isMuted ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>

            {/* Speaker Output Mute */}
            <button
              onClick={() => setIsSpeakerMuted(!isSpeakerMuted)}
              disabled={!isActive}
              className={`p-2.5 rounded-xl border transition-all flex items-center justify-center ${
                isSpeakerMuted
                  ? 'bg-amber-50 border-amber-200 text-amber-600'
                  : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
              } disabled:opacity-40 disabled:pointer-events-none`}
              title={isSpeakerMuted ? "Mute Speaker Output" : "Unmute Speaker Output"}
            >
              {isSpeakerMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>

            {/* Primary Start / Terminate Call Button */}
            <button
              onClick={isActive ? endSession : startVoiceSession}
              disabled={status === 'connecting'}
              className={`flex-1 py-3 px-4 rounded-xl font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-xs ${
                isActive
                  ? 'bg-slate-900 text-white hover:bg-slate-800'
                  : 'bg-teal-700 text-white hover:bg-teal-800'
              } disabled:opacity-50`}
            >
              {status === 'connecting' ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : isActive ? (
                <>
                  <Activity className="w-4 h-4 text-teal-400" />
                  <span>Disconnect Audio</span>
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4" />
                  <span>Connect Audio Link</span>
                </>
              )}
            </button>
          </div>

          {/* Emergency SOS Protocol Button */}
          <button
            onClick={() => {
              if (window.confirm("CRITICAL EMERGENCY OVERRIDE: Do you require immediate 911 dispatch?")) {
                window.location.href = "tel:911";
              }
            }}
            className="w-full py-2 px-3 rounded-lg bg-red-50 hover:bg-red-100 border border-red-200 text-red-700 text-xs font-bold transition-colors flex items-center justify-center gap-2"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
            <span>Emergency 911 Protocol Override</span>
          </button>
        </div>
      </div>
    </div>
  );
};
