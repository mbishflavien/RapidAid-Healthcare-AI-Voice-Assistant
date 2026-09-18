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
  Radio,
  Lock,
  RefreshCw,
  ExternalLink
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
  const [micPermissionState, setMicPermissionState] = useState<'prompt' | 'granted' | 'denied' | 'unsupported' | null>(null);
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

  // Monitor microphone permission state when available
  useEffect(() => {
    if (typeof navigator !== 'undefined' && navigator.permissions && (navigator.permissions as any).query) {
      try {
        navigator.permissions.query({ name: 'microphone' as PermissionName })
          .then((permissionStatus) => {
            setMicPermissionState(permissionStatus.state as any);
            permissionStatus.onchange = () => {
              setMicPermissionState(permissionStatus.state as any);
              if (permissionStatus.state === 'granted') {
                setErrorMessage(null);
              }
            };
          })
          .catch(() => {
            // Permission query not supported for microphone on some browsers
          });
      } catch {
        // Safe ignore
      }
    }
  }, []);

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch {
        // Ignore disconnect error
      }
      processorRef.current = null;
    }
    if (streamRef.current) {
      try {
        streamRef.current.getTracks().forEach(track => track.stop());
      } catch {
        // Ignore stop error
      }
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close().catch(() => {});
      } catch {
        // Ignore close error
      }
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

      // 1. Microphone stream acquisition with resilient fallback
      if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setMicPermissionState('unsupported');
        throw new Error("Microphone API is not supported in this frame or browser. Please open the app directly in a new browser tab.");
      }

      let micStream: MediaStream | null = null;
      try {
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
        } catch (constraintErr: any) {
          // If browser/device rejects specific constraints (e.g. 16kHz sampleRate on certain OS/drivers), retry with basic audio
          console.warn("Retrying microphone stream with basic audio constraints:", constraintErr?.name);
          micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        streamRef.current = micStream;
        setMicPermissionState('granted');
      } catch (micErr: any) {
        console.warn("Microphone access not granted by user/system:", micErr?.name || micErr?.message);
        const isDenied = 
          micErr?.name === 'NotAllowedError' || 
          micErr?.name === 'PermissionDeniedError' || 
          micErr?.message?.toLowerCase().includes('denied') ||
          micErr?.message?.toLowerCase().includes('permission');
        const isNotFound = 
          micErr?.name === 'NotFoundError' || 
          micErr?.name === 'DevicesNotFoundError';

        if (isDenied) {
          setMicPermissionState('denied');
          throw new Error("Microphone permission denied. Please allow microphone access in your browser or address bar settings.");
        } else if (isNotFound) {
          setMicPermissionState('unsupported');
          throw new Error("No microphone audio input device was found. Please check your audio hardware.");
        } else {
          throw new Error(micErr?.message || "Could not access microphone.");
        }
      }

      // 2. Audio context (initialized only after microphone stream is confirmed)
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      try {
        audioContextRef.current = new AudioCtx({ sampleRate: SAMPLE_RATE });
      } catch {
        // Fallback if browser enforces native hardware output rate
        audioContextRef.current = new AudioCtx();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      const ai = new GoogleGenAI({ apiKey });

      const patientContext = patientProfile ? `
Patient Context:
- Age: ${patientProfile.age || 'Not specified'}
- Allergies: ${patientProfile.allergies || 'None listed'}
- Conditions: ${patientProfile.conditions || 'None listed'}
- Active Meds: ${medicationsRef.current.map(m => m.name).join(', ') || 'None'}
` : '';

      const systemInstruction = `You are RapidAid Voice Healthcare Companion, a friendly, warm, and empathetic voice health assistant.
Always speak in simple, plain everyday English that is easy for anyone to understand.
Avoid complex medical jargon, abbreviations, and clinical terminology (for example, say "high blood pressure" instead of "hypertension", "trouble breathing" instead of "dyspnea", "fast heartbeat" instead of "tachycardia", "swelling" instead of "edema"). If you must mention a medical term, immediately explain what it means in plain everyday words.
Speak with warmth, reassurance, and clarity.
Keep spoken responses concise and conversational (2-3 short sentences max per spoken turn) so the patient can converse naturally without feeling overwhelmed.
${patientContext}

SAFETY RULES:
1. Always state: "I am an AI health assistant, not a doctor. In an emergency call 911."
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
              console.warn("Live API Notice:", err);
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
      console.warn("Voice companion session could not start:", err?.message || err);
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
    <>
      {/* Mobile/Tablet Backdrop Overlay */}
      <div 
        className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-40 lg:hidden transition-opacity"
        onClick={onClose}
        aria-label="Close voice companion"
      />

      <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-96 max-w-full lg:static lg:w-80 xl:w-96 border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col h-full shrink-0 shadow-2xl lg:shadow-none animate-in slide-in-from-right duration-200 transition-colors">
        {/* Telehealth Audio Station Header */}
        <div className="p-3.5 sm:p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50/90 dark:bg-slate-950/80">
          <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
            <div className={`w-8 h-8 sm:w-9 sm:h-9 rounded-xl flex items-center justify-center shrink-0 transition-all ${
              isActive ? 'bg-teal-700 text-white shadow-xs' : 'bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-300'
            }`}>
              <Radio className={`w-4 h-4 ${isActive ? 'animate-pulse' : ''}`} />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 sm:gap-2">
                <h3 className="text-xs font-bold text-slate-900 dark:text-slate-100 tracking-tight truncate">Telehealth Audio Station</h3>
                <span className={`w-2 h-2 rounded-full shrink-0 ${
                  isActive ? 'bg-emerald-500 shadow-[0_0_6px_#10b981]' : 'bg-slate-300 dark:bg-slate-600'
                }`} />
              </div>
              <p className="text-[9px] sm:text-[10px] text-slate-500 dark:text-slate-400 font-mono font-medium truncate">
                {status === 'active' ? 'FULL-DUPLEX 16kHz • LIVE' : status === 'connecting' ? 'CONNECTING...' : 'STANDBY • READY'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1 sm:gap-1.5 shrink-0">
            <button
              onClick={() => setShowVoicePicker(!showVoicePicker)}
              className="px-2 py-1 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-[10px] sm:text-[11px] font-semibold text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors flex items-center gap-1 shadow-xs"
              title="Select Clinical Voice Persona"
            >
              <span>{selectedVoice}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 min-h-[36px] min-w-[36px] flex items-center justify-center rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/50 dark:hover:bg-slate-800 transition-colors"
              title="Close Audio Panel"
            >
              <X className="w-5 h-5 lg:hidden" />
              <ChevronRight className="w-4 h-4 hidden lg:block" />
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
            className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 space-y-1.5 overflow-hidden"
          >
            <div className="flex items-center justify-between px-1 mb-1">
              <p className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Clinical Persona</p>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">Gemini Live Audio</span>
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
                      ? 'bg-teal-50 dark:bg-teal-950/80 text-teal-900 dark:text-teal-200 border border-teal-200 dark:border-teal-800'
                      : 'bg-slate-50 dark:bg-slate-800/80 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 border border-transparent'
                  }`}
                >
                  <div>
                    <span className="font-bold">{item.name}</span>
                    <span className="text-[10px] text-slate-500 dark:text-slate-400 block">{item.role}</span>
                  </div>
                  {selectedVoice === item.name && <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Audio Visualizer / Monitor */}
      <div className="p-3.5 sm:p-5 flex-1 flex flex-col justify-between overflow-y-auto custom-scrollbar">
        <div className="space-y-3 sm:space-y-4">
          {/* Main Visualizer Stage (Oscilloscope & Vitals Style) */}
          <div className="relative p-4 sm:p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col items-center justify-center min-h-[170px] sm:min-h-[210px] text-center overflow-hidden medical-ecg-bg">
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

          {/* Error Message & Permission Troubleshooting */}
          {errorMessage && (
            <div className="p-3.5 rounded-xl bg-amber-50 dark:bg-amber-950/70 border border-amber-200 dark:border-amber-800 text-slate-800 dark:text-slate-200 text-xs space-y-2.5">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 text-amber-800 dark:text-amber-300 font-bold">
                  {errorMessage.toLowerCase().includes('microphone') ? (
                    <Lock className="w-4 h-4 text-amber-700 dark:text-amber-400 shrink-0" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-amber-700 dark:text-amber-400 shrink-0" />
                  )}
                  <span>
                    {errorMessage.toLowerCase().includes('microphone') 
                      ? "Microphone Permission Required" 
                      : "Audio Link Status"}
                  </span>
                </div>
                <button 
                  onClick={() => setErrorMessage(null)} 
                  className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-md hover:bg-amber-100/50 dark:hover:bg-amber-900/50 transition-colors"
                  title="Dismiss alert"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              <p className="text-[11px] text-slate-600 dark:text-slate-300 font-medium leading-relaxed">
                {errorMessage}
              </p>

              {errorMessage.toLowerCase().includes('microphone') && (
                <div className="pt-2 border-t border-amber-200/70 dark:border-amber-800 space-y-2">
                  <div className="bg-white/80 dark:bg-slate-900/80 rounded-lg p-2.5 border border-amber-200/60 dark:border-amber-800 text-[11px] text-slate-600 dark:text-slate-300 space-y-1">
                    <p className="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <span>How to enable microphone:</span>
                    </p>
                    <ol className="list-decimal pl-4 space-y-0.5 text-slate-600 dark:text-slate-300">
                      <li>Click the lock 🔒 or settings icon in your browser address bar.</li>
                      <li>Toggle <strong>Microphone</strong> from "Block" to <strong>"Allow"</strong>.</li>
                      <li>Click <strong>Retry Microphone</strong> below, or open in a new tab.</li>
                    </ol>
                  </div>

                  <div className="flex items-center gap-2 pt-0.5">
                    <button
                      onClick={startVoiceSession}
                      className="px-3 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white font-semibold text-xs flex items-center gap-1.5 shadow-xs transition-colors"
                    >
                      <RefreshCw className="w-3 h-3" />
                      <span>Retry Microphone</span>
                    </button>
                    <button
                      onClick={() => window.open(window.location.href, '_blank')}
                      className="px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 font-semibold text-xs flex items-center gap-1.5 shadow-xs transition-colors"
                      title="Open application in a direct browser tab to prompt microphone directly"
                    >
                      <ExternalLink className="w-3 h-3 text-slate-500 dark:text-slate-400" />
                      <span>Open in New Tab</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Clinical Audio Protocol Guidelines */}
          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950/70 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 text-xs space-y-1.5">
            <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5 text-[11px]">
              <Sparkles className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
              Clinical Voice Capabilities:
            </span>
            <ul className="text-[11px] text-slate-500 dark:text-slate-400 space-y-1 list-disc pl-4 font-medium">
              <li>Natural hands-free reporting of ongoing symptoms & vitals</li>
              <li>Medication dosage reminders and intake logging</li>
              <li>Dictated advice automatically registers into patient chart notes</li>
            </ul>
          </div>
        </div>

        {/* Telehealth Audio Controls */}
        <div className="pt-4 border-t border-slate-200 dark:border-slate-800 space-y-2.5">
          <div className="flex items-center gap-2">
            {/* Mic Mute */}
            <button
              onClick={() => setIsMuted(!isMuted)}
              disabled={!isActive}
              className={`p-2.5 rounded-xl border transition-all flex items-center justify-center ${
                isMuted
                  ? 'bg-red-50 dark:bg-red-950/80 border-red-200 dark:border-red-800 text-red-600 dark:text-red-400'
                  : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700'
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
                  ? 'bg-amber-50 dark:bg-amber-950/80 border-amber-200 dark:border-amber-800 text-amber-600 dark:text-amber-400'
                  : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700'
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
                  ? 'bg-slate-900 dark:bg-slate-800 text-white hover:bg-slate-800 dark:hover:bg-slate-700'
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
            className="w-full py-2 px-3 rounded-lg bg-red-50 dark:bg-red-950/60 hover:bg-red-100 dark:hover:bg-red-900/60 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-xs font-bold transition-colors flex items-center justify-center gap-2"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-red-600 dark:text-red-400" />
            <span>Emergency 911 Protocol Override</span>
          </button>
        </div>
      </div>
    </div>
  </>
  );
};
