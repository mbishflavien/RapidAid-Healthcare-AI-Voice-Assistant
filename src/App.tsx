/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { Mic, MicOff, Activity, Stethoscope, AlertCircle, Info, X, Volume2, VolumeX, Globe, ExternalLink, BookOpen, Phone, Trash2, Download, Send, CheckCircle2, Clock, ShieldAlert } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Constants ---
const MODEL = "gemini-3.1-flash-live-preview";
const SAMPLE_RATE = 16000;

const MEDICAL_RESOURCES = [
  {
    name: "World Health Organization (WHO)",
    description: "Global health information, guidelines, and emergency updates.",
    url: "https://www.who.int",
    category: "Global Health"
  },
  {
    name: "Mayo Clinic",
    description: "Comprehensive health information on conditions, symptoms, and treatments.",
    url: "https://www.mayoclinic.org",
    category: "Medical Reference"
  },
  {
    name: "CDC (Centers for Disease Control)",
    description: "Public health information, disease tracking, and prevention guidelines.",
    url: "https://www.cdc.gov",
    category: "Public Health"
  },
  {
    name: "NHS Health A-Z",
    description: "Extensive library of health conditions, symptoms, and treatments.",
    url: "https://www.nhs.uk/conditions/",
    category: "Patient Care"
  },
  {
    name: "MedlinePlus",
    description: "Trusted health information from the US National Library of Medicine.",
    url: "https://medlineplus.gov",
    category: "Health Education"
  },
  {
    name: "National Institutes of Health (NIH)",
    description: "Leading medical research and health information resource.",
    url: "https://www.nih.gov",
    category: "Research"
  }
];
const SYSTEM_INSTRUCTION = `You are RapidAid, a real-time voice healthcare assistant designed for immediate, high-accuracy support.
Your goal is to provide accurate, helpful, and immediate medical information with a calm, professional, and natural tone.
VOICE CLARITY AND INTONATION:
- Use natural prosody and intonation. Avoid sounding robotic.
- Speak clearly and at a moderate pace.
- Use appropriate pauses and emphasis to sound more human and empathetic.
MULTI-LANGUAGE SUPPORT:
- You are capable of understanding and responding in multiple languages.
- Detect the user's language automatically and respond in the same language.
- IMPORTANT: Whenever you detect a language or the language changes, you MUST call the 'reportLanguage' tool with the name of the language (e.g., "English", "Spanish", "French").
- Maintain the same professional healthcare persona regardless of the language used.
SYMPTOM CHECKER CAPABILITY:
- When the user reports symptoms, ask clarifying questions (duration, severity, triggers).
- Once you have enough information, use the 'displaySymptomAnalysis' tool to provide a structured, detailed summary.
- The summary should include potential conditions (with likelihood and brief descriptions), an urgency level, and specific next steps.
- DO NOT use the tool prematurely; gather enough context first.
CRITICAL SAFETY RULES:
1. Always start or end with a disclaimer: "I am an AI assistant, not a doctor. This is for informational purposes only." (Translate this disclaimer to the user's language).
2. If the user mentions symptoms of a life-threatening emergency (chest pain, severe bleeding, difficulty breathing, stroke symptoms), immediately tell them to call emergency services (e.g., 911). You can trigger this automatically by calling the 'callEmergencyServices' tool.
3. Be concise, conversational, and rapid in your delivery.
4. If you are unsure, advise the user to consult a licensed medical professional.
5. Do not prescribe medication or give definitive diagnoses.`;

// --- Types ---
interface SymptomAnalysis {
  symptoms: string[];
  potentialConditions: { name: string; likelihood: string; description: string; }[];
  urgency: 'Low' | 'Medium' | 'High' | 'Emergency';
  recommendations: string[];
  disclaimer?: string;
}

interface Transcription {
  text?: string;
  analysis?: SymptomAnalysis;
  isUser: boolean;
  timestamp: number;
}

export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcriptions, setTranscriptions] = useState<Transcription[]>(() => {
    const saved = localStorage.getItem('rapidaid_history');
    return saved ? JSON.parse(saved) : [];
  });
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<string>('Puck');
  const [showVoiceMenu, setShowVoiceMenu] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState<string>('Detecting...');
  const [speechRate] = useState<number>(1.5);
  const [isConfigMissing, setIsConfigMissing] = useState(false);
  const [showResources, setShowResources] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [userVolume, setUserVolume] = useState(0);
  const [aiVolume, setAiVolume] = useState(0);
  const [liveCaption, setLiveCaption] = useState<{ text: string, isUser: boolean } | null>(null);

  const voices = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'];

  // Refs for audio and session
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const nextStartTimeRef = useRef<number>(0);
  const speechRateRef = useRef<number>(1.5);
  const transcriptionEndRef = useRef<HTMLDivElement>(null);

  // --- Audio Handling ---

  const stopAudio = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    nextStartTimeRef.current = 0;
    audioQueueRef.current = [];
  }, []);

  const playNextChunk = useCallback(async () => {
    if (!audioContextRef.current || audioQueueRef.current.length === 0) return;

    const pcmData = audioQueueRef.current.shift()!;
    
    // Convert Int16 to Float32
    const float32Data = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Data[i] = pcmData[i] / 32768.0;
    }

    const buffer = audioContextRef.current.createBuffer(1, float32Data.length, SAMPLE_RATE);
    buffer.getChannelData(0).set(float32Data);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.playbackRate.value = speechRateRef.current;
    source.connect(audioContextRef.current.destination);
    
    // Precise scheduling for gapless playback
    const now = audioContextRef.current.currentTime;
    if (nextStartTimeRef.current < now) {
      nextStartTimeRef.current = now + 0.05; // Initial buffer
    }
    
    source.start(nextStartTimeRef.current);
    nextStartTimeRef.current += buffer.duration / speechRateRef.current;
    
    // Recursively schedule all available chunks
    if (audioQueueRef.current.length > 0) {
      playNextChunk();
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setUserVolume(v => Math.max(0, v * 0.8));
      setAiVolume(v => Math.max(0, v * 0.8));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  const startSession = async (initialText?: string) => {
    try {
      // Priority: process.env (Vite define) -> import.meta.env.VITE_GEMINI_API_KEY -> process.env.API_KEY
      const apiKey = process.env.GEMINI_API_KEY || (import.meta as any).env?.VITE_GEMINI_API_KEY || (process.env as any).API_KEY;
      
      if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey === '') {
        setIsConfigMissing(true);
        setErrorMessage("Gemini API Key is missing. Please add GEMINI_API_KEY to your environment variables.");
        return;
      }

      setStatus('connecting');
      setErrorMessage(null);

      // 1. Initialize Audio Context (always needed for output)
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      
      // 2. Try to get Microphone (optional for text-only)
      let micAvailable = false;
      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
        micAvailable = true;
      } catch (micErr) {
        console.warn("Microphone access denied, proceeding in text-only mode:", micErr);
        // Don't set error message here, just log it. We'll show a warning if they try to unmute.
      }
      
      // 3. Initialize Gemini Live
      const ai = new GoogleGenAI({ apiKey });
      
      const session = await ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } },
          },
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: [
            {
              functionDeclarations: [
                {
                  name: "reportLanguage",
                  description: "Reports the detected language being used in the conversation.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {
                      language: {
                        type: Type.STRING,
                        description: "The name of the detected language (e.g., 'English', 'Spanish').",
                      },
                    },
                    required: ["language"],
                  },
                },
                {
                  name: "callEmergencyServices",
                  description: "Initiates a call to emergency services (e.g., 911). Use this ONLY when the user is in a life-threatening emergency.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {},
                  },
                },
                {
                  name: "displaySymptomAnalysis",
                  description: "Displays a structured medical symptom analysis to the user. Use this when you have sufficient information to provide a detailed summary of potential issues and recommended actions.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {
                      analysis: {
                        type: Type.OBJECT,
                        properties: {
                          symptoms: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of symptoms identified." },
                          potentialConditions: {
                            type: Type.ARRAY,
                            items: {
                              type: Type.OBJECT,
                              properties: {
                                name: { type: Type.STRING, description: "Name of the potential condition." },
                                likelihood: { type: Type.STRING, description: "Likelihood level (e.g., Low, Moderate, High)." },
                                description: { type: Type.STRING, description: "Brief description of the condition and why it matches the symptoms." }
                              },
                              required: ["name", "likelihood", "description"]
                            },
                            description: "Possible conditions based on the symptoms."
                          },
                          urgency: { type: Type.STRING, enum: ["Low", "Medium", "High", "Emergency"], description: "The recommended level of urgency for seeking care." },
                          recommendations: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Specific next steps or self-care advice." },
                          disclaimer: { type: Type.STRING, description: "A relevant medical disclaimer." }
                        },
                        required: ["symptoms", "potentialConditions", "urgency", "recommendations"]
                      }
                    },
                    required: ["analysis"]
                  }
                }
              ],
            },
          ],
          outputAudioTranscription: {},
          inputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus('active');
            setIsActive(true);
            
            // Start sending audio if mic is available
            if (micAvailable && streamRef.current) {
              const source = audioContextRef.current!.createMediaStreamSource(streamRef.current!);
              const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
              processorRef.current = processor as any;

              processor.onaudioprocess = (e) => {
                if (isMuted) return;
                const inputData = e.inputBuffer.getChannelData(0);
                
                // Calculate volume for visualization
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) {
                  sum += inputData[i] * inputData[i];
                }
                const rms = Math.sqrt(sum / inputData.length);
                setUserVolume(rms);

                // Convert Float32 to Int16
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                  pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
                }
                
                // Send to Gemini
                const base64Data = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
                session.sendRealtimeInput({
                  audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
                });
              };

              source.connect(processor);
              processor.connect(audioContextRef.current!.destination);
            }

            // Send initial text if provided
            if (initialText) {
              session.sendRealtimeInput({ text: initialText });
            }
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle Audio Output
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              const binaryString = atob(audioData);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }
              const pcmData = new Int16Array(bytes.buffer);
              
              // Calculate AI volume for visualization
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

            // Handle Interruption
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              nextStartTimeRef.current = 0;
            }

            // Handle Transcriptions
            if (message.serverContent) {
              const { userContent, modelTurn } = message.serverContent as any;
              
              if (userContent?.parts) {
                userContent.parts.forEach(part => {
                  if (part.text) {
                    setLiveCaption({ text: part.text, isUser: true });
                    setTranscriptions(prev => {
                      // Avoid duplicates from manual text send or rapid chunks
                      const last = prev[prev.length - 1];
                      if (last && last.isUser && (Date.now() - last.timestamp < 2000)) {
                        // Append to last if it's the same turn (approximate)
                        const updated = [...prev];
                        updated[updated.length - 1] = { ...last, text: last.text + " " + part.text, timestamp: Date.now() };
                        return updated;
                      }
                      return [...prev, { text: part.text, isUser: true, timestamp: Date.now() }];
                    });
                  }
                });
              }

              if (modelTurn?.parts) {
                modelTurn.parts.forEach(part => {
                  if (part.text) {
                    setLiveCaption({ text: part.text, isUser: false });
                    setTranscriptions(prev => {
                      const last = prev[prev.length - 1];
                      if (last && !last.isUser && (Date.now() - last.timestamp < 3000)) {
                        // Append to last if it's the same turn
                        const updated = [...prev];
                        updated[updated.length - 1] = { ...last, text: last.text + " " + part.text, timestamp: Date.now() };
                        return updated;
                      }
                      return [...prev, { text: part.text, isUser: false, timestamp: Date.now() }];
                    });
                  }
                });
              }
            }

            // Handle Tool Calls (Language Detection)
            const toolCall = message.toolCall;
            if (toolCall) {
              for (const fc of toolCall.functionCalls) {
                if (fc.name === "reportLanguage") {
                  const lang = (fc.args as any).language;
                  if (lang) {
                    setDetectedLanguage(lang);
                  }
                  
                  // Send response back to acknowledge tool call
                  session.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Language reported successfully." }
                    }]
                  });
                } else if (fc.name === "callEmergencyServices") {
                  window.location.href = "tel:911";
                  setErrorMessage("Emergency call initiated.");
                  
                  session.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Emergency services call initiated." }
                    }]
                  });
                } else if (fc.name === "displaySymptomAnalysis") {
                  const analysis = (fc.args as any).analysis;
                  setTranscriptions(prev => [...prev, { analysis, isUser: false, timestamp: Date.now() }]);
                  
                  session.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Symptom analysis displayed to user." }
                    }]
                  });
                }
              }
            }
          },
          onclose: () => {
            setIsActive(false);
            setStatus('idle');
            stopAudio();
            setUserVolume(0);
            setAiVolume(0);
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setErrorMessage("Connection lost. Please try again.");
            setStatus('error');
            stopAudio();
          }
        }
      });

      sessionRef.current = session;
    } catch (err) {
      console.error("Failed to start session:", err);
      setErrorMessage("Could not access microphone or connect to server.");
      setStatus('error');
      stopAudio();
    }
  };

  const endSession = () => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    setIsActive(false);
    setStatus('idle');
    stopAudio();
    setUserVolume(0);
    setAiVolume(0);
  };

  const handleSendText = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const text = textInput.trim();
    if (!text) return;

    if (!isActive || !sessionRef.current) {
      // Start session with initial text
      setTextInput('');
      setTranscriptions(prev => [...prev, { 
        text, 
        isUser: true, 
        timestamp: Date.now() 
      }]);
      await startSession(text);
      return;
    }

    sessionRef.current.sendRealtimeInput({
      text
    });
    
    // Add to transcriptions locally for immediate feedback
    setTranscriptions(prev => [...prev, { 
      text, 
      isUser: true, 
      timestamp: Date.now() 
    }]);
    
    setTextInput('');
  };

  // Auto-scroll transcriptions
  useEffect(() => {
    transcriptionEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    localStorage.setItem('rapidaid_history', JSON.stringify(transcriptions));
  }, [transcriptions]);

  useEffect(() => {
    if (liveCaption) {
      const timer = setTimeout(() => setLiveCaption(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [liveCaption]);

  const clearHistory = () => {
    if (window.confirm("Are you sure you want to clear your conversation history?")) {
      setTranscriptions([]);
      localStorage.removeItem('rapidaid_history');
    }
  };

  const downloadTranscript = () => {
    if (transcriptions.length === 0) return;
    
    const content = transcriptions.map(t => {
      const role = t.isUser ? "User" : "RapidAid";
      const time = new Date(t.timestamp).toLocaleTimeString();
      
      if (t.analysis) {
        const symptoms = t.analysis.symptoms.join(', ');
        const conditions = t.analysis.potentialConditions.map(c => `${c.name} (${c.likelihood}): ${c.description}`).join('\n- ');
        const recs = t.analysis.recommendations.join('\n- ');
        return `[${time}] ${role} [SYMPTOM ANALYSIS]:\nSymptoms: ${symptoms}\nUrgency: ${t.analysis.urgency}\n\nPotential Conditions:\n- ${conditions}\n\nRecommendations:\n- ${recs}`;
      }
      
      return `[${time}] ${role}: ${t.text}`;
    }).join('\n\n');
    
    const blob = new Blob([`RapidAid Health Consultation Transcript\nGenerated on: ${new Date().toLocaleString()}\n\n${content}`], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `RapidAid_Transcript_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.5;
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="min-h-screen bg-[#08090A] text-[#E4E4E7] font-sans selection:bg-[#10B981]/30 overflow-hidden">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-[#10B981]/5 blur-[140px] rounded-full animate-pulse" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-[#3B82F6]/5 blur-[140px] rounded-full animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-[30%] left-[40%] w-[30%] h-[30%] bg-[#10B981]/3 blur-[100px] rounded-full" />
      </div>

      {/* Header */}
      <header className="relative z-30 flex items-center justify-between px-8 py-5 border-b border-white/[0.03] glass-morphism">
        <div className="flex items-center gap-4">
          <div className="relative group">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-[#10B981] to-[#059669] flex items-center justify-center shadow-lg shadow-[#10B981]/20 overflow-hidden transition-transform group-hover:scale-105">
              <motion.div
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              >
                <Stethoscope className="text-white w-6 h-6 relative z-10" />
              </motion.div>
              <div className="absolute inset-0 bg-gradient-to-tr from-white/20 to-transparent" />
            </div>
            <div className="absolute -inset-1 bg-[#10B981]/20 blur-md rounded-2xl -z-10 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">RapidAid</h1>
            <div className="flex items-center gap-2">
              <span className="w-1 h-1 rounded-full bg-[#10B981]" />
              <p className="text-[10px] text-white/40 font-bold uppercase tracking-[0.2em] leading-none">Safe Voice Intelligence</p>
            </div>
          </div>
        </div>
        
          <div className="flex items-center gap-3">
            {transcriptions.length > 0 && (
              <div className="flex items-center gap-2 pr-2 border-r border-white/5 mr-2">
                <button 
                  onClick={downloadTranscript}
                  className="p-2 rounded-xl bg-white/[0.03] border border-white/[0.06] text-white/50 hover:text-[#10B981] hover:bg-[#10B981]/10 transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
                  title="Export Transcript"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span className="hidden lg:inline">Export</span>
                </button>
                
                <button 
                  onClick={clearHistory}
                  className="p-2 rounded-xl bg-white/[0.03] border border-white/[0.06] text-white/50 hover:text-red-400 hover:bg-red-400/10 transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
                  title="Clear History"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span className="hidden lg:inline">Clear</span>
                </button>
              </div>
            )}

            <button 
              onClick={() => setShowResources(true)}
              className="p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] text-white/60 hover:text-white hover:bg-white/10 transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
            >
              <BookOpen className="w-4 h-4" />
              <span className="hidden md:inline">Health Library</span>
            </button>

            {isActive && (
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 px-3 py-2 rounded-xl bg-[#10B981]/10 border border-[#10B981]/20 text-[#10B981] text-[10px] font-bold uppercase tracking-wider"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-pulse" />
                {detectedLanguage}
              </motion.div>
            )}

            <div className="relative">
              <button 
                onClick={() => setShowVoiceMenu(!showVoiceMenu)}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] text-[10px] font-bold uppercase tracking-wider hover:bg-white/10 transition-all"
              >
                <Volume2 className="w-4 h-4 text-[#10B981]" />
                {selectedVoice}
              </button>
              <AnimatePresence>
                {showVoiceMenu && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute top-full right-0 mt-3 w-40 bg-[#121417] border border-white/10 rounded-2xl overflow-hidden shadow-2xl z-50 p-1.5"
                  >
                    {voices.map(voice => (
                      <button
                        key={voice}
                        onClick={() => {
                          setSelectedVoice(voice);
                          setShowVoiceMenu(false);
                        }}
                        className={`w-full text-left px-4 py-2.5 text-[11px] font-medium rounded-xl transition-colors ${selectedVoice === voice ? 'text-[#10B981] bg-[#10B981]/10' : 'text-white/50 hover:bg-white/5'}`}
                      >
                        {voice}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

        <div className="flex items-center gap-3 pr-2">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full glass-morphism scale-90">
            <div className={`w-2 h-2 rounded-full ${
              status === 'active' ? 'bg-[#10B981] shadow-[0_0_8px_rgba(16,185,129,0.5)] animate-pulse' : 
              status === 'connecting' ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.5)] animate-pulse' : 
              'bg-white/20'
            }`} />
            <span className="text-[10px] font-bold uppercase tracking-widest text-white/60">
              {status === 'active' ? 'System Live' : status === 'connecting' ? 'Calibrating' : 'Standby'}
            </span>
          </div>
        </div>
      </header>

      <main className="relative z-20 max-w-5xl mx-auto px-8 py-10 h-[calc(100vh-84px)] flex flex-col">
        {/* Medical Disclaimer Banner */}
        <div className="mb-10 p-5 rounded-3xl bg-amber-500/[0.03] border border-amber-500/10 flex gap-4 items-center glass-morphism">
          <div className="w-10 h-10 rounded-2xl bg-amber-500/10 flex items-center justify-center shrink-0">
            <AlertCircle className="w-5 h-5 text-amber-500/70" />
          </div>
          <p className="text-sm text-amber-500/90 leading-relaxed font-medium">
            RapidAid provides health guidance based on AI intelligence. <span className="text-white/90">It is not a replacement for clinical advice.</span> In critical situations, contact medical professionals immediately.
          </p>
        </div>

        {/* Transcription Area */}
        <div className="flex-1 overflow-y-auto mb-10 space-y-8 pr-6 custom-scrollbar">
          {isConfigMissing ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-6 p-8 rounded-3xl bg-red-500/5 border border-red-500/10">
              <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center">
                <AlertCircle className="w-10 h-10 text-red-500" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white mb-2">Configuration Required</h2>
                <p className="text-sm text-white/60 max-w-md mx-auto leading-relaxed">
                  To use RapidAid, you must add your <span className="text-white font-mono">GEMINI_API_KEY</span> to your environment variables.
                </p>
                <div className="mt-4 p-4 rounded-2xl bg-white/5 border border-white/10 text-left">
                  <p className="text-[10px] text-white/40 uppercase tracking-widest font-bold mb-2">Setup Instructions:</p>
                  <ul className="text-xs text-white/60 space-y-2 list-disc pl-4">
                    <li><strong>Vercel:</strong> Project Settings → Environment Variables → Add <code className="text-[#10B981]">GEMINI_API_KEY</code></li>
                    <li><strong>AI Studio:</strong> Use the "Secrets" panel in the sidebar to add your key.</li>
                    <li><strong>Local:</strong> Create a <code className="text-[#10B981]">.env</code> file with <code className="text-[#10B981]">GEMINI_API_KEY=your_key</code></li>
                  </ul>
                </div>
              </div>
              <div className="flex flex-col gap-3 w-full max-w-sm">
                <a 
                  href="https://aistudio.google.com/app/apikey" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-6 py-3 rounded-xl bg-white/5 border border-white/10 text-sm font-medium hover:bg-white/10 transition-all"
                >
                  Get Gemini API Key
                </a>
                <div className="text-[10px] text-white/30 uppercase tracking-widest font-bold">
                  Vercel Setup: Settings → Environment Variables
                </div>
              </div>
            </div>
          ) : transcriptions.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-4 opacity-40">
              <Activity className="w-12 h-12 mb-2" />
              <h2 className="text-xl font-medium">How can I help you today?</h2>
              <p className="max-w-xs text-sm">Tap the microphone to start a real-time health consultation.</p>
            </div>
          ) : (
            <div className="space-y-8 pb-32">
              {transcriptions.map((t, i) => (
                <motion.div
                  key={t.timestamp + i}
                  initial={{ opacity: 0, y: 15, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.4, ease: [0.21, 0, 0.07, 1] }}
                  className={`flex ${t.isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`group relative max-w-[85%] transition-all ${
                    t.analysis 
                      ? 'w-full' 
                      : t.isUser 
                        ? 'p-5 rounded-[2rem] bg-gradient-to-br from-[#10B981] to-[#059669] text-white shadow-lg shadow-[#10B981]/10 rounded-tr-lg' 
                        : 'p-5 rounded-[2rem] glass-morphism text-white/90 shadow-sm rounded-tl-lg'
                  }`}>
                    {t.analysis ? (
                      <div className="w-full glass-morphism rounded-[2.5rem] overflow-hidden border border-white/10 safe-glow">
                        <div className={`p-6 flex items-center justify-between border-b border-white/5 ${
                          t.analysis.urgency === 'Emergency' ? 'bg-red-500/10' :
                          t.analysis.urgency === 'High' ? 'bg-amber-500/10' :
                          'bg-[#10B981]/10'
                        }`}>
                          <div className="flex items-center gap-4">
                            <div className={`p-3 rounded-2xl ${
                              t.analysis.urgency === 'Emergency' ? 'bg-red-500/20 text-red-500' :
                              t.analysis.urgency === 'High' ? 'bg-amber-500/20 text-amber-500' :
                              'bg-[#10B981]/20 text-[#10B981]'
                            }`}>
                              <ShieldAlert className="w-6 h-6" />
                            </div>
                            <div>
                              <h3 className="text-lg font-bold text-white tracking-tight">Symptom Analysis</h3>
                              <p className="text-[10px] font-bold uppercase tracking-widest text-white/40">Urgency: {t.analysis.urgency}</p>
                            </div>
                          </div>
                          <div className={`px-4 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest ${
                            t.analysis.urgency === 'Emergency' ? 'bg-red-500 text-white' :
                            t.analysis.urgency === 'High' ? 'bg-amber-500 text-white' :
                            'bg-[#10B981] text-white'
                          }`}>
                            {t.analysis.urgency === 'Low' ? 'Routine' : t.analysis.urgency}
                          </div>
                        </div>

                        <div className="p-8 space-y-8">
                          <div>
                            <p className="text-[10px] font-bold uppercase tracking-widest text-[#10B981] mb-3">Identified Symptoms</p>
                            <div className="flex flex-wrap gap-2">
                              {t.analysis.symptoms.map((s, idx) => (
                                <span key={idx} className="px-3 py-1.5 rounded-xl bg-white/5 border border-white/5 text-xs text-white/70 font-medium">
                                  {s}
                                </span>
                              ))}
                            </div>
                          </div>

                          <div>
                            <p className="text-[10px] font-bold uppercase tracking-widest text-[#10B981] mb-4">Potential Conditions</p>
                            <div className="space-y-4">
                              {t.analysis.potentialConditions.map((c, idx) => (
                                <div key={idx} className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-colors">
                                  <div className="flex items-center justify-between mb-2">
                                    <h4 className="font-bold text-white leading-none">{c.name}</h4>
                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${
                                      c.likelihood === 'High' ? 'bg-red-500/10 text-red-400' :
                                      c.likelihood === 'Moderate' ? 'bg-amber-500/10 text-amber-400' :
                                      'bg-[#10B981]/10 text-[#10B981]'
                                    }`}>
                                      {c.likelihood} Likelihood
                                    </span>
                                  </div>
                                  <p className="text-sm text-white/50 leading-relaxed">{c.description}</p>
                                </div>
                              ))}
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4 border-t border-white/5">
                            <div>
                              <div className="flex items-center gap-2 mb-4">
                                <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
                                <p className="text-[10px] font-bold uppercase tracking-widest text-[#10B981]">Recommendations</p>
                              </div>
                              <ul className="space-y-3">
                                {t.analysis.recommendations.map((r, idx) => (
                                  <li key={idx} className="flex gap-3 text-sm text-white/70 leading-relaxed font-medium">
                                    <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]/40 mt-1.5 shrink-0" />
                                    {r}
                                  </li>
                                ))}
                              </ul>
                            </div>
                            {t.analysis.disclaimer && (
                              <div className="p-5 rounded-2xl bg-amber-500/[0.03] border border-amber-500/10">
                                <div className="flex items-center gap-2 mb-3">
                                  <Info className="w-4 h-4 text-amber-500/60" />
                                  <p className="text-[10px] font-bold uppercase tracking-widest text-amber-500/60">AI Disclaimer</p>
                                </div>
                                <p className="text-xs text-amber-500/70 leading-relaxed italic">
                                  {t.analysis.disclaimer}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        <p className="text-[15px] leading-[1.6] font-medium">{t.text}</p>
                        <div className={`mt-3 flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300`}>
                          <span className="text-[10px] font-bold text-white/30 uppercase tracking-widest">
                            {new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          {!t.isUser && t.text && (
                            <button 
                              onClick={() => speakText(t.text!)}
                              className="p-1.5 rounded-lg bg-white/5 text-white/40 hover:text-white hover:bg-white/10 transition-all"
                              title="Read aloud"
                            >
                              <Volume2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          )}
          <div ref={transcriptionEndRef} />
        </div>

        {/* Controls Area */}
        <div className="relative">
          {/* Error Message */}
          <AnimatePresence>
            {errorMessage && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="absolute bottom-full left-0 right-0 mb-4 p-4 rounded-[2rem] bg-red-500/10 border border-red-500/20 text-red-500 text-sm flex items-center justify-between glass-morphism"
              >
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5" />
                  <span className="font-medium">{errorMessage}</span>
                </div>
                <button onClick={() => setErrorMessage(null)} className="p-2 hover:bg-red-500/10 rounded-xl transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Live Captions Overlay */}
          <AnimatePresence>
            {isActive && liveCaption && (
              <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 20, scale: 0.95 }}
                className="absolute bottom-44 left-1/2 -translate-x-1/2 w-full max-w-lg px-6 z-40 pointer-events-none"
              >
                <div className="p-6 rounded-[2.5rem] glass-morphism-heavy shadow-2xl text-center shadow-black/80 ring-1 ring-white/10">
                  <div className="flex items-center justify-center gap-3 mb-3">
                    <div className={`w-1.5 h-1.5 rounded-full ${liveCaption.isUser ? 'bg-blue-400' : 'bg-[#10B981]'} animate-pulse`} />
                    <span className="text-[10px] font-bold uppercase tracking-[0.3em] opacity-40">Direct Transcription</span>
                  </div>
                  <p className="text-lg font-medium leading-relaxed italic text-white/90">
                    "{liveCaption.text}"
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Main Control Bar */}
          <div className="p-6 rounded-[2.8rem] glass-morphism-heavy flex flex-col gap-6 relative shadow-[0_32px_64px_-12px_rgba(0,0,0,0.5)] ring-1 ring-white/[0.05]">
            <form 
              onSubmit={handleSendText}
              className="flex items-center gap-4 p-4 rounded-3xl bg-white/[0.03] border border-white/[0.06] focus-within:border-[#10B981]/40 focus-within:bg-white/[0.05] transition-all group"
            >
              <div className="pl-2">
                <Activity className="w-5 h-5 text-white/10 group-focus-within:text-[#10B981]/60 transition-colors" />
              </div>
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Share your symptoms or ask a medical question..."
                className="flex-1 bg-transparent border-none outline-none text-base px-2 text-white placeholder:text-white/20 font-medium"
              />
              <button 
                type="submit"
                disabled={!textInput.trim() || status === 'connecting'}
                className="p-3.5 rounded-2xl bg-gradient-to-br from-[#10B981] to-[#059669] text-white disabled:opacity-20 disabled:grayscale transition-all hover:scale-105 active:scale-95 shadow-xl shadow-[#10B981]/20 group-active:translate-x-1"
              >
                <Send className="w-6 h-6" />
              </button>
            </form>

            <div className="flex items-center justify-between gap-8">
              <div className="flex items-center gap-4">
                <motion.button 
                  onClick={() => setIsMuted(!isMuted)}
                  disabled={!isActive}
                  animate={{ 
                    scale: userVolume > 0.01 ? [1, 1.05, 1] : 1,
                  }}
                  transition={{ duration: 0.2 }}
                  className={`p-4 rounded-2xl transition-all border ${
                    isMuted 
                      ? 'bg-red-500/10 border-red-500/20 text-red-500' 
                      : 'bg-white/[0.03] border-white/[0.08] text-white/60 hover:text-white hover:bg-white/10'
                  } disabled:opacity-20 disabled:cursor-not-allowed`}
                >
                  {isMuted ? <MicOff className="w-7 h-7" /> : <Mic className="w-7 h-7" />}
                </motion.button>
              </div>

              {/* Pulse Visualization */}
              <div className="flex-1 h-16 flex items-center justify-center gap-1.5 px-6 rounded-[2rem] bg-black/40 overflow-hidden relative border border-white/[0.03]">
                {isActive ? (
                  Array.from({ length: 48 }).map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{ 
                        height: status === 'active' 
                          ? (aiVolume > 0.01 ? [8, aiVolume * 80 + 8, 8] : [8, userVolume * 80 + 8, 8]) 
                          : 4
                      }}
                      style={{ 
                        opacity: 0.2 + (i / 48) * 0.8,
                        backgroundColor: aiVolume > 0.01 
                          ? "#10B981" 
                          : userVolume > 0.01 
                            ? "#3B82F6" 
                            : "rgba(255, 255, 255, 0.1)"
                      }}
                      transition={{ 
                        duration: 0.15,
                        delay: i * 0.005,
                        repeat: Infinity,
                        repeatType: "mirror"
                      }}
                      className="w-1 rounded-full"
                    />
                  ))
                ) : (
                  <div className="flex items-center gap-1.5 opacity-10">
                    {Array.from({ length: 48 }).map((_, i) => (
                      <div key={i} className="w-1 h-[2.5px] bg-white rounded-full shrink-0" />
                    ))}
                  </div>
                )}
                {isActive && (
                  <div className="absolute inset-x-0 bottom-0 h-[3px] bg-gradient-to-r from-transparent via-[#10B981]/40 to-transparent blur-md" />
                )}
              </div>

              <div className="flex items-center gap-5">
                <button
                  onClick={() => window.location.href = "tel:911"}
                  className="px-6 py-4 rounded-2xl bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-all border border-red-500/20 flex items-center gap-3 text-[11px] font-black uppercase tracking-[0.25em] shadow-lg shadow-red-500/[0.05] group shrink-0"
                >
                  <Phone className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                  911
                </button>

                <button
                  onClick={isActive ? endSession : () => startSession()}
                  disabled={status === 'connecting'}
                  className={`px-12 py-4 rounded-2xl font-bold transition-all flex items-center gap-4 text-sm tracking-tight shrink-0 ${
                    isActive 
                      ? 'bg-white/10 text-white hover:bg-white/20 border border-white/5' 
                      : 'bg-gradient-to-br from-[#10B981] to-[#059669] text-white hover:brightness-110 shadow-2xl shadow-[#10B981]/30 scale-[1.02] hover:scale-[1.05] active:scale-100'
                  } disabled:opacity-50`}
                >
                  {status === 'connecting' ? (
                    <div className="w-6 h-6 border-[3px] border-white/30 border-t-white rounded-full animate-spin" />
                  ) : isActive ? (
                    'End Consultation'
                  ) : (
                    <>
                      <Activity className="w-6 h-6" />
                      Speak to RapidAid
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Resources Modal */}
      <AnimatePresence>
        {showResources && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowResources(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-[#0F1115] border border-white/10 rounded-[2.5rem] overflow-hidden shadow-2xl"
            >
              <div className="p-8 border-b border-white/[0.05] flex items-center justify-between glass-morphism">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-2xl bg-[#10B981]/10 flex items-center justify-center border border-[#10B981]/20">
                    <BookOpen className="w-6 h-6 text-[#10B981]" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-white tracking-tight">Health Knowledge Base</h2>
                    <p className="text-xs text-white/40 font-medium uppercase tracking-widest mt-0.5">Verified Medical Resources</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowResources(false)}
                  className="p-2.5 hover:bg-white/5 rounded-2xl transition-all border border-transparent hover:border-white/10"
                >
                  <X className="w-6 h-6 text-white/40" />
                </button>
              </div>

              <div className="p-8 max-h-[60vh] overflow-y-auto custom-scrollbar bg-[#0F1115]">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  {MEDICAL_RESOURCES.map((resource, idx) => (
                    <a 
                      key={idx}
                      href={resource.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group p-5 rounded-[2rem] bg-white/[0.02] border border-white/[0.05] hover:bg-white/[0.05] hover:border-[#10B981]/30 transition-all flex flex-col justify-between shadow-sm relative overflow-hidden"
                    >
                      <div className="relative z-10">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-[#10B981]/80">
                            {resource.category}
                          </span>
                          <ExternalLink className="w-4 h-4 text-white/10 group-hover:text-[#10B981] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
                        </div>
                        <h3 className="text-base font-bold text-white mb-2 group-hover:text-[#10B981] transition-colors leading-tight">
                          {resource.name}
                        </h3>
                        <p className="text-sm text-white/40 leading-relaxed font-medium">
                          {resource.description}
                        </p>
                      </div>
                      <div className="absolute top-0 right-0 w-24 h-24 bg-[#10B981]/5 blur-3xl opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>
                  ))}
                </div>

                <div className="mt-10 p-6 rounded-3xl bg-red-500/[0.03] border border-red-500/10">
                  <div className="flex items-start gap-4">
                    <AlertCircle className="w-6 h-6 text-red-500/70 shrink-0 mt-0.5" />
                    <p className="text-sm text-red-500/80 leading-relaxed font-medium">
                      <strong>Life-Saving Notice:</strong> These resources are for informational and preventative care only. If you are experiencing a life-threatening emergency, call your local emergency number (911, 999, etc.) immediately.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-8 bg-white/[0.02] border-t border-white/[0.05] flex justify-end">
                <button 
                  onClick={() => setShowResources(false)}
                  className="px-8 py-3 rounded-2xl bg-white/5 text-white/70 text-sm font-bold uppercase tracking-widest hover:bg-white/10 hover:text-white transition-all border border-white/5"
                >
                  Dismiss
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}} />
    </div>
  );
}
