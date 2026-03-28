/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { Mic, MicOff, Activity, Stethoscope, AlertCircle, Info, X, Volume2, VolumeX, Globe, ExternalLink, BookOpen, Phone, Trash2 } from 'lucide-react';
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
CRITICAL SAFETY RULES:
1. Always start or end with a disclaimer: "I am an AI assistant, not a doctor. This is for informational purposes only." (Translate this disclaimer to the user's language).
2. If the user mentions symptoms of a life-threatening emergency (chest pain, severe bleeding, difficulty breathing, stroke symptoms), immediately tell them to call emergency services (e.g., 911). You can trigger this automatically by calling the 'callEmergencyServices' tool.
3. Be concise, conversational, and rapid in your delivery.
4. If you are unsure, advise the user to consult a licensed medical professional.
5. Do not prescribe medication or give definitive diagnoses.`;

// --- Types ---
interface Transcription {
  text: string;
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

  const startSession = async () => {
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

  const speakText = (text: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.5;
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-[#E4E4E7] font-sans selection:bg-[#10B981]/30">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#10B981]/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-[#3B82F6]/10 blur-[120px] rounded-full" />
      </div>

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-white/5 bg-black/20 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#10B981] flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)]">
            <Stethoscope className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">RapidAid</h1>
            <p className="text-xs text-white/40 font-medium uppercase tracking-widest">Voice Health Assistant</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {transcriptions.length > 0 && (
            <button 
              onClick={clearHistory}
              className="p-2 rounded-xl bg-white/5 border border-white/10 text-white/40 hover:text-red-400 hover:bg-red-400/10 transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
              title="Clear History"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </button>
          )}

          <button 
            onClick={() => setShowResources(true)}
            className="p-2 rounded-xl bg-white/5 border border-white/10 text-white/60 hover:text-white hover:bg-white/10 transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Resources</span>
          </button>

          {isActive && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#10B981]/10 border border-[#10B981]/20 text-[#10B981] text-xs font-medium">
              <Globe className="w-3.5 h-3.5" />
              {detectedLanguage}
            </div>
          )}

          <div className="relative">
            <button 
              onClick={() => setShowVoiceMenu(!showVoiceMenu)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 text-xs font-medium hover:bg-white/10 transition-all"
            >
              <Volume2 className="w-3.5 h-3.5 text-[#10B981]" />
              {selectedVoice}
            </button>
            <AnimatePresence>
              {showVoiceMenu && (
                <motion.div 
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  className="absolute top-full right-0 mt-2 w-32 bg-[#18181B] border border-white/10 rounded-xl overflow-hidden shadow-2xl z-50"
                >
                  {voices.map(voice => (
                    <button
                      key={voice}
                      onClick={() => {
                        setSelectedVoice(voice);
                        setShowVoiceMenu(false);
                      }}
                      className={`w-full text-left px-4 py-2 text-xs hover:bg-white/5 transition-colors ${selectedVoice === voice ? 'text-[#10B981] bg-[#10B981]/5' : 'text-white/60'}`}
                    >
                      {voice}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border ${
            status === 'active' ? 'bg-[#10B981]/10 border-[#10B981]/20 text-[#10B981]' : 
            status === 'connecting' ? 'bg-[#F59E0B]/10 border-[#F59E0B]/20 text-[#F59E0B]' :
            'bg-white/5 border-white/10 text-white/40'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${
              status === 'active' ? 'bg-[#10B981] animate-pulse' : 
              status === 'connecting' ? 'bg-[#F59E0B] animate-pulse' : 
              'bg-white/20'
            }`} />
            {status === 'active' ? 'Live' : status === 'connecting' ? 'Connecting' : 'Offline'}
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-4xl mx-auto px-6 py-8 h-[calc(100vh-80px)] flex flex-col">
        {/* Medical Disclaimer Banner */}
        <div className="mb-6 p-4 rounded-2xl bg-[#F59E0B]/5 border border-[#F59E0B]/10 flex gap-3 items-start">
          <AlertCircle className="w-5 h-5 text-[#F59E0B] shrink-0 mt-0.5" />
          <p className="text-sm text-[#F59E0B]/80 leading-relaxed">
            <span className="font-bold">Medical Disclaimer:</span> RapidAid is an AI assistant for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. <span className="font-bold underline">In case of emergency, call 911 immediately.</span>
          </p>
        </div>

        {/* Transcription Area */}
        <div className="flex-1 overflow-y-auto mb-8 space-y-6 pr-4 custom-scrollbar">
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
            transcriptions.map((t, i) => (
              <motion.div
                key={t.timestamp + i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${t.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`group relative max-w-[80%] p-4 rounded-2xl ${
                  t.isUser 
                    ? 'bg-[#10B981]/10 border border-[#10B981]/20 text-white' 
                    : 'bg-white/5 border border-white/10 text-white/90'
                }`}>
                  <p className="text-sm leading-relaxed">{t.text}</p>
                  {!t.isUser && (
                    <button 
                      onClick={() => speakText(t.text)}
                      className="absolute -right-10 top-1/2 -translate-y-1/2 p-2 rounded-lg bg-white/5 text-white/40 hover:text-white hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-all"
                      title="Read aloud"
                    >
                      <Volume2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </motion.div>
            ))
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
                className="absolute bottom-full left-0 right-0 mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm flex items-center justify-between"
              >
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  {errorMessage}
                </div>
                <button onClick={() => setErrorMessage(null)} className="p-1 hover:bg-red-500/10 rounded-lg">
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Live Captions Overlay */}
        <AnimatePresence>
          {isActive && liveCaption && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="absolute bottom-32 left-1/2 -translate-x-1/2 w-full max-w-xl px-6 z-20 pointer-events-none"
            >
              <div className={`p-4 rounded-2xl backdrop-blur-xl border ${
                liveCaption.isUser 
                  ? 'bg-blue-500/10 border-blue-500/20 text-blue-200' 
                  : 'bg-[#10B981]/10 border-[#10B981]/20 text-[#10B981]'
              } shadow-2xl text-center`}>
                <p className="text-sm font-medium leading-relaxed">
                  {liveCaption.text}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Control Bar */}
          <div className="p-4 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl flex flex-col gap-4">
            <div className="flex items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <motion.button 
                  onClick={() => setIsMuted(!isMuted)}
                  disabled={!isActive}
                  animate={{ 
                    scale: userVolume > 0.01 ? [1, 1.1, 1] : 1,
                    boxShadow: userVolume > 0.01 ? `0 0 ${userVolume * 100}px rgba(16, 185, 129, 0.4)` : 'none'
                  }}
                  transition={{ duration: 0.2 }}
                  className={`p-3 rounded-2xl transition-all ${
                    isMuted ? 'bg-red-500/20 text-red-500' : 'bg-white/5 text-white/60 hover:text-white hover:bg-white/10'
                  } disabled:opacity-20 disabled:cursor-not-allowed`}
                >
                  {isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
                </motion.button>
              </div>

              {/* Pulse Visualization */}
              <div className="flex-1 h-12 flex items-center justify-center gap-1">
                {isActive ? (
                  Array.from({ length: 24 }).map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{ 
                        height: aiVolume > 0.01 
                          ? [8, Math.random() * (aiVolume * 100) + 8, 8]
                          : userVolume > 0.01 
                            ? [8, Math.random() * (userVolume * 100) + 8, 8]
                            : 8,
                        backgroundColor: aiVolume > 0.01 
                          ? "#10B981" // Green for AI
                          : userVolume > 0.01 
                            ? "#3B82F6" // Blue for User
                            : "rgba(255, 255, 255, 0.1)"
                      }}
                      transition={{ 
                        duration: 0.2,
                        ease: "easeInOut"
                      }}
                      className="w-1 rounded-full opacity-60"
                    />
                  ))
                ) : (
                  <div className="w-full h-[1px] bg-white/10" />
                )}
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => window.location.href = "tel:911"}
                  className="p-3 rounded-2xl bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-all border border-red-500/20 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
                >
                  <Phone className="w-5 h-5" />
                  911
                </button>

                <button
                  onClick={isActive ? endSession : () => startSession()}
                  disabled={status === 'connecting'}
                  className={`px-8 py-3 rounded-2xl font-semibold transition-all flex items-center gap-2 ${
                    isActive 
                      ? 'bg-white/10 text-white hover:bg-white/20' 
                      : 'bg-[#10B981] text-white hover:bg-[#059669] shadow-[0_0_20px_rgba(16,185,129,0.3)]'
                  } disabled:opacity-50`}
                >
                  {status === 'connecting' ? (
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : isActive ? (
                    'End Session'
                  ) : (
                    <>
                      <Activity className="w-5 h-5" />
                      Start Consultation
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
              className="relative w-full max-w-2xl bg-[#0A0A0A] border border-white/10 rounded-[2rem] overflow-hidden shadow-2xl"
            >
              <div className="p-6 border-b border-white/10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-[#10B981]/20 flex items-center justify-center">
                    <BookOpen className="w-5 h-5 text-[#10B981]" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">Medical Resources</h2>
                    <p className="text-xs text-white/40">Curated links to reputable health organizations</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowResources(false)}
                  className="p-2 hover:bg-white/5 rounded-xl transition-all"
                >
                  <X className="w-6 h-6 text-white/40" />
                </button>
              </div>

              <div className="p-6 max-h-[60vh] overflow-y-auto custom-scrollbar">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {MEDICAL_RESOURCES.map((resource, idx) => (
                    <a 
                      key={idx}
                      href={resource.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group p-4 rounded-2xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-[#10B981]/30 transition-all flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[10px] uppercase tracking-widest font-bold text-[#10B981] opacity-60">
                            {resource.category}
                          </span>
                          <ExternalLink className="w-3 h-3 text-white/20 group-hover:text-[#10B981] transition-colors" />
                        </div>
                        <h3 className="text-sm font-bold text-white mb-1 group-hover:text-[#10B981] transition-colors">
                          {resource.name}
                        </h3>
                        <p className="text-xs text-white/40 leading-relaxed">
                          {resource.description}
                        </p>
                      </div>
                    </a>
                  ))}
                </div>

                <div className="mt-8 p-4 rounded-2xl bg-red-500/5 border border-red-500/10">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                    <p className="text-xs text-red-500/80 leading-relaxed">
                      <strong>Important:</strong> These resources are for informational purposes only. In case of a medical emergency, please contact your local emergency services immediately.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-white/5 border-t border-white/10 flex justify-end">
                <button 
                  onClick={() => setShowResources(false)}
                  className="px-6 py-2 rounded-xl bg-white/10 text-white text-sm font-medium hover:bg-white/20 transition-all"
                >
                  Close
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
