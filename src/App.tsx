/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { Mic, MicOff, Activity, Stethoscope, AlertCircle, Info, X, Volume2, VolumeX, Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Constants ---
const MODEL = "gemini-3.1-flash-live-preview";
const SAMPLE_RATE = 16000;
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
2. If the user mentions symptoms of a life-threatening emergency (chest pain, severe bleeding, difficulty breathing, stroke symptoms), immediately tell them to call emergency services (e.g., 911).
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
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<string>('Puck');
  const [showVoiceMenu, setShowVoiceMenu] = useState(false);
  const [detectedLanguage, setDetectedLanguage] = useState<string>('Detecting...');
  const [speechRate, setSpeechRate] = useState<number>(1.0);

  const voices = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'];
  const speechRates = [0.75, 1.0, 1.25, 1.5];

  // Refs for audio and session
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const nextStartTimeRef = useRef<number>(0);
  const speechRateRef = useRef<number>(1.0);
  const transcriptionEndRef = useRef<HTMLDivElement>(null);

  // Sync speechRate state to ref for audio scheduling
  useEffect(() => {
    speechRateRef.current = speechRate;
  }, [speechRate]);

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

  const startSession = async () => {
    try {
      setStatus('connecting');
      setErrorMessage(null);

      // 1. Initialize Audio
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 2. Initialize Gemini Live
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
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
            
            // Start sending audio
            const source = audioContextRef.current!.createMediaStreamSource(streamRef.current!);
            const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor as any;

            processor.onaudioprocess = (e) => {
              if (isMuted) return;
              const inputData = e.inputBuffer.getChannelData(0);
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
              audioQueueRef.current.push(pcmData);
              playNextChunk();
            }

            // Handle Interruption
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              nextStartTimeRef.current = 0;
            }

            // Handle Transcriptions
            const serverContent = message.serverContent as any;
            const userText = serverContent?.userContent?.parts?.[0]?.text;
            if (userText) {
              setTranscriptions(prev => [...prev, { text: userText, isUser: true, timestamp: Date.now() }]);
            }

            const modelText = message.serverContent?.modelTurn?.parts?.[0]?.text;
            if (modelText) {
              setTranscriptions(prev => [...prev, { text: modelText, isUser: false, timestamp: Date.now() }]);
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
                }
              }
            }
          },
          onclose: () => {
            setIsActive(false);
            setStatus('idle');
            stopAudio();
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
  };

  // Auto-scroll transcriptions
  useEffect(() => {
    transcriptionEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcriptions]);

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
          <div className="flex items-center gap-1 p-1 rounded-xl bg-white/5 border border-white/10">
            {speechRates.map(rate => (
              <button
                key={rate}
                onClick={() => setSpeechRate(rate)}
                className={`px-2 py-1 rounded-lg text-[10px] font-bold transition-all ${
                  speechRate === rate 
                    ? 'bg-[#10B981] text-white shadow-[0_0_10px_rgba(16,185,129,0.3)]' 
                    : 'text-white/40 hover:text-white/60'
                }`}
              >
                {rate}x
              </button>
            ))}
          </div>

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
          {transcriptions.length === 0 ? (
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
                <div className={`max-w-[80%] p-4 rounded-2xl ${
                  t.isUser 
                    ? 'bg-[#10B981]/10 border border-[#10B981]/20 text-white' 
                    : 'bg-white/5 border border-white/10 text-white/90'
                }`}>
                  <p className="text-sm leading-relaxed">{t.text}</p>
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

          {/* Main Control Bar */}
          <div className="p-6 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl flex items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <button 
                onClick={() => setIsMuted(!isMuted)}
                disabled={!isActive}
                className={`p-3 rounded-2xl transition-all ${
                  isMuted ? 'bg-red-500/20 text-red-500' : 'bg-white/5 text-white/60 hover:text-white hover:bg-white/10'
                } disabled:opacity-20 disabled:cursor-not-allowed`}
              >
                {isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
              </button>
            </div>

            {/* Pulse Visualization */}
            <div className="flex-1 h-12 flex items-center justify-center gap-1">
              {isActive ? (
                Array.from({ length: 24 }).map((_, i) => (
                  <motion.div
                    key={i}
                    animate={{ 
                      height: [8, Math.random() * 32 + 8, 8],
                    }}
                    transition={{ 
                      repeat: Infinity, 
                      duration: 0.5 + Math.random() * 0.5,
                      ease: "easeInOut"
                    }}
                    className="w-1 rounded-full bg-[#10B981]/60"
                  />
                ))
              ) : (
                <div className="w-full h-[1px] bg-white/10" />
              )}
            </div>

            <button
              onClick={isActive ? endSession : startSession}
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
      </main>

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
