/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { Mic, MicOff, Activity, Stethoscope, AlertCircle, Info, X, Volume2, VolumeX, Globe, ExternalLink, BookOpen, Phone, Trash2, Download, Send, CheckCircle2, Clock, ShieldAlert, History, Plus, ChevronLeft, MessageSquare, LogOut, User as UserIcon, Menu } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAuth } from './context/AuthContext';
import { AuthModal } from './components/AuthModal';
import { ProfileModal } from './components/ProfileModal';
import { db, auth as firebaseAuth, handleFirestoreError, OperationType } from './lib/firebase';
import { collection, query, where, orderBy, onSnapshot, addDoc, deleteDoc, doc, updateDoc, getDocs, limit, serverTimestamp } from 'firebase/firestore';
import { signOut } from 'firebase/auth';

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
- Minimize fillers and provide direct, actionable advice where appropriate.
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

interface Session {
  id: string;
  title: string;
  timestamp: number;
  transcriptions: Transcription[];
}

export default function App() {
  const { user, userData, loading: authLoading } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showProfileModal, setShowProfileModal] = useState(false);

  const [isActive, setIsActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);

  const activeSession = sessions.find(s => s.id === currentSessionId);
  const transcriptions = activeSession ? activeSession.transcriptions : [];

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

  // Firestore Sync
  useEffect(() => {
    if (!user) {
      setSessions([]);
      setCurrentSessionId(null);
      return;
    }

    const q = query(
      collection(db, 'sessions'),
      where('userId', '==', user.uid),
      orderBy('timestamp', 'desc'),
      limit(50)
    );

    const unsubscribe = onSnapshot(q, async (snapshot) => {
      const sessionData: Session[] = [];
      
      for (const sessionDoc of snapshot.docs) {
        const data = sessionDoc.data();
        sessionData.push({
          id: sessionDoc.id,
          title: data.title,
          timestamp: data.timestamp,
          transcriptions: []
        });
      }
      setSessions(sessionData);
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'sessions');
    });

    return () => unsubscribe();
  }, [user]);

  // Sync Messages for ACTIVE Session
  useEffect(() => {
    if (!user || !currentSessionId) return;

    const q = query(
      collection(db, `sessions/${currentSessionId}/messages`),
      where('userId', '==', user.uid),
      orderBy('timestamp', 'asc')
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const msgs = snapshot.docs.map(d => d.data() as Transcription);
      setSessions(prev => prev.map(s => {
        if (s.id === currentSessionId) {
          return { ...s, transcriptions: msgs };
        }
        return s;
      }));
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, `sessions/${currentSessionId}/messages`);
    });

    return () => unsubscribe();
  }, [user, currentSessionId]);

  const updateTranscriptions = useCallback(async (updater: (prev: Transcription[]) => Transcription[]) => {
    if (!user) {
      // Fallback to local state for anonymous if we want, but user requested security/personalization
      // So let's REQUIRE login or just show a warning.
      setErrorMessage("Please log in to save your consultation.");
      setShowAuthModal(true);
      return;
    }

    let activeId = currentSessionId;

    if (!activeId) {
      try {
        const sessionRef = await addDoc(collection(db, 'sessions'), {
          userId: user.uid,
          title: `Consultation ${new Date().toLocaleDateString()}`,
          timestamp: Date.now(),
          updatedAt: serverTimestamp()
        });
        activeId = sessionRef.id;
        setCurrentSessionId(activeId);
      } catch (e) {
        handleFirestoreError(e, OperationType.CREATE, 'sessions');
        return;
      }
    }

    // Get current messages to run the updater
    const session = sessions.find(s => s.id === activeId);
    const currentMsgs = session ? session.transcriptions : [];
    const newMsgs = updater(currentMsgs);
    
    if (newMsgs.length > currentMsgs.length) {
      const latest = newMsgs[newMsgs.length - 1];
      try {
        await addDoc(collection(db, `sessions/${activeId}/messages`), {
          ...latest,
          sessionId: activeId,
          userId: user.uid
        });
      } catch (e) {
        handleFirestoreError(e, OperationType.CREATE, `sessions/${activeId}/messages`);
      }

      // Update session title if needed
      if (newMsgs.length === 1 && latest.text) {
        const title = latest.text.slice(0, 30) + (latest.text.length > 30 ? '...' : '');
        try {
          await updateDoc(doc(db, 'sessions', activeId), { title });
        } catch (e) {
          handleFirestoreError(e, OperationType.UPDATE, `sessions/${activeId}`);
        }
      }
    } else if (newMsgs.length > 0 && currentMsgs.length > 0) {
      const latestNew = newMsgs[newMsgs.length - 1];
      const latestOld = currentMsgs[currentMsgs.length - 1];
      if (latestNew.text !== latestOld.text) {
        const q = query(
          collection(db, `sessions/${activeId}/messages`),
          where('userId', '==', user.uid),
          orderBy('timestamp', 'desc'),
          limit(1)
        );
        try {
          const snap = await getDocs(q);
          if (!snap.empty) {
            await updateDoc(doc(db, `sessions/${activeId}/messages`, snap.docs[0].id), {
              text: latestNew.text,
              timestamp: latestNew.timestamp
            });
          }
        } catch (e) {
          handleFirestoreError(e, OperationType.GET, `sessions/${activeId}/messages`);
        }
      }
    }
  }, [user, currentSessionId, sessions]);

  const startNewSession = async () => {
    if (!user) {
      setShowAuthModal(true);
      return;
    }
    
    try {
      const sessionRef = await addDoc(collection(db, 'sessions'), {
        userId: user.uid,
        title: `Consultation ${new Date().toLocaleDateString()}`,
        timestamp: Date.now(),
        updatedAt: serverTimestamp()
      });
      setCurrentSessionId(sessionRef.id);
      setShowHistory(false);
      if (isActive) endSession();
    } catch (e) {
      handleFirestoreError(e, OperationType.CREATE, 'sessions');
    }
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm("Delete this consultation?")) {
      try {
        await deleteDoc(doc(db, 'sessions', id));
        if (currentSessionId === id) {
          setCurrentSessionId(null);
        }
      } catch (e) {
        handleFirestoreError(e, OperationType.DELETE, `sessions/${id}`);
      }
    }
  };

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

  const getSystemInstruction = useCallback(() => {
    let instruction = SYSTEM_INSTRUCTION;
    if (userData?.healthProfile) {
      const p = userData.healthProfile;
      const context = `
USER HEALTH CONTEXT:
- Age: ${p.age || 'Not shared'}
- Gender: ${p.gender || 'Not shared'}
- Pre-existing conditions: ${p.conditions || 'None shared'}
- Allergies: ${p.allergies || 'None shared'}
- Medications: ${p.medications || 'None shared'}
- Blood Type: ${p.bloodType || 'Not shared'}
Use this information to provide more personalized and relevant health guidance. Avoid repeating this info back to the user unless necessary for clarification.`;
      instruction += context;
    }
    return instruction;
  }, [userData]);

  const startSession = async (initialText?: string) => {
    try {
      if (!user && !initialText) {
        setShowAuthModal(true);
        return;
      }
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
          systemInstruction: getSystemInstruction(),
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
                    updateTranscriptions(prev => {
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
                    updateTranscriptions(prev => {
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
                  updateTranscriptions(prev => [...prev, { analysis, isUser: false, timestamp: Date.now() }]);
                  
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
      updateTranscriptions(prev => [...prev, { 
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
    updateTranscriptions(prev => [...prev, { 
      text, 
      isUser: true, 
      timestamp: Date.now() 
    }]);
    
    setTextInput('');
  };

    // Auto-scroll transcriptions
    useEffect(() => {
      transcriptionEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcriptions]);

  useEffect(() => {
    if (liveCaption) {
      const timer = setTimeout(() => setLiveCaption(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [liveCaption]);

  const clearHistory = () => {
    if (window.confirm("Are you sure you want to clear your conversation history?")) {
      updateTranscriptions(() => []);
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
    <div className="flex h-screen bg-white text-slate-900 font-sans selection:bg-blue-500/30 overflow-hidden">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] bg-blue-500/3 blur-[100px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-5%] w-[40%] h-[40%] bg-blue-400/3 blur-[100px] rounded-full" />
      </div>

      {/* History Sidebar - Real AI Layout */}
      <motion.aside
        initial={false}
        animate={{ width: showHistory ? 280 : 0, opacity: showHistory ? 1 : 0 }}
        className="relative flex-shrink-0 bg-slate-50 border-r border-slate-200 z-40 flex flex-col h-full overflow-hidden"
      >
        <div className="w-[280px] flex flex-col h-full">
          <div className="p-5 border-b border-slate-200/60 flex items-center justify-between bg-slate-50/80 backdrop-blur-md">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm">
                <Stethoscope className="w-5 h-5" />
              </div>
              <span className="font-bold text-slate-900 tracking-tight">RapidAid</span>
            </div>
            <button 
              onClick={() => setShowHistory(false)}
              className="p-1.5 hover:bg-slate-200 rounded-md transition-colors text-slate-400"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          <div className="p-4">
            <button
              onClick={startNewSession}
              className="w-full p-2.5 rounded-xl bg-white border border-slate-200 hover:border-blue-300 hover:shadow-sm transition-all flex items-center gap-3 text-sm font-semibold text-slate-700 group"
            >
              <Plus className="w-4 h-4 text-blue-600" />
              New Consultation
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-1 custom-scrollbar">
            {sessions.length === 0 ? (
              <div className="px-4 py-8 text-center text-slate-400">
                <p className="text-xs font-medium">No previous logs</p>
              </div>
            ) : (
              sessions.map(session => (
                <button
                  key={session.id}
                  onClick={() => setCurrentSessionId(session.id)}
                  className={`w-full p-3 rounded-xl text-left transition-all group flex flex-col gap-0.5 border ${
                    currentSessionId === session.id 
                      ? 'bg-blue-600 text-white border-blue-500 shadow-md shadow-blue-500/10' 
                      : 'bg-transparent border-transparent hover:bg-slate-200/50 text-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className={`text-[13px] font-semibold truncate flex-1 ${currentSessionId === session.id ? 'text-white' : 'text-slate-800'}`}>
                      {session.title}
                    </span>
                    <Trash2 
                      onClick={(e) => deleteSession(session.id, e)}
                      className={`w-3.5 h-3.5 opacity-0 group-hover:opacity-100 hover:text-red-400 transition-all ml-2 ${currentSessionId === session.id ? 'text-blue-100' : 'text-slate-400'}`} 
                    />
                  </div>
                  <span className={`text-[10px] ${currentSessionId === session.id ? 'text-blue-100' : 'text-slate-400'}`}>
                    {new Date(session.timestamp).toLocaleDateString()}
                  </span>
                </button>
              ))
            )}
          </div>

          <div className="p-4 border-t border-slate-200/60 bg-slate-50/50">
            {user ? (
              <div className="flex items-center gap-3 p-2 rounded-xl bg-white border border-slate-200 shadow-sm">
                <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600">
                  <UserIcon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[11px] font-bold text-slate-900 truncate">{user.email?.split('@')[0]}</p>
                  <button 
                    onClick={() => signOut(firebaseAuth)}
                    className="text-[9px] font-bold text-slate-400 hover:text-red-500 uppercase tracking-widest transition-colors"
                  >
                    Logout
                  </button>
                </div>
              </div>
            ) : (
              <button 
                onClick={() => setShowAuthModal(true)}
                className="w-full py-2 rounded-lg bg-slate-900 text-white text-[10px] font-bold uppercase tracking-widest hover:bg-slate-800 transition-colors"
              >
                Sign In
              </button>
            )}
          </div>
        </div>
      </motion.aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 relative h-full bg-white">
        {/* Simplified Sticky Header */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-slate-100 bg-white/80 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-3">
            {!showHistory && (
              <button 
                onClick={() => setShowHistory(true)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors text-slate-500"
              >
                <History className="w-5 h-5" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 'bg-slate-300'}`} />
              <span className="text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                {status === 'active' ? 'System Ready' : 'Standby'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={() => setShowResources(true)}
              className="p-2 text-slate-500 hover:text-blue-600 transition-colors"
              title="Health Library"
            >
              <BookOpen className="w-5 h-5" />
            </button>
            {user && (
              <button 
                onClick={() => setShowProfileModal(true)}
                className="p-2 text-slate-500 hover:text-blue-600 transition-colors"
                title="Profile"
              >
                <UserIcon className="w-5 h-5" />
              </button>
            )}
            <div className="w-[1px] h-4 bg-slate-200 mx-1" />
            <div className="relative">
              <button 
                onClick={() => setShowVoiceMenu(!showVoiceMenu)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-[10px] font-bold text-slate-600 hover:bg-slate-100 transition-all"
              >
                <Volume2 className="w-3.5 h-3.5 text-blue-500" />
                {selectedVoice}
              </button>
              <AnimatePresence>
                {showVoiceMenu && (
                  <motion.div 
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 5 }}
                    className="absolute top-full right-0 mt-2 w-32 bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xl z-50 p-1"
                  >
                    {voices.map(voice => (
                      <button
                        key={voice}
                        onClick={() => {
                          setSelectedVoice(voice);
                          setShowVoiceMenu(false);
                        }}
                        className={`w-full text-left px-3 py-2 text-[10px] font-semibold rounded-lg transition-colors ${selectedVoice === voice ? 'text-blue-600 bg-blue-50' : 'text-slate-500 hover:bg-slate-50'}`}
                      >
                        {voice}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-hidden flex flex-col relative w-full">
          {/* Main Scroll Container */}
          <div className="flex-1 overflow-y-auto custom-scrollbar px-6 scroll-smooth">
            <div className="max-w-3xl mx-auto py-10 space-y-12">
              {/* Medical Disclaimer Banner */}
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-2xl bg-amber-50 border border-amber-100 flex items-start gap-3 shadow-sm"
              >
                <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                <p className="text-[12px] text-amber-800 leading-relaxed font-medium">
                  <strong>Notice:</strong> RapidAid is an AI assistant, not a doctor. In critical situations, contact medical professionals immediately.
                </p>
              </motion.div>

              {/* Transcription Area */}
              <div className="space-y-8">
                {isConfigMissing ? (
                  <div className="p-10 rounded-3xl bg-slate-50 border border-slate-200 text-center">
                    <div className="w-16 h-16 rounded-2xl bg-blue-100 text-blue-600 flex items-center justify-center mx-auto mb-6">
                      <AlertCircle className="w-10 h-10" />
                    </div>
                    <h2 className="text-xl font-bold text-slate-900 mb-2">Configuration Required</h2>
                    <p className="text-sm text-slate-500 max-w-sm mx-auto leading-relaxed">
                      Please add your GEMINI_API_KEY to the environment variables to start using RapidAid.
                    </p>
                  </div>
                ) : transcriptions.length === 0 ? (
                  <div className="h-[50vh] flex flex-col items-center justify-center text-center space-y-6">
                    <div className="relative">
                      <div className="absolute inset-0 bg-blue-500/20 blur-3xl rounded-full" />
                      <Stethoscope className="w-16 h-16 text-blue-600 relative z-10" />
                    </div>
                    <div className="space-y-2">
                      <h2 className="text-3xl font-bold text-slate-900 tracking-tight">How can I help you?</h2>
                      <p className="text-slate-500 max-w-md mx-auto">
                        Start a voice consultation or type your symptoms to receive an AI-powered health analysis.
                      </p>
                    </div>
                  </div>
                ) : (
                  transcriptions.map((t, i) => (
                    <motion.div
                      key={t.timestamp + i}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${t.isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`relative ${t.analysis ? 'w-full' : 'max-w-[85%]'}`}>
                        {t.analysis ? (
                          <div className="bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-xl shadow-slate-200/50">
                            <div className={`px-6 py-4 flex items-center justify-between border-b border-slate-100 ${
                              t.analysis.urgency === 'Emergency' ? 'bg-red-50' :
                              t.analysis.urgency === 'High' ? 'bg-amber-50' :
                              'bg-blue-50/50'
                            }`}>
                              <div className="flex items-center gap-3">
                                <ShieldAlert className={`w-5 h-5 ${
                                  t.analysis.urgency === 'Emergency' ? 'text-red-600' :
                                  t.analysis.urgency === 'High' ? 'text-amber-600' :
                                  'text-blue-600'
                                }`} />
                                <span className="font-bold text-slate-900 text-sm">Health Analysis</span>
                              </div>
                              <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${
                                t.analysis.urgency === 'Emergency' ? 'bg-red-600 text-white' :
                                t.analysis.urgency === 'High' ? 'bg-amber-500 text-white' :
                                'bg-blue-600 text-white'
                              }`}>
                                {t.analysis.urgency}
                              </span>
                            </div>
                            <div className="p-6 space-y-6">
                              <div className="space-y-3">
                                <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Potential Considerations</p>
                                <div className="grid grid-cols-1 gap-3">
                                  {t.analysis.potentialConditions.map((c, idx) => (
                                    <div key={idx} className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
                                      <div className="flex items-center justify-between mb-1.5">
                                        <h4 className="text-sm font-bold text-slate-900">{c.name}</h4>
                                        <span className="text-[10px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{c.likelihood}</span>
                                      </div>
                                      <p className="text-[12px] text-slate-500 leading-relaxed font-medium">{c.description}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
                                <div className="space-y-3">
                                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Next Steps</p>
                                  <ul className="space-y-2">
                                    {t.analysis.recommendations.map((r, idx) => (
                                      <li key={idx} className="flex gap-2 text-[12px] text-slate-600 font-medium">
                                        <CheckCircle2 className="w-4 h-4 text-blue-500 shrink-0" />
                                        {r}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                                {t.analysis.disclaimer && (
                                  <div className="p-4 rounded-2xl bg-amber-50/50 border border-amber-100/50">
                                    <p className="text-[11px] text-amber-700/80 italic font-medium leading-relaxed">
                                      {t.analysis.disclaimer}
                                    </p>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className={`p-4 rounded-2xl ${
                            t.isUser 
                              ? 'bg-blue-600 text-white shadow-md shadow-blue-500/20' 
                              : 'bg-white border border-slate-200 text-slate-800'
                          }`}>
                            <p className="text-sm leading-relaxed font-medium">{t.text}</p>
                            <div className={`mt-2 flex items-center gap-2 ${t.isUser ? 'text-blue-200' : 'text-slate-400'}`}>
                              <span className="text-[9px] font-bold uppercase tracking-widest">
                                {new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              {!t.isUser && (
                                <button onClick={() => speakText(t.text!)} className="hover:text-blue-500 transition-colors">
                                  <Volume2 className="w-3 h-3" />
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ))
                )}
                <div ref={transcriptionEndRef} className="h-4" />
              </div>
            </div>
          </div>

          {/* Floating Live Captions and Error */}
          <div className="absolute bottom-24 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6 z-40 pointer-events-none space-y-4">
            <AnimatePresence>
              {errorMessage && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className="p-4 rounded-2xl bg-red-600 text-white shadow-xl shadow-red-500/20 flex items-center justify-between pointer-events-auto"
                >
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5" />
                    <span className="text-sm font-bold">{errorMessage}</span>
                  </div>
                  <button onClick={() => setErrorMessage(null)} className="p-1 hover:bg-white/20 rounded-lg">
                    <X className="w-4 h-4" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {isActive && liveCaption && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="p-5 rounded-2xl bg-white/90 border border-slate-200 shadow-2xl backdrop-blur-xl text-center"
                >
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <div className="w-1 h-1 rounded-full bg-blue-500 animate-pulse" />
                    <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">Live Captions</span>
                  </div>
                  <p className="text-slate-900 font-medium italic">"{liveCaption.text}"</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Docked Control Bar */}
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-white via-white/80 to-transparent pt-10 pb-8 px-6 z-30 pointer-events-none">
            <div className="max-w-3xl mx-auto w-full pointer-events-auto">
              <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-br from-blue-500 to-blue-600 rounded-[2.5rem] blur opacity-10 group-focus-within:opacity-20 transition-opacity" />
                <div className="relative bg-white border border-slate-200 rounded-[2rem] shadow-xl shadow-slate-200/50 p-3 pl-6 flex items-center gap-4">
                  <div className="flex-1 flex items-center gap-4">
                    <Activity className={`w-5 h-5 ${isActive ? 'text-blue-600' : 'text-slate-300'} shrink-0`} />
                    <form 
                      onSubmit={handleSendText}
                      className="flex-1"
                    >
                      <input
                        type="text"
                        value={textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        placeholder="Type a symptom or question..."
                        className="w-full bg-transparent border-none outline-none text-base font-semibold text-slate-800 placeholder:text-slate-400"
                      />
                    </form>
                  </div>

                  <div className="flex items-center gap-2">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setIsMuted(!isMuted)}
                      disabled={!isActive}
                      className={`p-3 rounded-xl transition-all ${
                        isMuted 
                          ? 'bg-red-50 text-red-500' 
                          : 'bg-slate-50 text-slate-500 hover:text-blue-600'
                      } disabled:opacity-20`}
                    >
                      {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                    </motion.button>

                    <button
                      onClick={isActive ? endSession : () => startSession()}
                      disabled={status === 'connecting'}
                      className={`h-12 px-6 rounded-2xl font-black text-xs uppercase tracking-widest transition-all shadow-lg flex items-center gap-3 ${
                        isActive 
                          ? 'bg-slate-900 text-white shadow-slate-400/20' 
                          : 'bg-blue-600 text-white shadow-blue-500/20'
                      } disabled:opacity-50`}
                    >
                      {status === 'connecting' ? (
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      ) : isActive ? (
                        'End'
                      ) : (
                        <>
                          <Volume2 className="w-4 h-4" />
                          Consult
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Pulse visualization below the bar */}
                {isActive && (
                  <div className="absolute -bottom-1 inset-x-12 h-1 overflow-hidden pointer-events-none">
                    <div className="flex items-center justify-center gap-1">
                      {Array.from({ length: 40 }).map((_, i) => (
                        <motion.div
                          key={i}
                          animate={{ 
                            height: (aiVolume > 0.01 ? aiVolume * 40 : userVolume * 40) + 2
                          }}
                          className="w-0.5 rounded-full bg-blue-500/40"
                          transition={{ duration: 0.1, repeat: Infinity, repeatType: 'reverse', delay: i * 0.01 }}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
        </div>

        {/* Global UI Components */}
        <AnimatePresence>
        {showResources && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowResources(false)}
              className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-2xl bg-white border border-slate-200 rounded-[2.5rem] overflow-hidden shadow-2xl"
            >
              <div className="p-8 border-b border-slate-100 flex items-center justify-between glass-morphism">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-2xl bg-blue-50 flex items-center justify-center border border-blue-100">
                    <BookOpen className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Health Knowledge Base</h2>
                    <p className="text-xs text-slate-400 font-medium uppercase tracking-widest mt-0.5">Verified Medical Resources</p>
                  </div>
                </div>
                <button 
                  onClick={() => setShowResources(false)}
                  className="p-2.5 hover:bg-slate-50 rounded-2xl transition-all border border-transparent hover:border-slate-100"
                >
                  <X className="w-6 h-6 text-slate-400" />
                </button>
              </div>

              <div className="p-8 max-h-[60vh] overflow-y-auto custom-scrollbar bg-slate-50">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  {MEDICAL_RESOURCES.map((resource, idx) => (
                    <a 
                      key={idx}
                      href={resource.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group p-5 rounded-[2rem] bg-white border border-slate-200 hover:border-blue-400 hover:shadow-md transition-all flex flex-col justify-between shadow-sm relative overflow-hidden"
                    >
                      <div className="relative z-10">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-blue-600/80">
                            {resource.category}
                          </span>
                          <ExternalLink className="w-4 h-4 text-slate-200 group-hover:text-blue-500 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
                        </div>
                        <h3 className="text-base font-bold text-slate-900 mb-2 group-hover:text-blue-600 transition-colors leading-tight">
                          {resource.name}
                        </h3>
                        <p className="text-sm text-slate-500 leading-relaxed font-medium">
                          {resource.description}
                        </p>
                      </div>
                      <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/5 blur-3xl opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>
                  ))}
                </div>

                <div className="mt-10 p-6 rounded-3xl bg-red-50 border border-red-100">
                  <div className="flex items-start gap-4">
                    <AlertCircle className="w-6 h-6 text-red-500/70 shrink-0 mt-0.5" />
                    <p className="text-sm text-red-600 leading-relaxed font-medium">
                      <strong>Life-Saving Notice:</strong> These resources are for informational and preventative care only. If you are experiencing a life-threatening emergency, call your local emergency number (911, 999, etc.) immediately.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-8 bg-slate-50 border-t border-slate-100 flex justify-end">
                <button 
                  onClick={() => setShowResources(false)}
                  className="px-8 py-3 rounded-2xl bg-white text-slate-600 text-sm font-bold uppercase tracking-widest hover:bg-slate-50 hover:text-blue-600 transition-all border border-slate-200"
                >
                  Dismiss
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
      <ProfileModal isOpen={showProfileModal} onClose={() => setShowProfileModal(false)} />

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.2);
        }
      `}} />
    </div>
  );
}
