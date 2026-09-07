/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Type } from "@google/genai";
import { Mic, MicOff, Activity, Stethoscope, AlertCircle, Info, X, Volume2, VolumeX, Globe, ExternalLink, BookOpen, Phone, Trash2, Download, Send, CheckCircle2, Clock, ShieldAlert, History, Plus, ChevronLeft, MessageSquare, LogOut, User as UserIcon, Menu, Pill } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAuth } from './context/AuthContext';
import { AuthModal } from './components/AuthModal';
import { ProfileModal } from './components/ProfileModal';
import { db, auth as firebaseAuth, handleFirestoreError, OperationType } from './lib/firebase';
import { collection, query, where, orderBy, onSnapshot, addDoc, deleteDoc, doc, updateDoc, getDocs, limit, serverTimestamp } from 'firebase/firestore';
import { signOut } from 'firebase/auth';

import { MedicationPanel } from './components/MedicationPanel';
import { Medication, subscribeToMedications, addMedication, deleteMedication } from './lib/medications';

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
MEDICATION & REMINDER CAPABILITY:
- You can help users manage their medication schedule.
- When a user wants to set a reminder or add a medication, gather: name, dosage, frequency, and specific times.
- Once you have the details, use the 'addMedicationReminder' tool.
- If the user asks about their current medications, use the 'listMedications' tool to get the current list before responding.
- You can also remove reminders using 'removeMedicationReminder' if the user requests it.
- Proactively suggest setting reminders if the user mentions new medications during the consultation.
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
  const [showResources, setShowResources] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [userVolume, setUserVolume] = useState(0);
  const [aiVolume, setAiVolume] = useState(0);
  const [liveCaption, setLiveCaption] = useState<{ text: string, isUser: boolean } | null>(null);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [showMedications, setShowMedications] = useState(false);
  const [activeReminders, setActiveReminders] = useState<string[]>([]);
  const medicationsRef = useRef<Medication[]>([]);

  useEffect(() => {
    medicationsRef.current = medications;
  }, [medications]);

  // Medication Reminder Checker
  useEffect(() => {
    const checkReminders = () => {
      const now = new Date();
      const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      
      const due = medicationsRef.current.filter(med => med.times.includes(currentTime));
      if (due.length > 0) {
        const names = due.map(d => d.name);
        setActiveReminders(prev => [...new Set([...prev, ...names])]);
        // Trigger voice reminder if session is active
        if (isActive && sessionRef.current) {
          sessionRef.current.sendRealtimeInput({
            text: `SYSTEM NOTIFICATION: It is now ${currentTime}. The user has medication reminders for: ${names.join(', ')}. Please gently remind them and ask if they have taken their dose.`
          });
        }
      }
    };

    const interval = setInterval(checkReminders, 60000); // Check every minute
    checkReminders(); // Initial check
    return () => clearInterval(interval);
  }, [isActive]);

  useEffect(() => {
    if (!user) {
      setSessions([]);
      setCurrentSessionId(null);
      setMedications([]);
      return;
    }

    // Subscribe to Medications
    const unsubscribeMeds = subscribeToMedications(user.uid, (meds) => {
      setMedications(meds);
    });

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

    return () => {
      unsubscribeMeds();
      unsubscribe();
    };
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
      
      const apiKey = process.env.GEMINI_API_KEY;
      
      if (!apiKey) {
        setErrorMessage("Gemini API Key is missing. Please select an API key in the 'Secrets' menu.");
        setStatus('error');
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
                },
                {
                  name: "addMedicationReminder",
                  description: "Adds a new medication reminder for the user.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {
                      name: { type: Type.STRING, description: "Name of the medication." },
                      dosage: { type: Type.STRING, description: "Dosage amount." },
                      frequency: { type: Type.STRING, description: "Frequency (e.g. Daily, Weekly)." },
                      times: { type: Type.ARRAY, items: { type: Type.STRING }, description: "List of times in 24h format (e.g. ['08:00', '20:00'])." }
                    },
                    required: ["name", "dosage", "frequency", "times"]
                  }
                },
                {
                  name: "listMedications",
                  description: "Returns the current list of medications and reminders for the user.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {}
                  }
                },
                {
                  name: "removeMedicationReminder",
                  description: "Removes a medication reminder by name.",
                  parameters: {
                    type: Type.OBJECT,
                    properties: {
                      name: { type: Type.STRING, description: "The exact name of the medication to remove." }
                    },
                    required: ["name"]
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
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Language reported successfully." }
                    }]
                  });
                } else if (fc.name === "callEmergencyServices") {
                  window.location.href = "tel:911";
                  setErrorMessage("Emergency call initiated.");
                  
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Emergency services call initiated." }
                    }]
                  });
                } else if (fc.name === "displaySymptomAnalysis") {
                  const analysis = (fc.args as any).analysis;
                  updateTranscriptions(prev => [...prev, { analysis, isUser: false, timestamp: Date.now() }]);
                  
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: "Symptom analysis displayed to user." }
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
                    
                    sessionRef.current?.sendToolResponse({
                      functionResponses: [{
                        name: fc.name,
                        id: fc.id,
                        response: { output: `Medication ${name} registered successfully.` }
                      }]
                    });
                    setShowMedications(true);
                  }
                } else if (fc.name === "listMedications") {
                  const medsList = medicationsRef.current.map(m => `- ${m.name}: ${m.dosage} (${m.frequency}) at ${m.times.join(', ')}`).join('\n');
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      name: fc.name,
                      id: fc.id,
                      response: { output: medsList || "No medications registered." }
                    }]
                  });
                } else if (fc.name === "removeMedicationReminder") {
                  if (user) {
                    const { name } = fc.args as any;
                    const med = medicationsRef.current.find(m => m.name.toLowerCase() === name.toLowerCase());
                    if (med) {
                      await deleteMedication(user.uid, med.id);
                      sessionRef.current?.sendToolResponse({
                        functionResponses: [{
                          name: fc.name,
                          id: fc.id,
                          response: { output: `Medication ${name} removed.` }
                        }]
                      });
                    } else {
                      sessionRef.current?.sendToolResponse({
                        functionResponses: [{
                          name: fc.name,
                          id: fc.id,
                          response: { output: `Medication ${name} not found.` }
                        }]
                      });
                    }
                  }
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
          onerror: (err: any) => {
            console.error("Live API Error:", err);
            let message = "Connection lost. Please try again.";
            
            // Check for API key related errors
            const errStr = String(err).toLowerCase();
            if (errStr.includes("permission_denied") || errStr.includes("api_key_invalid") || errStr.includes("403") || errStr.includes("400")) {
              message = "Invalid or restricted API Key. Please select a valid key in 'Settings > Secrets'.";
            } else if (errStr.includes("resource_exhausted") || errStr.includes("429")) {
              message = "Quota exceeded. Please select a billing-enabled API key in 'Settings > Secrets'.";
            }
            
            setErrorMessage(message);
            setStatus('error');
            stopAudio();
          }
        }
      });

      sessionRef.current = session;

      // Initialize Audio Processing AFTER receiving session to avoid ReferenceError
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
          sessionRef.current?.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        };

        source.connect(processor);
        processor.connect(audioContextRef.current!.destination);
      }

      // Send initial text if provided
      if (initialText) {
        sessionRef.current?.sendRealtimeInput({ text: initialText });
      }

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

      {/* History Sidebar - Premium Layout */}
      <motion.aside
        initial={false}
        animate={{ width: showHistory ? 300 : 0, opacity: showHistory ? 1 : 0 }}
        className="relative flex-shrink-0 bg-slate-50 border-r border-slate-200/50 z-40 flex flex-col h-full overflow-hidden"
      >
        <div className="w-[300px] flex flex-col h-full">
          <div className="p-6 border-b border-slate-200/60 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-lg shadow-blue-600/20">
                <Stethoscope className="w-6 h-6" />
              </div>
              <div>
                <span className="font-bold text-slate-900 tracking-tight block leading-tight">RapidAid</span>
                <span className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">Medical Suite</span>
              </div>
            </div>
            <button 
              onClick={() => setShowHistory(false)}
              className="p-2 hover:bg-slate-200 rounded-lg transition-colors text-slate-400 hover:text-slate-600"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
          </div>

          <div className="p-5">
            <button
              onClick={startNewSession}
              className="w-full py-3 px-4 rounded-2xl bg-white border border-slate-200 shadow-sm hover:border-blue-400 hover:shadow-md transition-all flex items-center gap-3 text-sm font-bold text-slate-700 group"
            >
              <div className="w-7 h-7 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                <Plus className="w-4 h-4" />
              </div>
              New Consultation
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-2 custom-scrollbar">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-2 mb-3">Previous Consultations</p>
            {sessions.length === 0 ? (
              <div className="p-8 text-center bg-slate-100/50 rounded-2xl border border-dashed border-slate-200">
                <p className="text-xs font-semibold text-slate-400">No logs found</p>
              </div>
            ) : (
              sessions.map(session => (
                <button
                  key={session.id}
                  onClick={() => setCurrentSessionId(session.id)}
                  className={`w-full p-4 rounded-2xl text-left transition-all group flex flex-col gap-1 border ${
                    currentSessionId === session.id 
                      ? 'bg-white border-blue-200 shadow-lg shadow-blue-500/5' 
                      : 'bg-transparent border-transparent hover:bg-slate-200/50 text-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className={`text-[13px] font-bold truncate flex-1 ${currentSessionId === session.id ? 'text-blue-600' : 'text-slate-800'}`}>
                      {session.title}
                    </span>
                    <Trash2 
                      onClick={(e) => deleteSession(session.id, e)}
                      className={`w-4 h-4 opacity-0 group-hover:opacity-100 hover:text-red-500 transition-all ml-2 ${currentSessionId === session.id ? 'text-slate-300' : 'text-slate-400'}`} 
                    />
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <Clock className="w-3 h-3 text-slate-300" />
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-tight">
                      {new Date(session.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                    </span>
                  </div>
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
        {/* Premium Header */}
        <header className="h-20 flex items-center justify-between px-8 bg-white/70 backdrop-blur-xl border-b border-slate-200/50 sticky top-0 z-30">
          <div className="flex items-center gap-4">
            {!showHistory && (
              <button 
                onClick={() => setShowHistory(true)}
                className="p-3 hover:bg-slate-100 rounded-2xl transition-all text-slate-500 hover:text-blue-600 bg-slate-50/50"
              >
                <History className="w-5 h-5" />
              </button>
            )}
            <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${status === 'active' ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-slate-300'} animate-pulse`} />
                <span className="text-[11px] font-black text-slate-900 uppercase tracking-[0.2em]">
                  {status === 'active' ? 'Neural Link Active' : 'System Standby'}
                </span>
              </div>
              {isActive && (
                <span className="text-[10px] text-blue-600 font-bold uppercase tracking-wider mt-0.5 ml-4">
                  Voice Session in Progress
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-6 mr-4 border-r border-slate-200 pr-6 h-10">
              <div className="flex flex-col items-end">
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Language</span>
                <span className="text-xs font-bold text-slate-700">{detectedLanguage === 'Detecting...' ? 'Auto-Detect' : detectedLanguage}</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Signal</span>
                <span className="text-xs font-bold text-slate-700">{status === 'connecting' ? 'Calibrating...' : 'Encrypted'}</span>
              </div>
            </div>

            <div className="flex items-center gap-2 bg-slate-100/50 p-1 rounded-2xl">
              <button 
                onClick={() => setShowMedications(true)}
                className="p-2.5 text-slate-500 hover:text-blue-600 hover:bg-white rounded-xl transition-all shadow-none hover:shadow-sm"
                title="Medications & Reminders"
              >
                <Pill className="w-5 h-5" />
              </button>
              <button 
                onClick={() => setShowResources(true)}
                className="p-2.5 text-slate-500 hover:text-blue-600 hover:bg-white rounded-xl transition-all shadow-none hover:shadow-sm"
                title="Health Library"
              >
                <BookOpen className="w-5 h-5" />
              </button>
              {user && (
                <button 
                  onClick={() => setShowProfileModal(true)}
                  className="p-2.5 text-slate-500 hover:text-blue-600 hover:bg-white rounded-xl transition-all shadow-none hover:shadow-sm"
                  title="Profile"
                >
                  <UserIcon className="w-5 h-5" />
                </button>
              )}
            </div>

            <div className="relative">
              <button 
                onClick={() => setShowVoiceMenu(!showVoiceMenu)}
                className="flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-slate-900 border border-slate-800 text-[11px] font-bold text-white hover:bg-slate-800 transition-all shadow-lg shadow-slate-900/10"
              >
                <div className="w-2 h-2 rounded-full bg-blue-400" />
                {selectedVoice}
                <Volume2 className="w-4 h-4 opacity-50" />
              </button>
              <AnimatePresence>
                {showVoiceMenu && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 10 }}
                    className="absolute top-full right-0 mt-3 w-44 bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-2xl z-50 p-2"
                  >
                    <p className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] p-3">Voice Profile</p>
                    {voices.map(voice => (
                      <button
                        key={voice}
                        onClick={() => {
                          setSelectedVoice(voice);
                          setShowVoiceMenu(false);
                        }}
                        className={`w-full text-left px-4 py-3 text-xs font-bold rounded-xl transition-all flex items-center justify-between ${selectedVoice === voice ? 'text-blue-600 bg-blue-50' : 'text-slate-600 hover:bg-slate-50'}`}
                      >
                        {voice}
                        {selectedVoice === voice && <CheckCircle2 className="w-4 h-4" />}
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-hidden flex flex-col relative w-full bg-[#FAFAFB]">
          {/* Main Scroll Container */}
          <div className="flex-1 overflow-y-auto custom-scrollbar px-6 sm:px-10 scroll-smooth">
            <div className="max-w-3xl mx-auto py-12 space-y-12 pb-40">
              {/* Medical Disclaimer Banner */}
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-5 rounded-3xl bg-white border border-slate-200 flex items-start gap-4 shadow-sm relative overflow-hidden"
              >
                <div className="absolute top-0 left-0 w-1 h-full bg-amber-500" />
                <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center shrink-0">
                  <Info className="w-6 h-6 text-amber-600" />
                </div>
                <div>
                  <p className="text-[13px] text-slate-700 leading-relaxed font-bold">
                    Official Medical Notice
                  </p>
                  <p className="text-[12px] text-slate-500 leading-relaxed font-medium mt-1">
                    RapidAid is powered by generative AI and should not be used as a primary diagnostic tool. In case of emergency, contact local first responders immediately.
                  </p>
                </div>
              </motion.div>

              {activeReminders.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="p-6 rounded-[2.5rem] bg-blue-600 text-white shadow-xl shadow-blue-600/20 flex items-center justify-between"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center border border-white/20">
                      <Pill className="w-7 h-7" />
                    </div>
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-blue-100">Medication Reminder</p>
                      <h4 className="text-lg font-bold leading-tight">Time for {activeReminders.join(', ')}</h4>
                    </div>
                  </div>
                  <button 
                    onClick={() => setActiveReminders([])}
                    className="px-6 py-2 rounded-xl bg-white text-blue-600 text-xs font-black uppercase tracking-widest hover:bg-blue-50 transition-colors"
                  >
                    Acknowledge
                  </button>
                </motion.div>
              )}

              {/* Transcription Area */}
              <div className="space-y-10">
                {transcriptions.length === 0 ? (
                  <div className="h-[60vh] flex flex-col items-center justify-center text-center space-y-8">
                    <div className="relative">
                      <div className="absolute inset-0 bg-blue-600/10 blur-[80px] rounded-full animate-pulse" />
                      <div className="w-24 h-24 rounded-[2.5rem] bg-white border border-slate-200 flex items-center justify-center shadow-2xl relative z-10">
                        <Activity className="w-10 h-10 text-blue-600 animate-[bounce_3s_infinite]" />
                      </div>
                    </div>
                    <div className="space-y-3">
                      <h2 className="text-4xl font-bold text-slate-900 tracking-tight">How can I assist you today?</h2>
                      <p className="text-slate-500 max-w-md mx-auto text-lg font-medium leading-relaxed">
                        Speak naturally about your health concerns or type symptoms for a rapid AI-driven analysis.
                      </p>
                    </div>
                    <div className="flex items-center gap-4 pt-4">
                      <div className="flex flex-col items-center gap-2">
                        <div className="w-12 h-12 rounded-2xl bg-white border border-slate-200 flex items-center justify-center shadow-sm">
                          <Mic className="w-5 h-5 text-slate-400" />
                        </div>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Voice</span>
                      </div>
                      <div className="w-8 h-[1px] bg-slate-200" />
                      <div className="flex flex-col items-center gap-2">
                        <div className="w-12 h-12 rounded-2xl bg-white border border-slate-200 flex items-center justify-center shadow-sm">
                          <MessageSquare className="w-5 h-5 text-slate-400" />
                        </div>
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Text</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  transcriptions.map((t, i) => (
                    <motion.div
                      key={t.timestamp + i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${t.isUser ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`relative ${t.analysis ? 'w-full' : 'max-w-[85%]'}`}>
                        {t.analysis ? (
                          <div className="bg-white border border-slate-200 rounded-[2.5rem] overflow-hidden shadow-2xl shadow-slate-200/40">
                            <div className={`px-8 py-6 flex items-center justify-between border-b border-slate-100 ${
                              t.analysis.urgency === 'Emergency' ? 'bg-red-50/50' :
                              t.analysis.urgency === 'High' ? 'bg-amber-50/50' :
                              'bg-blue-50/50'
                            }`}>
                              <div className="flex items-center gap-4">
                                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${
                                  t.analysis.urgency === 'Emergency' ? 'bg-red-100 text-red-600' :
                                  t.analysis.urgency === 'High' ? 'bg-amber-100 text-amber-600' :
                                  'bg-blue-100 text-blue-600'
                                } shadow-sm`}>
                                  <ShieldAlert className="w-7 h-7" />
                                </div>
                                <div>
                                  <h3 className="font-bold text-slate-900 text-lg">Health Assessment</h3>
                                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">AI Generated Report</p>
                                </div>
                              </div>
                              <div className={`px-6 py-2 rounded-2xl text-[11px] font-black uppercase tracking-[0.2em] border-2 ${
                                t.analysis.urgency === 'Emergency' ? 'bg-red-600 text-white border-red-600' :
                                t.analysis.urgency === 'High' ? 'bg-amber-500 text-white border-amber-500' :
                                'bg-blue-600 text-white border-blue-600'
                              } shadow-lg shadow-current/20`}>
                                {t.analysis.urgency}
                              </div>
                            </div>
                            <div className="p-8 space-y-10">
                              <div className="space-y-5">
                                <div className="flex items-center gap-3">
                                  <span className="w-8 h-[1px] bg-blue-600" />
                                  <p className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-400">Potential Conditions</p>
                                </div>
                                <div className="grid grid-cols-1 gap-4">
                                  {t.analysis.potentialConditions.map((c, idx) => (
                                    <div key={idx} className="p-6 rounded-[2rem] bg-slate-50 border border-slate-100 hover:bg-white hover:border-blue-200 hover:shadow-sm transition-all group">
                                      <div className="flex items-center justify-between mb-2">
                                        <h4 className="text-base font-bold text-slate-900 group-hover:text-blue-600 transition-colors">{c.name}</h4>
                                        <div className="flex items-center gap-1.5 px-3 py-1 bg-blue-50 rounded-full border border-blue-100">
                                          <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                                          <span className="text-[10px] font-black text-blue-600 uppercase tracking-widest">{c.likelihood}</span>
                                        </div>
                                      </div>
                                      <p className="text-sm text-slate-500 leading-relaxed font-medium">{c.description}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-10 pt-8 border-t border-slate-100">
                                <div className="space-y-5">
                                  <div className="flex items-center gap-3">
                                    <span className="w-8 h-[1px] bg-blue-600" />
                                    <p className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-400">Patient Roadmap</p>
                                  </div>
                                  <ul className="space-y-4">
                                    {t.analysis.recommendations.map((r, idx) => (
                                      <li key={idx} className="flex gap-4 p-4 rounded-2xl bg-white border border-slate-100 shadow-sm">
                                        <div className="w-6 h-6 rounded-lg bg-green-50 flex items-center justify-center shrink-0 border border-green-100 mt-0.5">
                                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                                        </div>
                                        <p className="text-[13px] text-slate-600 font-bold leading-relaxed">{r}</p>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                                {t.analysis.disclaimer && (
                                  <div className="flex flex-col justify-end">
                                    <div className="p-6 rounded-3xl bg-indigo-50 border border-indigo-100">
                                      <p className="text-[12px] text-indigo-700/80 italic font-bold leading-relaxed">
                                        {t.analysis.disclaimer}
                                      </p>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className={`p-5 rounded-[2rem] ${
                            t.isUser 
                              ? 'bg-slate-900 text-white shadow-xl shadow-slate-900/10' 
                              : 'bg-white border border-slate-200 text-slate-800 shadow-sm'
                          }`}>
                            <p className="text-[15px] leading-relaxed font-bold">{t.text}</p>
                            <div className={`mt-3 flex items-center justify-between border-t ${t.isUser ? 'border-white/10 pt-2' : 'border-slate-100 pt-2'} `}>
                              <div className={`flex items-center gap-2 ${t.isUser ? 'text-slate-400' : 'text-slate-400'}`}>
                                <Clock className="w-3 h-3" />
                                <span className="text-[10px] font-bold uppercase tracking-widest">
                                  {new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                              </div>
                              {!t.isUser && (
                                <button 
                                  onClick={() => speakText(t.text!)} 
                                  className="w-8 h-8 rounded-lg bg-slate-50 flex items-center justify-center text-slate-400 hover:text-blue-600 hover:bg-blue-50 transition-all border border-slate-100"
                                >
                                  <Volume2 className="w-4 h-4" />
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
          <div className="absolute bottom-28 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6 z-40 pointer-events-none space-y-4">
            <AnimatePresence>
              {errorMessage && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="p-5 rounded-[2rem] bg-slate-900 text-white shadow-2xl flex items-center justify-between pointer-events-auto border border-slate-800"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-red-500/10 flex items-center justify-center border border-red-500/20">
                      <AlertCircle className="w-6 h-6 text-red-500" />
                    </div>
                    <div>
                      <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">System Error</p>
                      <span className="text-sm font-bold">{errorMessage}</span>
                    </div>
                  </div>
                  <button 
                    onClick={() => setErrorMessage(null)} 
                    className="p-2 hover:bg-white/10 rounded-xl transition-colors"
                  >
                    <X className="w-5 h-5 text-slate-400" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {isActive && liveCaption && (
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="p-6 rounded-[2.5rem] bg-white/80 border border-white shadow-2xl backdrop-blur-2xl text-center relative overflow-hidden"
                >
                  <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
                  <div className="flex items-center justify-center gap-3 mb-3">
                    <div className="flex gap-1">
                      <motion.span animate={{ height: [4, 12, 4] }} transition={{ repeat: Infinity, duration: 1 }} className="w-0.5 bg-blue-500 rounded-full" />
                      <motion.span animate={{ height: [8, 4, 8] }} transition={{ repeat: Infinity, duration: 1, delay: 0.2 }} className="w-0.5 bg-blue-500 rounded-full" />
                      <motion.span animate={{ height: [4, 12, 4] }} transition={{ repeat: Infinity, duration: 1, delay: 0.4 }} className="w-0.5 bg-blue-500 rounded-full" />
                    </div>
                    <span className="text-[10px] font-black uppercase tracking-[0.3em] text-blue-600">Voice Recognition</span>
                  </div>
                  <p className="text-slate-900 text-lg font-bold italic leading-tight">
                    <span className="opacity-40 italic">"</span>
                    {liveCaption.text}
                    <span className="opacity-40 italic">"</span>
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Premium Docked Control Bar */}
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-[#FAFAFB] via-[#FAFAFB]/90 to-transparent pt-20 pb-10 px-6 z-30 pointer-events-none">
            <div className="max-w-3xl mx-auto w-full pointer-events-auto">
              <div className="relative">
                <div className="absolute -inset-4 bg-blue-600/5 blur-[40px] rounded-full opacity-0 group-focus-within:opacity-100 transition-opacity" />
                <div className="relative bg-white border border-slate-200/60 rounded-[2.5rem] shadow-2xl shadow-slate-200/60 p-3 pl-8 flex items-center gap-4 backdrop-blur-md">
                  <div className="flex-1 flex items-center gap-5">
                    <div className={`w-10 h-10 rounded-2xl flex items-center justify-center transition-all ${isActive ? 'bg-blue-600 shadow-lg shadow-blue-600/20' : 'bg-slate-100'}`}>
                      <Activity className={`w-5 h-5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                    </div>
                    <form 
                      onSubmit={handleSendText}
                      className="flex-1"
                    >
                      <input
                        type="text"
                        value={textInput}
                        onChange={(e) => setTextInput(e.target.value)}
                        placeholder="Consult via secure text..."
                        className="w-full bg-transparent border-none outline-none text-base font-bold text-slate-900 placeholder:text-slate-400 placeholder:font-medium"
                      />
                    </form>
                  </div>

                  <div className="flex items-center gap-2">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setIsMuted(!isMuted)}
                      disabled={!isActive}
                      className={`w-12 h-12 rounded-2xl transition-all flex items-center justify-center ${
                        isMuted 
                          ? 'bg-red-50 text-red-500 border border-red-100' 
                          : 'bg-slate-50 text-slate-500 hover:text-blue-600 hover:bg-blue-50 hover:border-blue-100 border border-slate-100'
                      } disabled:opacity-30 disabled:grayscale`}
                    >
                      {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                    </motion.button>

                    <button
                      onClick={isActive ? endSession : () => startSession()}
                      disabled={status === 'connecting'}
                      className={`h-14 px-8 rounded-[1.75rem] font-black text-xs uppercase tracking-[0.2em] transition-all shadow-xl flex items-center gap-3 overflow-hidden relative group/btn ${
                        isActive 
                          ? 'bg-slate-900 text-white shadow-slate-900/20' 
                          : 'bg-blue-600 text-white shadow-blue-600/30 hover:bg-blue-700'
                      } disabled:opacity-50`}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 -translate-x-full group-hover/btn:translate-x-full transition-transform duration-1000" />
                      {status === 'connecting' ? (
                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      ) : isActive ? (
                        'Terminate'
                      ) : (
                        <>
                          <Volume2 className="w-4 h-4" />
                          Initialize
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Hardware-style visualization */}
                {isActive && (
                  <div className="absolute -bottom-4 inset-x-16 h-8 flex items-end justify-center gap-1.5 pointer-events-none opacity-40">
                    {Array.from({ length: 32 }).map((_, i) => (
                      <motion.div
                        key={i}
                        animate={{ 
                          height: (aiVolume > 0.01 ? aiVolume * 30 : userVolume * 30) + 4
                        }}
                        className="w-1 rounded-t-full bg-blue-600"
                        transition={{ duration: 0.1, repeat: Infinity, repeatType: 'reverse', delay: i * 0.005 }}
                      />
                    ))}
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

      <AnimatePresence>
        {showMedications && user && (
          <div className="fixed inset-0 z-[200] flex justify-end">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowMedications(false)}
              className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="relative w-full max-w-md bg-white shadow-2xl h-full border-l border-slate-200 z-[201]"
            >
              <MedicationPanel 
                userId={user.uid} 
                medications={medications} 
                onClose={() => setShowMedications(false)} 
              />
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
