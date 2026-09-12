/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Mic, 
  Activity, 
  Stethoscope, 
  AlertCircle, 
  Info, 
  X, 
  Volume2, 
  VolumeX, 
  ExternalLink, 
  BookOpen, 
  Trash2, 
  Download, 
  Send, 
  CheckCircle2, 
  Clock, 
  ShieldAlert, 
  History, 
  Plus, 
  ChevronLeft, 
  MessageSquare, 
  User as UserIcon, 
  Pill, 
  Copy, 
  Check, 
  Square, 
  Sparkles, 
  Radio 
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';
import { useAuth } from './context/AuthContext';
import { AuthModal } from './components/AuthModal';
import { ProfileModal } from './components/ProfileModal';
import { MedicationPanel } from './components/MedicationPanel';
import { VoiceCompanion } from './components/VoiceCompanion';
import { db, auth as firebaseAuth, handleFirestoreError, OperationType } from './lib/firebase';
import { 
  collection, 
  query, 
  where, 
  orderBy, 
  onSnapshot, 
  addDoc, 
  deleteDoc, 
  doc, 
  updateDoc, 
  getDocs, 
  limit, 
  serverTimestamp 
} from 'firebase/firestore';
import { signOut } from 'firebase/auth';
import { Medication, subscribeToMedications } from './lib/medications';
import { streamClinicalChat } from './lib/geminiChat';
import { Transcription, SymptomAnalysis, Session } from './types';

const MEDICAL_RESOURCES = [
  {
    name: "World Health Organization (WHO)",
    description: "Global health guidelines, disease prevention, and emergency updates.",
    url: "https://www.who.int",
    category: "Global Health"
  },
  {
    name: "Mayo Clinic",
    description: "Comprehensive medical information on clinical conditions and treatments.",
    url: "https://www.mayoclinic.org",
    category: "Medical Reference"
  },
  {
    name: "CDC (Centers for Disease Control)",
    description: "Public health information, vaccine guidance, and infection tracking.",
    url: "https://www.cdc.gov",
    category: "Public Health"
  },
  {
    name: "MedlinePlus (National Library of Medicine)",
    description: "Trusted consumer health encyclopedia and prescription drug references.",
    url: "https://medlineplus.gov",
    category: "Health Education"
  },
  {
    name: "National Institutes of Health (NIH)",
    description: "Leading biomedical research and clinical trial outcomes.",
    url: "https://www.nih.gov",
    category: "Research"
  }
];

const SUGGESTED_PROMPTS = [
  {
    icon: "🤒",
    title: "Symptom Evaluation",
    prompt: "I have had a mild fever (100.8°F), body aches, and a scratchy dry cough for 2 days. Can you evaluate my symptoms?"
  },
  {
    icon: "💊",
    title: "Medication Safety",
    prompt: "Can I take ibuprofen for a headache if I am already taking blood pressure medication?"
  },
  {
    icon: "🫀",
    title: "Chest Tightness Triage",
    prompt: "What are the key clinical differences between anxiety-induced chest tightness and a cardiovascular issue?"
  },
  {
    icon: "🩹",
    title: "First Aid Protocol",
    prompt: "What is the recommended first aid procedure for a minor second-degree steam burn on the forearm?"
  }
];

export default function App() {
  const { user, userData } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Sessions and conversation history
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Transcription[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Voice Companion on the side
  const [showVoiceCompanion, setShowVoiceCompanion] = useState(true);

  // Chat input and generation state
  const [textInput, setTextInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Modals & Navigation
  const [showResources, setShowResources] = useState(false);
  const [showMedications, setShowMedications] = useState(false);
  const [medications, setMedications] = useState<Medication[]>([]);
  const [activeReminders, setActiveReminders] = useState<string[]>([]);

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null);
  const isGeneratingRef = useRef<boolean>(false);
  const currentSessionIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Keep currentSessionIdRef synchronized
  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);

  // Active session metadata & active messages list
  const activeSession = sessions.find(s => s.id === currentSessionId);
  const currentMessages = messages;

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Medication reminders checker
  useEffect(() => {
    if (medications.length === 0) return;
    const checkReminders = () => {
      const now = new Date();
      const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      const due = medications.filter(med => med.times.includes(currentTime));
      if (due.length > 0) {
        const names = due.map(d => d.name);
        setActiveReminders(prev => [...new Set([...prev, ...names])]);
      }
    };

    const interval = setInterval(checkReminders, 60000);
    checkReminders();
    return () => clearInterval(interval);
  }, [medications]);

  // Subscribe to User's Medications & Consultations Metadata
  useEffect(() => {
    if (!user) {
      setSessions([]);
      setCurrentSessionId(null);
      currentSessionIdRef.current = null;
      setMedications([]);
      return;
    }

    const unsubscribeMeds = subscribeToMedications(user.uid, (meds) => {
      setMedications(meds);
    });

    const q = query(
      collection(db, 'sessions'),
      where('userId', '==', user.uid),
      orderBy('timestamp', 'desc'),
      limit(50)
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const sessionData: Session[] = snapshot.docs.map(sessionDoc => ({
        id: sessionDoc.id,
        title: sessionDoc.data().title || 'Medical Consultation',
        timestamp: sessionDoc.data().timestamp || Date.now()
      }));
      setSessions(sessionData);

      // Auto-select latest session if none selected and not actively generating
      if (sessionData.length > 0 && !currentSessionIdRef.current && !isGeneratingRef.current) {
        const initialId = sessionData[0].id;
        setCurrentSessionId(initialId);
        currentSessionIdRef.current = initialId;
      }
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, 'sessions');
    });

    return () => {
      unsubscribeMeds();
      unsubscribe();
    };
  }, [user]);

  // Subscribe to Messages for ACTIVE Session
  useEffect(() => {
    if (!user) return;
    if (!currentSessionId) {
      if (!isGeneratingRef.current) {
        setMessages([]);
      }
      return;
    }

    const q = query(
      collection(db, `sessions/${currentSessionId}/messages`),
      where('userId', '==', user.uid)
    );

    const unsubscribe = onSnapshot(q, (snapshot) => {
      // Do not overwrite local state while response is actively streaming
      if (isGeneratingRef.current) return;

      const msgs = snapshot.docs.map(d => ({
        id: d.id,
        ...d.data()
      } as Transcription));

      // Sort chronologically in memory (avoids missing composite index errors)
      msgs.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
      setMessages(msgs);
    }, (error) => {
      handleFirestoreError(error, OperationType.LIST, `sessions/${currentSessionId}/messages`);
    });

    return () => unsubscribe();
  }, [user, currentSessionId]);

  // Start a fresh consultation session
  const startNewSession = async () => {
    if (isGenerating && abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsGenerating(false);
      isGeneratingRef.current = false;
    }

    setMessages([]);
    setCurrentSessionId(null);
    currentSessionIdRef.current = null;
    setShowHistory(false);
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm("Delete this consultation record?")) {
      try {
        await deleteDoc(doc(db, 'sessions', id));
        if (currentSessionId === id) {
          const remaining = sessions.filter(s => s.id !== id);
          const nextId = remaining.length > 0 ? remaining[0].id : null;
          setCurrentSessionId(nextId);
          currentSessionIdRef.current = nextId;
          if (!nextId) {
            setMessages([]);
          }
        }
      } catch (e) {
        handleFirestoreError(e, OperationType.DELETE, `sessions/${id}`);
      }
    }
  };

  // Add transcription (from Voice Companion or voice speech)
  const addTranscriptionToRecord = useCallback(async (transcription: Transcription) => {
    setMessages(prev => [...prev, transcription]);

    if (!user) return;

    let activeId = currentSessionIdRef.current;
    if (!activeId) {
      try {
        const sessionRef = await addDoc(collection(db, 'sessions'), {
          userId: user.uid,
          title: transcription.text ? transcription.text.slice(0, 32) : `Consultation ${new Date().toLocaleDateString()}`,
          timestamp: Date.now(),
          updatedAt: serverTimestamp()
        });
        activeId = sessionRef.id;
        currentSessionIdRef.current = activeId;
        setCurrentSessionId(activeId);
      } catch (e) {
        handleFirestoreError(e, OperationType.CREATE, 'sessions');
        return;
      }
    }

    try {
      await addDoc(collection(db, `sessions/${activeId}/messages`), {
        ...transcription,
        sessionId: activeId,
        userId: user.uid
      });
    } catch (e) {
      handleFirestoreError(e, OperationType.CREATE, `sessions/${activeId}/messages`);
    }
  }, [user]);

  // Primary Clinical Text Chat Handler
  const handleSendText = async (e?: React.FormEvent, customPrompt?: string) => {
    e?.preventDefault();
    const query = (customPrompt || textInput).trim();
    if (!query || isGenerating) return;

    setTextInput('');
    setErrorMessage(null);

    // 1. Create and display User message immediately
    const userMessage: Transcription = {
      id: `user-${Date.now()}`,
      text: query,
      isUser: true,
      timestamp: Date.now(),
      fromVoice: false
    };

    // 2. Prepare pending Assistant message for real-time streaming
    const assistantMessageId = `assistant-${Date.now()}`;
    const initialAssistantMessage: Transcription = {
      id: assistantMessageId,
      text: "",
      isUser: false,
      timestamp: Date.now(),
      fromVoice: false
    };

    // Update UI state for instant response
    setMessages(prev => [...prev, userMessage, initialAssistantMessage]);

    setIsGenerating(true);
    isGeneratingRef.current = true;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // If user is authenticated, ensure a session document is created immediately
    let activeId = currentSessionIdRef.current;
    if (user && !activeId) {
      try {
        const sessionRef = await addDoc(collection(db, 'sessions'), {
          userId: user.uid,
          title: query.slice(0, 32) + (query.length > 32 ? '...' : ''),
          timestamp: Date.now(),
          updatedAt: serverTimestamp()
        });
        activeId = sessionRef.id;
        currentSessionIdRef.current = activeId;
        setCurrentSessionId(activeId);
      } catch (err) {
        console.warn("Could not create session document:", err);
      }
    }

    try {
      let currentAccumulated = "";
      let capturedAnalysis: SymptomAnalysis | undefined;

      const result = await streamClinicalChat({
        userMessage: query,
        history: messages,
        profile: userData?.healthProfile,
        activeMedications: medications.map(m => m.name),
        signal: controller.signal,
        onChunk: (chunkText) => {
          currentAccumulated = chunkText;
          setMessages(prev => prev.map(m => 
            m.id === assistantMessageId ? { ...m, text: chunkText } : m
          ));
        },
        onAnalysis: (analysis) => {
          capturedAnalysis = analysis;
        }
      });

      const finalAssistantMessage: Transcription = {
        id: assistantMessageId,
        text: result.text || currentAccumulated,
        analysis: result.analysis || capturedAnalysis,
        isUser: false,
        timestamp: Date.now(),
        fromVoice: false
      };

      // Set final message content in state
      setMessages(prev => prev.map(m =>
        m.id === assistantMessageId ? finalAssistantMessage : m
      ));

      // Persist permanently to Firestore if logged in
      if (user && activeId) {
        try {
          // Save User Msg
          await addDoc(collection(db, `sessions/${activeId}/messages`), {
            text: userMessage.text,
            isUser: true,
            timestamp: userMessage.timestamp,
            sessionId: activeId,
            userId: user.uid
          });

          // Save Assistant Msg
          await addDoc(collection(db, `sessions/${activeId}/messages`), {
            text: finalAssistantMessage.text,
            analysis: finalAssistantMessage.analysis || null,
            isUser: false,
            timestamp: finalAssistantMessage.timestamp,
            sessionId: activeId,
            userId: user.uid
          });

          // Update Title if it was initial question
          if (messages.length <= 2) {
            await updateDoc(doc(db, 'sessions', activeId), {
              title: query.slice(0, 32) + (query.length > 32 ? '...' : ''),
              updatedAt: serverTimestamp()
            });
          } else {
            await updateDoc(doc(db, 'sessions', activeId), {
              updatedAt: serverTimestamp()
            });
          }
        } catch (saveErr) {
          console.error("Error saving consultation messages:", saveErr);
        }
      }
    } catch (err: any) {
      if (!controller.signal.aborted) {
        console.error("Clinical chat stream failed:", err);
        setErrorMessage(err?.message || "Could not generate clinical consultation. Please try again.");
        setMessages(prev => prev.filter(m => m.id !== assistantMessageId));
      }
    } finally {
      setIsGenerating(false);
      isGeneratingRef.current = false;
      abortControllerRef.current = null;
    }
  };

  const handleStopGenerating = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsGenerating(false);
    isGeneratingRef.current = false;
  };

  // Text-To-Speech
  const speakText = (text: string, msgId: string) => {
    if (!('speechSynthesis' in window)) return;

    if (speakingMessageId === msgId) {
      window.speechSynthesis.cancel();
      setSpeakingMessageId(null);
      return;
    }

    window.speechSynthesis.cancel();
    // Strip markdown formatting for cleaner speech
    const cleanSpeech = text.replace(/[*#_`~\[\]]/g, '').trim();
    const utterance = new SpeechSynthesisUtterance(cleanSpeech);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.onend = () => setSpeakingMessageId(null);
    utterance.onerror = () => setSpeakingMessageId(null);

    setSpeakingMessageId(msgId);
    window.speechSynthesis.speak(utterance);
  };

  // Copy text helper
  const copyMessage = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Download consultation transcript
  const downloadTranscript = () => {
    if (currentMessages.length === 0) return;

    const content = currentMessages.map(t => {
      const role = t.isUser ? "Patient" : "RapidAid Assistant";
      const time = new Date(t.timestamp).toLocaleTimeString();
      const channel = t.fromVoice ? " [Voice Call]" : " [Text Chat]";

      if (t.analysis) {
        const symptoms = t.analysis.symptoms.join(', ');
        const conditions = t.analysis.potentialConditions.map(c => `${c.name} (${c.likelihood}): ${c.description}`).join('\n- ');
        const recs = t.analysis.recommendations.join('\n- ');
        return `[${time}] ${role}${channel} [CLINICAL ASSESSMENT]:\nReported Symptoms: ${symptoms}\nTriage Urgency: ${t.analysis.urgency}\n\nPotential Causes:\n- ${conditions}\n\nClinical Recommendations:\n- ${recs}\n`;
      }

      return `[${time}] ${role}${channel}:\n${t.text}\n`;
    }).join('\n----------------------------------------\n\n');

    const header = `RAPIDAID CLINICAL CONSULTATION RECORD\nGenerated: ${new Date().toLocaleString()}\nPatient: ${user?.email || 'Guest Patient'}\n\n========================================\n\n`;
    const blob = new Blob([header + content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `RapidAid_Consultation_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearCurrentChat = async () => {
    if (window.confirm("Clear this consultation chat?")) {
      setMessages([]);
      if (user && currentSessionId) {
        try {
          const q = query(
            collection(db, `sessions/${currentSessionId}/messages`),
            where('userId', '==', user.uid)
          );
          const snap = await getDocs(q);
          const deletions = snap.docs.map(d => deleteDoc(d.ref));
          await Promise.all(deletions);
        } catch (e) {
          console.error("Error clearing consultation messages:", e);
        }
      }
    }
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] text-slate-900 font-sans selection:bg-blue-500/20 overflow-hidden">
      {/* Consultations History Drawer (Left) */}
      <motion.aside
        initial={false}
        animate={{ width: showHistory ? 320 : 0, opacity: showHistory ? 1 : 0 }}
        className="relative shrink-0 bg-white border-r border-slate-200 z-40 flex flex-col h-full overflow-hidden shadow-xl lg:shadow-none"
      >
        <div className="w-80 flex flex-col h-full">
          {/* Brand Header */}
          <div className="p-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/60">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
                <Stethoscope className="w-5 h-5" />
              </div>
              <div>
                <span className="font-extrabold text-slate-900 tracking-tight block text-base leading-tight">RapidAid</span>
                <span className="text-[10px] text-blue-600 font-bold uppercase tracking-widest">Medical Assistant</span>
              </div>
            </div>
            <button
              onClick={() => setShowHistory(false)}
              className="p-1.5 hover:bg-slate-200/60 rounded-xl transition-colors text-slate-400 hover:text-slate-700"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
          </div>

          {/* New Consultation CTA */}
          <div className="p-4">
            <button
              onClick={startNewSession}
              className="w-full py-3 px-4 rounded-2xl bg-blue-50 border border-blue-200/80 hover:bg-blue-100/70 hover:border-blue-300 transition-all flex items-center gap-3 text-xs font-bold text-blue-700 group shadow-sm"
            >
              <div className="w-7 h-7 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-sm">
                <Plus className="w-4 h-4" />
              </div>
              <span>New Consultation</span>
            </button>
          </div>

          {/* Past Consultations List */}
          <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1.5 custom-scrollbar">
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-3 mb-2">History Records</p>
            {user ? (
              sessions.length === 0 ? (
                <div className="p-6 text-center bg-slate-50 rounded-2xl border border-dashed border-slate-200">
                  <p className="text-xs font-semibold text-slate-400">No consultation records yet</p>
                </div>
              ) : (
                sessions.map(session => (
                  <button
                    key={session.id}
                    onClick={() => {
                      if (isGenerating && abortControllerRef.current) {
                        abortControllerRef.current.abort();
                        setIsGenerating(false);
                        isGeneratingRef.current = false;
                      }
                      setCurrentSessionId(session.id);
                      currentSessionIdRef.current = session.id;
                      if (window.innerWidth < 1024) setShowHistory(false);
                    }}
                    className={`w-full p-3.5 rounded-2xl text-left transition-all group flex flex-col gap-1 border ${
                      currentSessionId === session.id
                        ? 'bg-blue-50/70 border-blue-200 shadow-sm'
                        : 'bg-transparent border-transparent hover:bg-slate-100/70 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-xs font-bold truncate flex-1 ${currentSessionId === session.id ? 'text-blue-700' : 'text-slate-800'}`}>
                        {session.title}
                      </span>
                      <Trash2
                        onClick={(e) => deleteSession(session.id, e)}
                        className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 hover:text-red-600 transition-opacity ml-2 text-slate-400 shrink-0"
                      />
                    </div>
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-medium">
                      <Clock className="w-3 h-3" />
                      <span>{new Date(session.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                    </div>
                  </button>
                ))
              )
            ) : (
              <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs">
                <span className="font-bold block mb-1">Guest Mode</span>
                <span className="text-[11px] text-amber-700 leading-relaxed block mb-2">
                  Sign in to save and sync your medical consultations across devices.
                </span>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="w-full py-2 bg-amber-600 text-white rounded-xl font-bold text-xs hover:bg-amber-700 transition-colors"
                >
                  Sign In / Register
                </button>
              </div>
            )}
          </div>

          {/* User Account / Sign In Bar */}
          <div className="p-3 border-t border-slate-100 bg-slate-50/50">
            {user ? (
              <div className="flex items-center gap-3 p-2.5 rounded-xl bg-white border border-slate-200 shadow-sm">
                <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600 font-bold text-xs">
                  {user.email ? user.email[0].toUpperCase() : 'P'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-bold text-slate-900 truncate">{user.displayName || user.email}</p>
                  <button
                    onClick={() => signOut(firebaseAuth)}
                    className="text-[10px] font-bold text-slate-400 hover:text-red-500 uppercase tracking-wider transition-colors"
                  >
                    Log out
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="w-full py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold tracking-wide hover:bg-slate-800 transition-colors"
              >
                Sign In to Save Records
              </button>
            )}
          </div>
        </div>
      </motion.aside>

      {/* Main Medical Workspace (Center: Chat & Consultation) */}
      <div className="flex-1 flex flex-col min-w-0 relative h-full bg-[#F8FAFC]">
        {/* Top Header Bar */}
        <header className="h-18 px-5 lg:px-8 border-b border-slate-200 bg-white/90 backdrop-blur-md flex items-center justify-between sticky top-0 z-20">
          <div className="flex items-center gap-3.5">
            {!showHistory && (
              <button
                onClick={() => setShowHistory(true)}
                className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors"
                title="Consultation History"
              >
                <History className="w-4 h-4" />
              </button>
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
                <h1 className="text-sm font-black text-slate-900 tracking-tight">
                  {user ? (activeSession?.title || 'Clinical Consultation') : 'RapidAid Healthcare Consultation'}
                </h1>
              </div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                Clinical Intelligence System • Gemini 3.8
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Quick action buttons */}
            <button
              onClick={() => setShowMedications(true)}
              className="px-3 py-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-bold text-slate-700 transition-colors flex items-center gap-1.5"
              title="Medication Schedule"
            >
              <Pill className="w-3.5 h-3.5 text-blue-600" />
              <span className="hidden sm:inline">Medications</span>
              {medications.length > 0 && (
                <span className="ml-0.5 px-1.5 py-0.2 bg-blue-100 text-blue-700 rounded-full text-[10px]">
                  {medications.length}
                </span>
              )}
            </button>

            <button
              onClick={() => setShowResources(true)}
              className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors"
              title="Medical Knowledge Library"
            >
              <BookOpen className="w-4 h-4" />
            </button>

            {user && (
              <button
                onClick={() => setShowProfileModal(true)}
                className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors"
                title="Patient Health Profile"
              >
                <UserIcon className="w-4 h-4" />
              </button>
            )}

            {currentMessages.length > 0 && (
              <>
                <button
                  onClick={downloadTranscript}
                  className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors"
                  title="Export Consultation Transcript"
                >
                  <Download className="w-4 h-4" />
                </button>
                <button
                  onClick={clearCurrentChat}
                  className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-red-500 transition-colors"
                  title="Clear Chat"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </>
            )}

            {/* Prominent Voice Option on the side toggle button */}
            <button
              onClick={() => setShowVoiceCompanion(prev => !prev)}
              className={`px-3.5 py-2 rounded-xl border text-xs font-bold transition-all flex items-center gap-2 shadow-sm ${
                showVoiceCompanion
                  ? 'bg-blue-600 border-blue-600 text-white shadow-blue-500/20'
                  : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50'
              }`}
              title={showVoiceCompanion ? "Hide Voice Companion" : "Show Voice Companion on the side"}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Voice Assistant</span>
              <span className={`w-1.5 h-1.5 rounded-full ${showVoiceCompanion ? 'bg-white animate-pulse' : 'bg-slate-400'}`} />
            </button>
          </div>
        </header>

        {/* Main Chat Stream Area */}
        <main className="flex-1 overflow-y-auto custom-scrollbar px-4 sm:px-8 py-6 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6 pb-28">
            {/* Clinical Notice Banner */}
            <div className="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex items-start gap-3.5">
              <div className="w-8 h-8 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center shrink-0">
                <Info className="w-4 h-4 text-amber-600" />
              </div>
              <div className="text-xs">
                <span className="font-bold text-slate-900 block">Informational Clinical Assistant</span>
                <span className="text-slate-500 leading-relaxed font-medium">
                  RapidAid provides evidence-based guidance and symptom triage. It does not replace emergency medical services or physician evaluations. In severe emergencies, call 911 immediately.
                </span>
              </div>
            </div>

            {/* Active Medication Alert if Due */}
            {activeReminders.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-2xl bg-blue-600 text-white shadow-lg flex items-center justify-between"
              >
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-white/20 flex items-center justify-center">
                    <Pill className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider font-extrabold text-blue-200">Scheduled Dose Due</p>
                    <p className="text-sm font-bold">Time to take: {activeReminders.join(', ')}</p>
                  </div>
                </div>
                <button
                  onClick={() => setActiveReminders([])}
                  className="px-3.5 py-1.5 rounded-xl bg-white text-blue-600 font-bold text-xs hover:bg-blue-50 transition-colors"
                >
                  Acknowledge
                </button>
              </motion.div>
            )}

            {/* Empty State / Suggested Medical Prompts */}
            {currentMessages.length === 0 && (
              <div className="py-10 text-center space-y-8">
                <div className="relative inline-block">
                  <div className="w-20 h-20 rounded-3xl bg-blue-50 border border-blue-100 flex items-center justify-center mx-auto shadow-sm">
                    <Activity className="w-9 h-9 text-blue-600" />
                  </div>
                </div>
                <div className="space-y-2 max-w-lg mx-auto">
                  <h2 className="text-2xl font-bold text-slate-900 tracking-tight">How can RapidAid support your health today?</h2>
                  <p className="text-sm text-slate-500 font-medium leading-relaxed">
                    Type your symptoms or health questions below for immediate clinical triage, medication guidance, and personalized care roadmaps.
                  </p>
                </div>

                {/* Suggested Prompt Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left max-w-2xl mx-auto pt-2">
                  {SUGGESTED_PROMPTS.map((item, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendText(undefined, item.prompt)}
                      className="p-4 rounded-2xl bg-white border border-slate-200/80 hover:border-blue-400 hover:shadow-md transition-all group flex flex-col justify-between text-left"
                    >
                      <div>
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-lg">{item.icon}</span>
                          <span className="text-xs font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                            {item.title}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 leading-relaxed font-medium line-clamp-2">
                          "{item.prompt}"
                        </p>
                      </div>
                      <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider mt-3 flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                        Ask RapidAid &rarr;
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Conversation Messages */}
            {currentMessages.map((msg, idx) => (
              <motion.div
                key={msg.id || msg.timestamp + idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`relative ${msg.analysis ? 'w-full' : 'max-w-[85%]'}`}>
                  {msg.isUser ? (
                    // User Message Bubble
                    <div className="p-4 rounded-3xl bg-slate-900 text-white shadow-md shadow-slate-900/10">
                      <p className="text-sm leading-relaxed font-medium whitespace-pre-wrap">{msg.text}</p>
                      <div className="mt-2 flex items-center justify-end gap-2 text-[10px] text-slate-400">
                        {msg.fromVoice && (
                          <span className="px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-bold uppercase tracking-wider">
                            Voice
                          </span>
                        )}
                        <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                    </div>
                  ) : (
                    // Assistant Clinical Response Card
                    <div className="space-y-3 w-full">
                      {/* Structured Symptom Analysis Card if generated */}
                      {msg.analysis && (
                        <div className="bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-lg shadow-slate-200/50 mb-3">
                          {/* Card Header with Urgency Pill */}
                          <div className={`px-6 py-4 flex items-center justify-between border-b ${
                            msg.analysis.urgency === 'Emergency' ? 'bg-red-50/80 border-red-200 text-red-700' :
                            msg.analysis.urgency === 'High' ? 'bg-amber-50/80 border-amber-200 text-amber-700' :
                            'bg-blue-50/80 border-blue-200 text-blue-700'
                          }`}>
                            <div className="flex items-center gap-3">
                              <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${
                                msg.analysis.urgency === 'Emergency' ? 'bg-red-100 text-red-600' :
                                msg.analysis.urgency === 'High' ? 'bg-amber-100 text-amber-600' :
                                'bg-blue-100 text-blue-600'
                              }`}>
                                <ShieldAlert className="w-5 h-5" />
                              </div>
                              <div>
                                <h4 className="text-xs font-bold uppercase tracking-wider">Clinical Symptom Assessment</h4>
                                <p className="text-[11px] font-medium opacity-80">
                                  Symptoms: {msg.analysis.symptoms.join(', ')}
                                </p>
                              </div>
                            </div>
                            <span className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider ${
                              msg.analysis.urgency === 'Emergency' ? 'bg-red-600 text-white' :
                              msg.analysis.urgency === 'High' ? 'bg-amber-600 text-white' :
                              'bg-blue-600 text-white'
                            }`}>
                              {msg.analysis.urgency} Urgency
                            </span>
                          </div>

                          {/* Card Content */}
                          <div className="p-6 space-y-6">
                            {/* Potential Conditions */}
                            <div>
                              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-3">
                                Potential Clinical Causes
                              </p>
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {msg.analysis.potentialConditions.map((cond, cIdx) => (
                                  <div key={cIdx} className="p-3.5 rounded-2xl bg-slate-50 border border-slate-100">
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="text-xs font-bold text-slate-900">{cond.name}</span>
                                      <span className="text-[10px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
                                        {cond.likelihood}
                                      </span>
                                    </div>
                                    <p className="text-[11px] text-slate-500 font-medium leading-relaxed">{cond.description}</p>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Recommendations */}
                            <div>
                              <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-3">
                                Recommended Action Roadmap
                              </p>
                              <div className="space-y-2">
                                {msg.analysis.recommendations.map((rec, rIdx) => (
                                  <div key={rIdx} className="flex items-start gap-2.5 text-xs text-slate-700 font-medium">
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                                    <span>{rec}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Main Message Text (Markdown Formatted) */}
                      {msg.text && (
                        <div className="p-5 rounded-3xl bg-white border border-slate-200 text-slate-800 shadow-sm relative">
                          <div className="prose prose-sm prose-slate max-w-none prose-p:leading-relaxed prose-headings:font-bold prose-headings:text-slate-900 prose-ul:my-2 prose-li:my-0.5">
                            <Markdown>{msg.text}</Markdown>
                          </div>

                          {/* Message Footer / Controls */}
                          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 rounded-full bg-blue-600/10 flex items-center justify-center text-[9px] font-bold text-blue-600">
                                R
                              </div>
                              <span className="text-[10px] font-bold uppercase tracking-wider">RapidAid AI</span>
                              {msg.fromVoice && (
                                <span className="px-1.5 py-0.5 rounded bg-blue-50 text-blue-600 text-[10px] font-bold">
                                  Spoken in Call
                                </span>
                              )}
                              <span>•</span>
                              <span className="text-[10px]">
                                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                            </div>

                            <div className="flex items-center gap-1">
                              {/* Speak Aloud Button */}
                              <button
                                onClick={() => speakText(msg.text!, msg.id || `${idx}`)}
                                className={`p-1.5 rounded-lg transition-colors ${
                                  speakingMessageId === (msg.id || `${idx}`)
                                    ? 'bg-blue-50 text-blue-600 font-bold'
                                    : 'hover:bg-slate-100 text-slate-400 hover:text-slate-600'
                                }`}
                                title={speakingMessageId === (msg.id || `${idx}`) ? "Stop Speaking" : "Read Aloud"}
                              >
                                {speakingMessageId === (msg.id || `${idx}`) ? (
                                  <VolumeX className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                                ) : (
                                  <Volume2 className="w-3.5 h-3.5" />
                                )}
                              </button>

                              {/* Copy Button */}
                              <button
                                onClick={() => copyMessage(msg.text!, msg.id || `${idx}`)}
                                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                                title="Copy Advice"
                              >
                                {copiedId === (msg.id || `${idx}`) ? (
                                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                                ) : (
                                  <Copy className="w-3.5 h-3.5" />
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}

            {/* Live Generation Typing Pulse */}
            {isGenerating && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 p-4 rounded-2xl bg-white border border-slate-200 max-w-xs shadow-sm"
              >
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs font-bold text-slate-500">RapidAid is analyzing...</span>
                <button
                  onClick={handleStopGenerating}
                  className="ml-auto p-1 rounded-md hover:bg-slate-100 text-slate-400 hover:text-slate-600"
                  title="Stop Response"
                >
                  <Square className="w-3.5 h-3.5" />
                </button>
              </motion.div>
            )}

            {/* Error Banner */}
            {errorMessage && (
              <div className="p-4 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
                  <span className="font-semibold">{errorMessage}</span>
                </div>
                <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-600">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <div ref={messagesEndRef} className="h-4" />
          </div>
        </main>

        {/* Sticky Chat Input Box */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-[#F8FAFC] via-[#F8FAFC]/95 to-transparent pt-6 pb-5 px-4 sm:px-8 z-20">
          <div className="max-w-3xl mx-auto">
            <form
              onSubmit={(e) => handleSendText(e)}
              className="relative bg-white border border-slate-200/90 rounded-2xl shadow-lg shadow-slate-200/40 p-2 pl-4 flex items-center gap-2 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 transition-all"
            >
              <textarea
                ref={inputRef}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendText();
                  }
                }}
                rows={1}
                placeholder="Describe symptoms, medications, or questions for clinical triage..."
                className="flex-1 bg-transparent border-none outline-none text-sm font-medium text-slate-900 placeholder:text-slate-400 resize-none max-h-32 py-1.5"
              />

              {isGenerating ? (
                <button
                  type="button"
                  onClick={handleStopGenerating}
                  className="p-2.5 rounded-xl bg-slate-100 text-slate-600 hover:bg-slate-200 font-bold text-xs transition-colors flex items-center gap-1.5"
                >
                  <Square className="w-3.5 h-3.5" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!textInput.trim()}
                  className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs transition-colors disabled:opacity-40 disabled:pointer-events-none shadow-md shadow-blue-500/20 flex items-center justify-center shrink-0"
                >
                  <Send className="w-4 h-4" />
                </button>
              )}
            </form>
            <p className="text-[11px] text-slate-400 text-center mt-2 font-medium">
              Medical Artificial Intelligence • Independent text consultation with optional voice companion on the side
            </p>
          </div>
        </div>
      </div>

      {/* Voice Assistant Companion (On the Side) */}
      <VoiceCompanion
        isOpen={showVoiceCompanion}
        onClose={() => setShowVoiceCompanion(false)}
        onTranscription={addTranscriptionToRecord}
        user={user}
        patientProfile={userData?.healthProfile}
        medications={medications}
        onMedicationsUpdated={() => {
          if (user) {
            subscribeToMedications(user.uid, setMedications);
          }
        }}
      />

      {/* Modals */}
      <AuthModal isOpen={showAuthModal} onClose={() => setShowAuthModal(false)} />
      <ProfileModal isOpen={showProfileModal} onClose={() => setShowProfileModal(false)} />
      {showMedications && (
        <MedicationPanel 
          userId={user?.uid || 'guest'} 
          medications={medications} 
          onClose={() => setShowMedications(false)} 
        />
      )}

      {/* Medical Knowledge Library Modal */}
      <AnimatePresence>
        {showResources && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowResources(false)}
              className="absolute inset-0 bg-slate-900/30 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              className="relative w-full max-w-2xl bg-white border border-slate-200 rounded-3xl overflow-hidden shadow-2xl z-10"
            >
              <div className="p-6 border-b border-slate-100 flex items-center justify-between bg-slate-50/60">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600">
                    <BookOpen className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-slate-900">Verified Health Resources</h3>
                    <p className="text-xs text-slate-400 font-medium">Authoritative medical references and directories</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowResources(false)}
                  className="p-1.5 hover:bg-slate-200/50 rounded-xl transition-colors text-slate-400"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="p-6 max-h-[60vh] overflow-y-auto custom-scrollbar space-y-3">
                {MEDICAL_RESOURCES.map((res, i) => (
                  <a
                    key={i}
                    href={res.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-4 rounded-2xl bg-slate-50 hover:bg-blue-50/50 border border-slate-200/70 hover:border-blue-300 transition-all flex items-start justify-between group block"
                  >
                    <div>
                      <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">{res.category}</span>
                      <h4 className="text-sm font-bold text-slate-900 group-hover:text-blue-600 transition-colors mt-0.5">{res.name}</h4>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed font-medium">{res.description}</p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-slate-300 group-hover:text-blue-500 shrink-0 ml-3 mt-1" />
                  </a>
                ))}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
