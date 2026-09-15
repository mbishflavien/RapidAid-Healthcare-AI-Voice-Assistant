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
  Radio,
  FileText,
  ShieldCheck,
  MoreVertical
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
import { Transcription, SymptomAnalysis, Session, VitalSigns, AcuityLevel } from './types';
import { ClinicalPatientBanner } from './components/ClinicalPatientBanner';
import { SoapNoteModal } from './components/SoapNoteModal';
import { ClinicalDecisionSupportModal } from './components/ClinicalDecisionSupportModal';

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
    icon: "🩺",
    title: "Acute Symptom Triage",
    prompt: "Patient presenting with low-grade fever (100.8°F), diffuse myalgias, and dry cough for 48 hours. Please provide clinical triage assessment and red flags."
  },
  {
    icon: "💊",
    title: "Pharmacotherapy Review",
    prompt: "Review potential contraindications or drug-drug interactions between oral NSAIDs (ibuprofen) and ACE inhibitors (lisinopril)."
  },
  {
    icon: "🫀",
    title: "Cardiovascular vs. Non-Cardiac Triage",
    prompt: "Provide diagnostic differentiation guidelines between acute musculoskeletal/anxiety-induced precordial chest discomfort and acute coronary syndromes."
  },
  {
    icon: "🩹",
    title: "Wound & Burn Care Protocol",
    prompt: "Outline standard clinical first-line management protocol for a partial-thickness (second-degree) thermal steam burn on the forearm."
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

  // Voice Companion on the side - auto-open on desktop, closed by default on mobile/tablet
  const [showVoiceCompanion, setShowVoiceCompanion] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth >= 1024;
    }
    return false;
  });

  // Mobile overflow action sheet dropdown
  const [showMobileActions, setShowMobileActions] = useState(false);

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

  // Clinical Telemetry & Triage State
  const [vitals, setVitals] = useState<VitalSigns>({
    heartRate: 74,
    bloodPressureSystolic: 118,
    bloodPressureDiastolic: 78,
    oxygenSaturation: 99,
    temperature: 98.6,
    respiratoryRate: 16,
    painLevel: 2,
    lastRecorded: Date.now()
  });
  const [acuity, setAcuity] = useState<AcuityLevel>('ESI-3');
  const [showSoapModal, setShowSoapModal] = useState(false);
  const [showCdsModal, setShowCdsModal] = useState(false);

  // Helper to inject structured clinical note or score directly into chat composer
  const handleInjectTextToInput = (text: string) => {
    setTextInput(prev => prev ? `${prev}\n\n${text}` : text);
    inputRef.current?.focus();
  };

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

  // Handle window resize for responsive layout syncing
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setShowMobileActions(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

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
        vitals,
        acuity,
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
    <div className="flex h-screen bg-[#F8FAFC] text-slate-900 font-sans selection:bg-teal-500/20 overflow-hidden">
      {/* Mobile Backdrop for History Drawer */}
      {showHistory && (
        <div 
          className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-40 lg:hidden transition-opacity"
          onClick={() => setShowHistory(false)}
          aria-label="Close history drawer"
        />
      )}

      {/* Consultations / Medical Record History Drawer (Left) */}
      <motion.aside
        initial={false}
        animate={{ width: showHistory ? 320 : 0, opacity: showHistory ? 1 : 0 }}
        className={`fixed inset-y-0 left-0 z-50 lg:relative lg:z-30 shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-hidden shadow-2xl lg:shadow-none transition-all ${
          !showHistory ? 'pointer-events-none lg:pointer-events-auto' : ''
        }`}
      >
        <div className="w-80 max-w-[85vw] sm:max-w-none flex flex-col h-full">
          {/* Institutional Clinic Header */}
          <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/80">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-teal-700 flex items-center justify-center text-white shadow-xs">
                <Stethoscope className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-slate-900 tracking-tight block text-sm leading-tight">RapidAid</span>
                  <span className="px-1.5 py-0.5 rounded bg-teal-100 text-teal-800 font-mono text-[9px] font-bold">CLINICAL</span>
                </div>
                <span className="text-[10px] text-slate-500 font-medium">Electronic Health Records</span>
              </div>
            </div>
            <button
              onClick={() => setShowHistory(false)}
              className="p-1.5 hover:bg-slate-200/60 rounded-lg transition-colors text-slate-400 hover:text-slate-700"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          {/* New Encounter CTA */}
          <div className="p-3.5">
            <button
              onClick={startNewSession}
              className="w-full py-2.5 px-3.5 rounded-xl bg-teal-50 border border-teal-200 hover:bg-teal-100/80 hover:border-teal-300 transition-all flex items-center gap-2.5 text-xs font-semibold text-teal-900 group shadow-xs"
            >
              <div className="w-6 h-6 rounded-lg bg-teal-700 flex items-center justify-center text-white shadow-xs">
                <Plus className="w-3.5 h-3.5" />
              </div>
              <span>New Patient Encounter</span>
            </button>
          </div>

          {/* Past Consultations / Encounters List */}
          <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1 custom-scrollbar">
            <div className="px-2 py-1 flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400">
              <span>Patient Encounters</span>
              <span className="font-mono text-slate-400 font-normal">{sessions.length} recorded</span>
            </div>
            {user ? (
              sessions.length === 0 ? (
                <div className="p-6 text-center bg-slate-50 rounded-xl border border-dashed border-slate-200">
                  <FileText className="w-6 h-6 text-slate-300 mx-auto mb-2" />
                  <p className="text-xs font-medium text-slate-500">No patient encounters recorded yet</p>
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
                    className={`w-full p-3 rounded-xl text-left transition-all group flex flex-col gap-1 border ${
                      currentSessionId === session.id
                        ? 'bg-teal-50/80 border-teal-300 shadow-xs'
                        : 'bg-transparent border-transparent hover:bg-slate-100/80 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-xs font-bold truncate flex-1 ${currentSessionId === session.id ? 'text-teal-950' : 'text-slate-800'}`}>
                        {session.title}
                      </span>
                      <Trash2
                        onClick={(e) => deleteSession(session.id, e)}
                        className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 hover:text-red-600 transition-opacity ml-2 text-slate-400 shrink-0"
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 font-medium">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        <span>{new Date(session.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                      </div>
                      <span className="font-mono text-[9px] text-slate-400">ENC#{session.id.slice(0, 5).toUpperCase()}</span>
                    </div>
                  </button>
                ))
              )
            ) : (
              <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-200 text-amber-900 text-xs space-y-2">
                <span className="font-bold block">Patient Session Unregistered</span>
                <span className="text-[11px] text-amber-700 leading-relaxed block">
                  Sign in with clinician or patient credentials to persist health records securely.
                </span>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="w-full py-1.5 bg-amber-700 text-white rounded-lg font-semibold text-xs hover:bg-amber-800 transition-colors"
                >
                  Clinical Sign In
                </button>
              </div>
            )}
          </div>

          {/* User Account / Identity Bar */}
          <div className="p-3 border-t border-slate-200 bg-slate-50/60">
            {user ? (
              <div className="flex items-center gap-2.5 p-2 rounded-xl bg-white border border-slate-200 shadow-xs">
                <div className="w-8 h-8 rounded-lg bg-teal-100 text-teal-800 flex items-center justify-center font-bold text-xs">
                  {user.email ? user.email[0].toUpperCase() : 'P'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-bold text-slate-900 truncate">{user.displayName || user.email}</p>
                  <button
                    onClick={() => signOut(firebaseAuth)}
                    className="text-[10px] font-semibold text-slate-400 hover:text-red-600 uppercase tracking-wider transition-colors"
                  >
                    Disconnect Session
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="w-full py-2 rounded-xl bg-slate-900 text-white text-xs font-semibold hover:bg-slate-800 transition-colors"
              >
                Access Patient Portal
              </button>
            )}
          </div>
        </div>
      </motion.aside>

      {/* Main Medical Workspace (Center: Chat & Consultation) */}
      <div className="flex-1 flex flex-col min-w-0 relative h-full bg-[#F8FAFC]">
        {/* Clinical Workspace Header Bar */}
        <header className="h-14 sm:h-16 px-2.5 sm:px-6 lg:px-8 border-b border-slate-200 bg-white/95 backdrop-blur-xs flex items-center justify-between sticky top-0 z-20">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            {!showHistory && (
              <button
                onClick={() => setShowHistory(true)}
                className="p-1.5 sm:p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors shrink-0"
                title="Consultation History"
              >
                <History className="w-4 h-4" />
              </button>
            )}
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 sm:gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_6px_#10b981] shrink-0" />
                <h1 className="text-xs sm:text-sm font-bold text-slate-900 tracking-tight truncate max-w-[110px] xs:max-w-[160px] sm:max-w-xs md:max-w-md">
                  {user ? (activeSession?.title || 'Clinical Encounter') : 'RapidAid Consultation'}
                </h1>
                <span className="hidden md:inline-flex items-center px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-mono text-[9px] font-semibold border border-slate-200 shrink-0">
                  ENC-ACTIVE
                </span>
              </div>
              <p className="text-[10px] text-slate-500 font-medium hidden sm:block truncate">
                Clinical Diagnostic Protocol • Evidence-Based Triage Matrix
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1 sm:gap-2 shrink-0">
            {/* Clinical SOAP Chart Button */}
            <button
              onClick={() => setShowSoapModal(true)}
              className="px-2 sm:px-2.5 py-1.5 rounded-lg bg-teal-50 hover:bg-teal-100 border border-teal-200 text-xs font-semibold text-teal-900 transition-colors flex items-center gap-1.5 shadow-xs shrink-0"
              title="Open Clinical Encounter Progress Note (SOAP)"
            >
              <FileText className="w-3.5 h-3.5 text-teal-800" />
              <span className="hidden sm:inline">SOAP Note</span>
            </button>

            {/* Clinical Decision Support (CDS) Tools */}
            <button
              onClick={() => setShowCdsModal(true)}
              className="px-2 sm:px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-semibold text-slate-700 transition-colors flex items-center gap-1.5 shadow-xs shrink-0"
              title="Clinical Decision Support (qSOFA, Red Flags, GCS)"
            >
              <Stethoscope className="w-3.5 h-3.5 text-teal-700" />
              <span className="hidden md:inline">CDS Tools</span>
            </button>

            {/* Quick Action Buttons */}
            <button
              onClick={() => setShowMedications(true)}
              className="px-2 sm:px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-xs font-semibold text-slate-700 transition-colors flex items-center gap-1.5 shadow-xs shrink-0"
              title="eMAR Active Pharmacotherapy"
            >
              <Pill className="w-3.5 h-3.5 text-teal-700" />
              <span className="hidden sm:inline">eMAR</span>
              {medications.length > 0 && (
                <span className="px-1.5 py-0.2 bg-teal-100 text-teal-800 rounded-full text-[10px] font-mono font-bold">
                  {medications.length}
                </span>
              )}
            </button>

            {/* Desktop Direct Action Buttons */}
            <div className="hidden lg:flex items-center gap-1.5">
              <button
                onClick={() => setShowResources(true)}
                className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors shadow-xs"
                title="Medical Reference Library"
              >
                <BookOpen className="w-4 h-4" />
              </button>

              {user && (
                <button
                  onClick={() => setShowProfileModal(true)}
                  className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors shadow-xs"
                  title="Patient Chart & Health Profile"
                >
                  <UserIcon className="w-4 h-4" />
                </button>
              )}

              {currentMessages.length > 0 && (
                <>
                  <button
                    onClick={downloadTranscript}
                    className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors shadow-xs"
                    title="Export Clinical Encounter Record"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={clearCurrentChat}
                    className="p-2 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-red-600 transition-colors shadow-xs"
                    title="Clear Encounter Messages"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </>
              )}
            </div>

            {/* Mobile/Tablet Secondary Actions Dropdown */}
            <div className="relative lg:hidden">
              <button
                onClick={() => setShowMobileActions(!showMobileActions)}
                className="p-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600 transition-colors shadow-xs"
                title="More clinical actions"
                aria-label="More actions"
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {showMobileActions && (
                <>
                  <div 
                    className="fixed inset-0 z-30" 
                    onClick={() => setShowMobileActions(false)} 
                  />
                  <div className="absolute right-0 mt-2 w-52 bg-white border border-slate-200 rounded-xl shadow-xl z-40 py-1 text-xs">
                    {user && (
                      <button
                        onClick={() => {
                          setShowMobileActions(false);
                          setShowProfileModal(true);
                        }}
                        className="w-full px-3.5 py-2.5 text-left flex items-center gap-2.5 hover:bg-slate-50 text-slate-700"
                      >
                        <UserIcon className="w-4 h-4 text-teal-700" />
                        <span>Patient Health Profile</span>
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setShowMobileActions(false);
                        setShowResources(true);
                      }}
                      className="w-full px-3.5 py-2.5 text-left flex items-center gap-2.5 hover:bg-slate-50 text-slate-700"
                    >
                      <BookOpen className="w-4 h-4 text-teal-700" />
                      <span>Medical Library</span>
                    </button>
                    {currentMessages.length > 0 && (
                      <>
                        <button
                          onClick={() => {
                            setShowMobileActions(false);
                            downloadTranscript();
                          }}
                          className="w-full px-3.5 py-2.5 text-left flex items-center gap-2.5 hover:bg-slate-50 text-slate-700"
                        >
                          <Download className="w-4 h-4 text-teal-700" />
                          <span>Export Encounter</span>
                        </button>
                        <button
                          onClick={() => {
                            setShowMobileActions(false);
                            clearCurrentChat();
                          }}
                          className="w-full px-3.5 py-2.5 text-left flex items-center gap-2.5 hover:bg-red-50 text-red-700"
                        >
                          <Trash2 className="w-4 h-4 text-red-600" />
                          <span>Clear Chat Messages</span>
                        </button>
                      </>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* Telehealth Audio Station Toggle */}
            <button
              onClick={() => setShowVoiceCompanion(prev => !prev)}
              className={`px-2.5 sm:px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all flex items-center gap-1.5 shadow-xs shrink-0 ${
                showVoiceCompanion
                  ? 'bg-teal-700 border-teal-700 text-white'
                  : 'bg-white border-slate-200 text-slate-800 hover:bg-slate-50'
              }`}
              title={showVoiceCompanion ? "Hide Telehealth Audio Panel" : "Open Telehealth Audio Exam Station"}
            >
              <Radio className={`w-3.5 h-3.5 ${showVoiceCompanion ? 'animate-pulse' : ''}`} />
              <span className="hidden sm:inline">Audio Exam</span>
              <span className={`w-1.5 h-1.5 rounded-full ${showVoiceCompanion ? 'bg-emerald-300' : 'bg-slate-400'}`} />
            </button>
          </div>
        </header>

        {/* Clinical Telemetry & Triage Banner */}
        <ClinicalPatientBanner
          vitals={vitals}
          onUpdateVitals={setVitals}
          acuity={acuity}
          onUpdateAcuity={setAcuity}
          patientProfile={userData?.healthProfile}
          user={user}
          onOpenProfile={() => setShowProfileModal(true)}
          onInjectVitals={handleInjectTextToInput}
        />

        {/* Main Chat Stream Area */}
        <main className="flex-1 overflow-y-auto custom-scrollbar px-3 sm:px-6 lg:px-8 py-4 sm:py-5 space-y-4 sm:space-y-5">
          <div className="max-w-3xl lg:max-w-4xl xl:max-w-5xl mx-auto space-y-4 sm:space-y-5 pb-32 sm:pb-36">
            {/* Clinical Safety Protocol Notice */}
            <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-xs flex items-start gap-3">
              <div className="w-7 h-7 rounded-lg bg-amber-50 border border-amber-200 flex items-center justify-center shrink-0 mt-0.5">
                <Info className="w-4 h-4 text-amber-700" />
              </div>
              <div className="text-xs">
                <span className="font-bold text-slate-900 block">Clinical Triage Protocol & Decision Support</span>
                <span className="text-slate-600 leading-relaxed font-medium text-[11px]">
                  RapidAid applies institutional clinical assessment algorithms. For life-threatening emergencies (acute anaphylaxis, severe cardiac chest pain, stroke symptoms), invoke immediate emergency dispatch (911).
                </span>
              </div>
            </div>

            {/* Active Medication Scheduled Dose Alert */}
            {activeReminders.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3.5 rounded-xl bg-teal-800 text-white shadow-sm flex flex-wrap sm:flex-nowrap items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center shrink-0">
                    <Pill className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-wider font-mono font-bold text-teal-200">Scheduled Medication Administration Due</p>
                    <p className="text-xs font-semibold">Active Prescription: {activeReminders.join(', ')}</p>
                  </div>
                </div>
                <button
                  onClick={() => setActiveReminders([])}
                  className="px-3 py-1 rounded-lg bg-white text-teal-900 font-semibold text-xs hover:bg-teal-50 transition-colors shadow-xs shrink-0"
                >
                  Confirm Dose
                </button>
              </motion.div>
            )}

            {/* Empty State / Suggested Medical Intake Scenarios */}
            {currentMessages.length === 0 && (
              <div className="py-6 text-center space-y-6">
                <div className="relative inline-block">
                  <div className="w-14 h-14 rounded-2xl bg-teal-50 border border-teal-200 flex items-center justify-center mx-auto shadow-xs">
                    <Stethoscope className="w-7 h-7 text-teal-800" />
                  </div>
                </div>
                <div className="space-y-1.5 max-w-lg mx-auto">
                  <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-teal-50 border border-teal-200 text-teal-800 text-[10px] font-mono font-bold uppercase tracking-wider">
                    <span>Dept. of Emergency & Ambulatory Triage</span>
                    <span>•</span>
                    <span>Station 04</span>
                  </div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">Clinical Decision Support Station</h2>
                  <p className="text-xs text-slate-500 font-medium leading-relaxed">
                    Submit patient symptoms, vital signs, or pharmacology questions to initiate an evidence-based clinical evaluation, or select a rapid clinical tool below.
                  </p>
                </div>

                {/* Quick Bedside Actions */}
                <div className="flex flex-wrap items-center justify-center gap-2 max-w-xl mx-auto">
                  <button
                    onClick={() => setShowCdsModal(true)}
                    className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition-colors"
                  >
                    <Activity className="w-3.5 h-3.5 text-teal-700" />
                    <span>qSOFA Sepsis Screen</span>
                  </button>
                  <button
                    onClick={() => setShowCdsModal(true)}
                    className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition-colors"
                  >
                    <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
                    <span>Red Flag Checklist</span>
                  </button>
                  <button
                    onClick={() => setShowSoapModal(true)}
                    className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition-colors"
                  >
                    <FileText className="w-3.5 h-3.5 text-teal-700" />
                    <span>Open Blank SOAP Note</span>
                  </button>
                  <button
                    onClick={() => setShowMedications(true)}
                    className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition-colors"
                  >
                    <Pill className="w-3.5 h-3.5 text-teal-700" />
                    <span>eMAR Medications</span>
                  </button>
                </div>

                {/* Suggested Intake Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left max-w-2xl mx-auto pt-1">
                  {SUGGESTED_PROMPTS.map((item, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendText(undefined, item.prompt)}
                      className="p-4 rounded-xl bg-white border border-slate-200 hover:border-teal-500 hover:shadow-xs transition-all group flex flex-col justify-between text-left"
                    >
                      <div>
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-base">{item.icon}</span>
                          <span className="text-xs font-bold text-slate-900 group-hover:text-teal-700 transition-colors">
                            {item.title}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500 leading-relaxed font-medium line-clamp-2">
                          "{item.prompt}"
                        </p>
                      </div>
                      <span className="text-[10px] font-bold text-teal-700 uppercase tracking-wider mt-3 flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                        Initiate Assessment &rarr;
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
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`relative ${msg.analysis ? 'w-full' : 'max-w-[92%] sm:max-w-[85%] md:max-w-[80%]'}`}>
                  {msg.isUser ? (
                    // Patient Message Bubble
                    <div className="p-3.5 sm:p-4 rounded-2xl bg-slate-900 text-white shadow-xs">
                      <div className="flex items-center gap-2 mb-1 text-[10px] text-teal-400 font-mono font-bold uppercase tracking-wider">
                        <span>Patient Intake</span>
                        {msg.fromVoice && (
                          <span className="px-1.5 py-0.2 rounded bg-teal-900/60 text-teal-300 font-semibold">
                            Telehealth Audio
                          </span>
                        )}
                      </div>
                      <p className="text-xs leading-relaxed font-medium whitespace-pre-wrap text-slate-100">{msg.text}</p>
                      <div className="mt-2 flex items-center justify-end text-[10px] font-mono text-slate-400">
                        <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                    </div>
                  ) : (
                    // Assistant Clinical Response Card
                    <div className="space-y-3 w-full">
                      {/* Structured Clinical Triage Assessment Card if generated */}
                      {msg.analysis && (
                        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-xs mb-3">
                          {/* Card Header with Clinical Urgency Pill */}
                          <div className={`px-4 sm:px-5 py-3 sm:py-3.5 flex flex-wrap items-center justify-between gap-2 border-b ${
                            msg.analysis.urgency === 'Emergency' ? 'bg-red-50/90 border-red-200 text-red-800' :
                            msg.analysis.urgency === 'High' ? 'bg-amber-50/90 border-amber-200 text-amber-800' :
                            'bg-teal-50/90 border-teal-200 text-teal-900'
                          }`}>
                            <div className="flex items-center gap-2.5">
                              <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                                msg.analysis.urgency === 'Emergency' ? 'bg-red-100 text-red-700' :
                                msg.analysis.urgency === 'High' ? 'bg-amber-100 text-amber-700' :
                                'bg-teal-100 text-teal-800'
                              }`}>
                                <ShieldAlert className="w-4 h-4" />
                              </div>
                              <div>
                                <div className="flex items-center gap-2">
                                  <h4 className="text-xs font-bold uppercase tracking-wider">Clinical Triage Assessment</h4>
                                  <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-white/60 border border-current/20">ICD-TRIAGE</span>
                                </div>
                                <p className="text-[11px] font-medium opacity-85">
                                  Reported: {msg.analysis.symptoms.join(', ')}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
                              <button
                                onClick={() => setShowSoapModal(true)}
                                className="px-2 py-1 rounded bg-white/90 hover:bg-white text-slate-800 text-[10px] font-mono font-bold flex items-center gap-1 border border-current/20 transition-colors shadow-xs"
                                title="Open full Encounter SOAP Progress Note"
                              >
                                <FileText className="w-3 h-3 text-teal-800" />
                                <span>SOAP Note</span>
                              </button>
                              <span className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-bold uppercase tracking-wider ${
                                msg.analysis.urgency === 'Emergency' ? 'bg-red-700 text-white' :
                                msg.analysis.urgency === 'High' ? 'bg-amber-700 text-white' :
                                'bg-teal-800 text-white'
                              }`}>
                                {msg.analysis.urgency} Urgency
                              </span>
                            </div>
                          </div>

                          {/* Assessment Content */}
                          <div className="p-3.5 sm:p-5 space-y-4 sm:space-y-5">
                            {/* Differential Diagnoses / Potential Causes */}
                            <div>
                              <div className="flex items-center justify-between mb-2.5">
                                <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                                  Differential Diagnoses / Etiology
                                </p>
                                <span className="text-[9px] font-mono text-slate-400">CLINICAL LIKELIHOOD</span>
                              </div>
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                                {msg.analysis.potentialConditions.map((cond, cIdx) => (
                                  <div key={cIdx} className="p-3 rounded-xl bg-slate-50/80 border border-slate-200">
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="text-xs font-bold text-slate-900">{cond.name}</span>
                                      <span className="text-[9px] font-mono font-bold text-teal-800 bg-teal-50 border border-teal-200 px-1.5 py-0.5 rounded">
                                        {cond.likelihood}
                                      </span>
                                    </div>
                                    <p className="text-[11px] text-slate-600 font-medium leading-relaxed">{cond.description}</p>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Recommended Clinical Roadmap */}
                            <div>
                              <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-2.5">
                                Recommended Clinical Action Protocol
                              </p>
                              <div className="space-y-1.5">
                                {msg.analysis.recommendations.map((rec, rIdx) => (
                                  <div key={rIdx} className="flex items-start gap-2.5 text-xs text-slate-700 font-medium p-2 rounded-lg bg-slate-50/50 border border-slate-100">
                                    <CheckCircle2 className="w-3.5 h-3.5 text-teal-700 shrink-0 mt-0.5" />
                                    <span className="leading-relaxed">{rec}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Main Clinical Note (Markdown Formatted) */}
                      {msg.text && (
                        <div className="p-3.5 sm:p-5 rounded-2xl bg-white border border-slate-200 text-slate-800 shadow-xs relative">
                          <div className="prose prose-sm prose-slate max-w-none prose-p:leading-relaxed prose-headings:font-bold prose-headings:text-slate-900 prose-ul:my-2 prose-li:my-0.5 text-xs sm:text-sm">
                            <Markdown>{msg.text}</Markdown>
                          </div>

                          {/* Message Footer / Clinical Telemetry */}
                          <div className="mt-3 sm:mt-4 pt-2.5 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 rounded bg-teal-700 text-white flex items-center justify-center text-[9px] font-bold">
                                +
                              </div>
                              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600">RapidAid Clinical AI</span>
                              {msg.fromVoice && (
                                <span className="px-1.5 py-0.2 rounded bg-teal-50 border border-teal-200 text-teal-800 text-[9px] font-mono font-bold">
                                  VOICE ENCOUNTER
                                </span>
                              )}
                              <span>•</span>
                              <span className="text-[10px] font-mono">
                                {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                            </div>

                            <div className="flex items-center gap-1">
                              {/* Audio Read Aloud */}
                              <button
                                onClick={() => speakText(msg.text!, msg.id || `${idx}`)}
                                className={`p-1.5 rounded-lg transition-colors ${
                                  speakingMessageId === (msg.id || `${idx}`)
                                    ? 'bg-teal-50 text-teal-700 font-bold'
                                    : 'hover:bg-slate-100 text-slate-400 hover:text-slate-600'
                                }`}
                                title={speakingMessageId === (msg.id || `${idx}`) ? "Stop Audio Readout" : "Audio Readout"}
                              >
                                {speakingMessageId === (msg.id || `${idx}`) ? (
                                  <VolumeX className="w-3.5 h-3.5 text-teal-700 animate-pulse" />
                                ) : (
                                  <Volume2 className="w-3.5 h-3.5" />
                                )}
                              </button>

                              {/* Copy Clinical Advice */}
                              <button
                                onClick={() => copyMessage(msg.text!, msg.id || `${idx}`)}
                                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                                title="Copy Clinical Text"
                              >
                                {copiedId === (msg.id || `${idx}`) ? (
                                  <Check className="w-3.5 h-3.5 text-teal-700" />
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
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 p-3.5 rounded-xl bg-white border border-slate-200 max-w-xs shadow-xs"
              >
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs font-semibold text-slate-600 font-mono text-[11px]">Synthesizing clinical evaluation...</span>
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
              <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
                  <span className="font-semibold">{errorMessage}</span>
                </div>
                <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-700">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <div ref={messagesEndRef} className="h-4" />
          </div>
        </main>

        {/* Sticky Clinical Order & Intake Input Box */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-[#F8FAFC] via-[#F8FAFC]/95 to-transparent pt-3 sm:pt-4 pb-3 sm:pb-4 px-2.5 sm:px-6 lg:px-8 z-20">
          <div className="max-w-3xl lg:max-w-4xl xl:max-w-5xl mx-auto">
            <form
              onSubmit={(e) => handleSendText(e)}
              className="relative bg-white border border-slate-300 rounded-xl shadow-xs p-2 pl-3 sm:pl-3.5 flex items-center gap-2 focus-within:border-teal-600 focus-within:ring-2 focus-within:ring-teal-100 transition-all"
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
                placeholder="Enter patient symptoms, vitals, or pharmacology inquiry..."
                className="flex-1 bg-transparent border-none outline-none text-xs sm:text-sm font-medium text-slate-900 placeholder:text-slate-400 resize-none max-h-32 py-1"
              />

              {isGenerating ? (
                <button
                  type="button"
                  onClick={handleStopGenerating}
                  className="px-2.5 sm:px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-semibold text-xs transition-colors flex items-center gap-1.5 shrink-0"
                >
                  <Square className="w-3.5 h-3.5" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!textInput.trim()}
                  className="px-2.5 sm:px-3 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white font-semibold text-xs transition-colors disabled:opacity-40 disabled:pointer-events-none shadow-xs flex items-center justify-center shrink-0 gap-1.5"
                >
                  <span className="hidden sm:inline">Evaluate</span>
                  <Send className="w-3.5 h-3.5" />
                </button>
              )}
            </form>
            <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1.5 px-1 font-mono">
              <span className="truncate">RapidAid Clinical Intelligence</span>
              <span className="hidden sm:inline">CONFIDENTIAL • HIPAA PROTOCOL</span>
              <span className="sm:hidden">CONFIDENTIAL</span>
            </div>
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
              className="absolute inset-0 bg-slate-950/40 backdrop-blur-xs"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 10 }}
              className="relative w-full max-w-2xl bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-2xl z-10"
            >
              <div className="p-4 sm:p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50/80">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-teal-100 border border-teal-200 flex items-center justify-center text-teal-800">
                    <BookOpen className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-bold text-slate-900">Clinical Knowledge & Reference Library</h3>
                      <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-teal-100 text-teal-800 font-bold">VERIFIED</span>
                    </div>
                    <p className="text-[11px] text-slate-500 font-medium">Authoritative biomedical guidelines and evidence repositories</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowResources(false)}
                  className="p-1.5 hover:bg-slate-200/60 rounded-lg transition-colors text-slate-400 hover:text-slate-700"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="p-4 sm:p-5 max-h-[60vh] overflow-y-auto custom-scrollbar space-y-2.5">
                {MEDICAL_RESOURCES.map((res, i) => (
                  <a
                    key={i}
                    href={res.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-3.5 rounded-xl bg-slate-50/80 hover:bg-teal-50/60 border border-slate-200 hover:border-teal-300 transition-all flex items-start justify-between group block shadow-xs"
                  >
                    <div>
                      <span className="text-[9px] font-mono font-bold text-teal-800 bg-white border border-slate-200 px-1.5 py-0.5 rounded uppercase tracking-wider">{res.category}</span>
                      <h4 className="text-xs font-bold text-slate-900 group-hover:text-teal-900 transition-colors mt-1.5">{res.name}</h4>
                      <p className="text-[11px] text-slate-600 mt-1 leading-relaxed font-medium">{res.description}</p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-teal-700 shrink-0 ml-3 mt-1" />
                  </a>
                ))}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Clinical Encounter Progress Note (SOAP) Modal */}
      <SoapNoteModal
        isOpen={showSoapModal}
        onClose={() => setShowSoapModal(false)}
        vitals={vitals}
        acuity={acuity}
        patientProfile={userData?.healthProfile}
        medications={medications}
        messages={messages}
        user={user}
      />

      {/* Clinical Decision Support & Scoring Modal (qSOFA, Red Flags, GCS) */}
      <ClinicalDecisionSupportModal
        isOpen={showCdsModal}
        onClose={() => setShowCdsModal(false)}
        vitals={vitals}
        onInjectNote={handleInjectTextToInput}
      />
    </div>
  );
}
