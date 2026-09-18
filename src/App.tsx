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
  MoreVertical,
  Sun,
  Moon,
  SlidersHorizontal,
  ChevronDown
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';
import { useAuth } from './context/AuthContext';
import { useTheme } from './context/ThemeContext';
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
import { ChatMessageItem } from './components/ChatMessageItem';

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
  const { isDark, toggleTheme } = useTheme();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Sessions and conversation history
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Transcription[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // Voice Companion on the side - closed by default to keep workspace minimal
  const [showVoiceCompanion, setShowVoiceCompanion] = useState(false);

  // Unified Tools dropdown menu
  const [showToolsMenu, setShowToolsMenu] = useState(false);

  // Patient telemetry banner - hidden by default for minimal distraction-free triage
  const [showPatientBanner, setShowPatientBanner] = useState(false);

  // Dismissible safety notice - subtle
  const [showSafetyNotice, setShowSafetyNotice] = useState(false);

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

  // Scroll to bottom only when switching to a consultation session from history
  useEffect(() => {
    if (!currentSessionId || isGeneratingRef.current) return;
    const timeout = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
    }, 150);
    return () => clearTimeout(timeout);
  }, [currentSessionId]);

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
      setShowToolsMenu(false);
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
    setTimeout(() => {
      const el = document.getElementById(`msg-${transcription.id}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 40);

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

    // Position view once at the user query and start of the incoming response
    setTimeout(() => {
      const userEl = document.getElementById(`msg-${userMessage.id}`);
      if (userEl) {
        userEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 40);

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
      let renderRaf: number | null = null;
      let lastRenderTime = 0;

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
          const now = performance.now();
          // First token renders instantaneously; subsequent chunks throttled to ~30ms for smooth 60fps streaming
          if (now - lastRenderTime > 30 || !lastRenderTime) {
            lastRenderTime = now;
            if (renderRaf) {
              cancelAnimationFrame(renderRaf);
              renderRaf = null;
            }
            setMessages(prev => prev.map(m => 
              m.id === assistantMessageId ? { ...m, text: chunkText } : m
            ));
          } else if (!renderRaf) {
            renderRaf = requestAnimationFrame(() => {
              renderRaf = null;
              lastRenderTime = performance.now();
              setMessages(prev => prev.map(m => 
                m.id === assistantMessageId ? { ...m, text: currentAccumulated } : m
              ));
            });
          }
        },
        onAnalysis: (analysis) => {
          capturedAnalysis = analysis;
        }
      });

      if (renderRaf) {
        cancelAnimationFrame(renderRaf);
      }

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
    <div className="flex h-screen bg-[#F8FAFC] dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans selection:bg-teal-500/20 overflow-hidden transition-colors">
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
        className={`fixed inset-y-0 left-0 z-50 lg:relative lg:z-30 shrink-0 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col h-full overflow-hidden shadow-2xl lg:shadow-none transition-all ${
          !showHistory ? 'pointer-events-none lg:pointer-events-auto' : ''
        }`}
      >
        <div className="w-80 max-w-[85vw] sm:max-w-none flex flex-col h-full">
          {/* Institutional Clinic Header */}
          <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50/80 dark:bg-slate-950/80">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-teal-700 text-white flex items-center justify-center shadow-xs shrink-0">
                <Stethoscope className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-bold text-slate-900 dark:text-slate-100 tracking-tight block text-sm leading-tight">RapidAid</span>
                  <span className="px-1.5 py-0.5 rounded bg-teal-100 dark:bg-teal-950/80 text-teal-800 dark:text-teal-300 border dark:border-teal-800 font-mono text-[9px] font-bold">CLINICAL</span>
                </div>
                <span className="text-[10px] text-slate-500 dark:text-slate-400 font-medium">Electronic Health Records</span>
              </div>
            </div>
            <button
              onClick={() => setShowHistory(false)}
              className="p-1.5 hover:bg-slate-200/60 dark:hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          {/* New Encounter CTA */}
          <div className="p-3.5">
            <button
              onClick={startNewSession}
              className="w-full py-2.5 px-3.5 rounded-xl bg-teal-50 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800 hover:bg-teal-100/80 dark:hover:bg-teal-900/50 hover:border-teal-300 dark:hover:border-teal-700 transition-all flex items-center gap-2.5 text-xs font-semibold text-teal-900 dark:text-teal-200 group shadow-xs"
            >
              <div className="w-6 h-6 rounded-lg bg-teal-700 flex items-center justify-center text-white shadow-xs">
                <Plus className="w-3.5 h-3.5" />
              </div>
              <span>New Patient Encounter</span>
            </button>
          </div>

          {/* Past Consultations / Encounters List */}
          <div className="flex-1 overflow-y-auto px-3 pb-4 space-y-1 custom-scrollbar">
            <div className="px-2 py-1 flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
              <span>Patient Encounters</span>
              <span className="font-mono text-slate-400 dark:text-slate-500 font-normal">{sessions.length} recorded</span>
            </div>
            {user ? (
              sessions.length === 0 ? (
                <div className="p-6 text-center bg-slate-50 dark:bg-slate-800/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-800">
                  <FileText className="w-6 h-6 text-slate-300 dark:text-slate-600 mx-auto mb-2" />
                  <p className="text-xs font-medium text-slate-500 dark:text-slate-400">No patient encounters recorded yet</p>
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
                        ? 'bg-teal-50/80 dark:bg-teal-950/70 border-teal-300 dark:border-teal-700 shadow-xs'
                        : 'bg-transparent border-transparent hover:bg-slate-100/80 dark:hover:bg-slate-800/80 text-slate-700 dark:text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-xs font-bold truncate flex-1 ${currentSessionId === session.id ? 'text-teal-950 dark:text-teal-200' : 'text-slate-800 dark:text-slate-200'}`}>
                        {session.title}
                      </span>
                      <Trash2
                        onClick={(e) => deleteSession(session.id, e)}
                        className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 hover:text-red-600 dark:hover:text-red-400 transition-opacity ml-2 text-slate-400 dark:text-slate-500 shrink-0"
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 dark:text-slate-500 font-medium">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-3 h-3" />
                        <span>{new Date(session.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>
                      </div>
                      <span className="font-mono text-[9px] text-slate-400 dark:text-slate-500">ENC#{session.id.slice(0, 5).toUpperCase()}</span>
                    </div>
                  </button>
                ))
              )
            ) : (
              <div className="p-3.5 rounded-xl bg-amber-50/80 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800 text-amber-900 dark:text-amber-200 text-xs space-y-2">
                <span className="font-bold block">Patient Session Unregistered</span>
                <span className="text-[11px] text-amber-700 dark:text-amber-300 leading-relaxed block">
                  Sign in with clinician or patient credentials to persist health records securely.
                </span>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="w-full py-1.5 bg-amber-700 hover:bg-amber-800 text-white rounded-lg font-semibold text-xs transition-colors"
                >
                  Clinical Sign In
                </button>
              </div>
            )}
          </div>

          {/* User Account / Identity Bar */}
          <div className="p-3 border-t border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-950/60">
            {user ? (
              <div className="flex items-center gap-2.5 p-2 rounded-xl bg-white dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700 shadow-xs">
                <div className="w-8 h-8 rounded-lg bg-teal-100 dark:bg-teal-950 text-teal-800 dark:text-teal-300 border dark:border-teal-800 flex items-center justify-center font-bold text-xs shrink-0">
                  {user.email ? user.email[0].toUpperCase() : 'P'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-bold text-slate-900 dark:text-slate-100 truncate">{user.displayName || user.email}</p>
                  <button
                    onClick={() => signOut(firebaseAuth)}
                    className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400 uppercase tracking-wider transition-colors"
                  >
                    Disconnect Session
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setShowAuthModal(true)}
                className="w-full py-2 rounded-xl bg-slate-900 dark:bg-teal-700 text-white text-xs font-semibold hover:bg-slate-800 dark:hover:bg-teal-800 transition-colors"
              >
                Access Patient Portal
              </button>
            )}
          </div>
        </div>
      </motion.aside>

      {/* Main Medical Workspace (Center: Chat & Consultation) */}
      <div className="flex-1 flex flex-col min-w-0 relative h-full bg-[#F8FAFC] dark:bg-slate-950 transition-colors">
        {/* Minimal Clinical Workspace Header */}
        <header className="h-14 px-3 sm:px-6 border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-900/95 backdrop-blur-xs flex items-center justify-between sticky top-0 z-20 transition-colors">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            {!showHistory && (
              <button
                onClick={() => setShowHistory(true)}
                className="p-1.5 min-h-[36px] min-w-[36px] flex items-center justify-center rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 transition-colors shrink-0"
                title="Open Encounters History"
              >
                <History className="w-4 h-4" />
              </button>
            )}
            <div className="flex items-center gap-2 min-w-0">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0 shadow-[0_0_6px_#10b981]" />
              <h1 className="text-sm font-bold text-slate-900 dark:text-slate-100 tracking-tight truncate">
                {user ? (activeSession?.title || 'Clinical Encounter') : 'RapidAid'}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-1 sm:gap-2 shrink-0">
            {/* Clinical SOAP Chart Button */}
            <button
              onClick={() => setShowSoapModal(true)}
              className="px-2 sm:px-2.5 py-1.5 min-h-[36px] rounded-lg bg-teal-50 dark:bg-teal-950/50 hover:bg-teal-100 dark:hover:bg-teal-900/50 border border-teal-200 dark:border-teal-800 text-xs font-semibold text-teal-900 dark:text-teal-200 transition-colors flex items-center gap-1.5 shadow-xs"
              title="Open Clinical Encounter Progress Note (SOAP)"
            >
              <FileText className="w-3.5 h-3.5 text-teal-800 dark:text-teal-300" />
              <span className="hidden xs:inline">SOAP</span>
            </button>

            {/* Audio Exam Station Toggle */}
            <button
              onClick={() => setShowVoiceCompanion(prev => !prev)}
              className={`px-2 sm:px-2.5 py-1.5 min-h-[36px] rounded-lg border text-xs font-semibold transition-all flex items-center gap-1.5 shadow-xs ${
                showVoiceCompanion
                  ? 'bg-teal-700 border-teal-700 text-white'
                  : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
              title={showVoiceCompanion ? "Close Audio Exam Station" : "Open Audio Exam Station"}
            >
              <Radio className={`w-3.5 h-3.5 ${showVoiceCompanion ? 'animate-pulse' : ''}`} />
              <span className="hidden sm:inline">Audio</span>
            </button>

            {/* Quick Dark/Light Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-1.5 min-h-[36px] min-w-[36px] flex items-center justify-center rounded-lg bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 transition-colors shadow-xs"
              title={isDark ? "Switch to Light Theme" : "Switch to Dark Theme"}
            >
              {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
            </button>

            {/* Unified Tools Dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowToolsMenu(!showToolsMenu)}
                className={`p-1.5 min-h-[36px] min-w-[36px] flex items-center justify-center rounded-lg border text-xs font-semibold transition-colors shadow-xs ${
                  showToolsMenu
                    ? 'bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100'
                    : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
                }`}
                title="Clinical Tools & Options"
              >
                <MoreVertical className="w-4 h-4" />
              </button>

              {showToolsMenu && (
                <>
                  <div
                    className="fixed inset-0 z-30"
                    onClick={() => setShowToolsMenu(false)}
                  />
                  <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xl z-40 py-1.5 text-xs">
                    {/* Patient Vitals Bar Toggle */}
                    <button
                      onClick={() => {
                        setShowToolsMenu(false);
                        setShowPatientBanner(!showPatientBanner);
                      }}
                      className="w-full px-3.5 py-2 text-left flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                    >
                      <div className="flex items-center gap-2.5">
                        <Activity className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                        <span>Patient Vitals Bar</span>
                      </div>
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                        showPatientBanner
                          ? 'bg-teal-100 dark:bg-teal-950 text-teal-800 dark:text-teal-300'
                          : 'bg-slate-100 dark:bg-slate-800 text-slate-500'
                      }`}>
                        {showPatientBanner ? 'ON' : 'OFF'}
                      </span>
                    </button>

                    {/* CDS Tools */}
                    <button
                      onClick={() => {
                        setShowToolsMenu(false);
                        setShowCdsModal(true);
                      }}
                      className="w-full px-3.5 py-2 text-left flex items-center gap-2.5 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                    >
                      <Stethoscope className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                      <span>Decision Support (CDS)</span>
                    </button>

                    {/* eMAR Medications */}
                    <button
                      onClick={() => {
                        setShowToolsMenu(false);
                        setShowMedications(true);
                      }}
                      className="w-full px-3.5 py-2 text-left flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                    >
                      <div className="flex items-center gap-2.5">
                        <Pill className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                        <span>eMAR Medications</span>
                      </div>
                      {medications.length > 0 && (
                        <span className="px-1.5 py-0.2 bg-teal-100 dark:bg-teal-950 text-teal-800 dark:text-teal-300 rounded-full text-[10px] font-mono font-bold">
                          {medications.length}
                        </span>
                      )}
                    </button>

                    {/* Medical Library */}
                    <button
                      onClick={() => {
                        setShowToolsMenu(false);
                        setShowResources(true);
                      }}
                      className="w-full px-3.5 py-2 text-left flex items-center gap-2.5 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                    >
                      <BookOpen className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                      <span>Medical Library</span>
                    </button>

                    {/* Patient Profile */}
                    {user && (
                      <button
                        onClick={() => {
                          setShowToolsMenu(false);
                          setShowProfileModal(true);
                        }}
                        className="w-full px-3.5 py-2 text-left flex items-center gap-2.5 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                      >
                        <UserIcon className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                        <span>Patient Health Profile</span>
                      </button>
                    )}

                    <div className="my-1 border-t border-slate-100 dark:border-slate-800" />

                    {/* Export Encounter */}
                    {currentMessages.length > 0 && (
                      <button
                        onClick={() => {
                          setShowToolsMenu(false);
                          downloadTranscript();
                        }}
                        className="w-full px-3.5 py-2 text-left flex items-center gap-2.5 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                      >
                        <Download className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                        <span>Export Encounter</span>
                      </button>
                    )}

                    {/* Clear Conversation */}
                    {currentMessages.length > 0 && (
                      <button
                        onClick={() => {
                          setShowToolsMenu(false);
                          clearCurrentChat();
                        }}
                        className="w-full px-3.5 py-2 text-left flex items-center gap-2.5 hover:bg-red-50 dark:hover:bg-red-950/50 text-red-600 dark:text-red-400"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>Clear Consultation</span>
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </header>

        {/* Optional Clinical Telemetry & Triage Banner */}
        {showPatientBanner && (
          <ClinicalPatientBanner
            vitals={vitals}
            onUpdateVitals={setVitals}
            acuity={acuity}
            onUpdateAcuity={setAcuity}
            patientProfile={userData?.healthProfile}
            user={user}
            onOpenProfile={() => setShowProfileModal(true)}
            onInjectVitals={handleInjectTextToInput}
            onClose={() => setShowPatientBanner(false)}
          />
        )}

        {/* Main Chat Stream Area */}
        <main className="flex-1 overflow-y-auto custom-scrollbar px-3 sm:px-6 lg:px-8 py-4 sm:py-5 space-y-4 sm:space-y-5">
          <div className="max-w-3xl lg:max-w-4xl xl:max-w-5xl mx-auto space-y-4 sm:space-y-5 pb-32 sm:pb-36">
            {/* Optional Dismissible Safety Notice */}
            {showSafetyNotice && (
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center justify-between gap-3 text-xs shadow-xs">
                <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                  <Info className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400 shrink-0" />
                  <span className="text-[11px]">RapidAid clinical decision support. In acute life-threatening emergencies, call 911 immediately.</span>
                </div>
                <button onClick={() => setShowSafetyNotice(false)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Active Medication Scheduled Dose Alert */}
            {activeReminders.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3.5 rounded-xl bg-teal-800 dark:bg-teal-900 text-white shadow-sm flex flex-wrap sm:flex-nowrap items-center justify-between gap-3"
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

            {/* Empty State / Minimal Intake Scenarios */}
            {currentMessages.length === 0 && (
              <div className="py-12 sm:py-20 text-center max-w-lg mx-auto space-y-5">
                <div className="w-12 h-12 rounded-2xl bg-teal-50 dark:bg-teal-950/60 border border-teal-200 dark:border-teal-800 flex items-center justify-center mx-auto text-teal-800 dark:text-teal-300 shadow-xs">
                  <Stethoscope className="w-6 h-6" />
                </div>
                <div className="space-y-1.5">
                  <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 tracking-tight">
                    RapidAid Clinical Triage
                  </h2>
                  <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 font-medium leading-relaxed max-w-sm mx-auto">
                    Enter patient symptoms, vital signs, or pharmacology inquiries to begin evaluation.
                  </p>
                </div>

                {/* Minimalist Starter Chips */}
                <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                  {SUGGESTED_PROMPTS.slice(0, 3).map((item, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendText(undefined, item.prompt)}
                      className="px-3.5 py-2 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-teal-500 dark:hover:border-teal-400 text-slate-700 dark:text-slate-300 text-xs font-medium hover:text-teal-700 dark:hover:text-teal-300 transition-colors shadow-xs flex items-center gap-1.5 text-left"
                    >
                      <span>{item.icon}</span>
                      <span>{item.title}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Conversation Messages */}
            {currentMessages.map((msg, idx) => (
              <ChatMessageItem
                key={msg.id || `${msg.timestamp}-${idx}`}
                msg={msg}
                index={idx}
                isSpeaking={speakingMessageId === (msg.id || `${idx}`)}
                isCopied={copiedId === (msg.id || `${idx}`)}
                onSpeak={speakText}
                onCopy={copyMessage}
                onOpenSoap={() => setShowSoapModal(true)}
              />
            ))}

            {/* Live Generation Typing Pulse - displayed only while waiting for initial text to stream */}
            {isGenerating && (!currentMessages.length || !currentMessages[currentMessages.length - 1]?.text) && (
              <div className="flex items-center gap-3 p-3.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 max-w-xs shadow-xs">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 rounded-full bg-teal-600 animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <span className="text-xs font-semibold text-slate-600 dark:text-slate-400 font-mono text-[11px]">Synthesizing clinical evaluation...</span>
                <button
                  onClick={handleStopGenerating}
                  className="ml-auto p-1 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300"
                  title="Stop Response"
                >
                  <Square className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Error Banner */}
            {errorMessage && (
              <div className="p-3.5 rounded-xl bg-red-50 dark:bg-red-950/60 border border-red-200 dark:border-red-900 text-red-800 dark:text-red-200 text-xs flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400 shrink-0" />
                  <span className="font-semibold">{errorMessage}</span>
                </div>
                <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-700 dark:hover:text-red-300">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <div ref={messagesEndRef} className="h-4" />
          </div>
        </main>

        {/* Sticky Clinical Order & Intake Input Box */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-[#F8FAFC] dark:from-slate-950 via-[#F8FAFC]/95 dark:via-slate-950/95 to-transparent pt-3 sm:pt-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] px-2.5 sm:px-6 lg:px-8 z-20 transition-colors">
          <div className="max-w-3xl lg:max-w-4xl xl:max-w-5xl mx-auto">
            <form
              onSubmit={(e) => handleSendText(e)}
              className="relative bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl shadow-xs p-2 pl-3 sm:pl-3.5 flex items-center gap-2 focus-within:border-teal-600 dark:focus-within:border-teal-500 focus-within:ring-2 focus-within:ring-teal-100 dark:focus-within:ring-teal-950 transition-all"
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
                className="flex-1 bg-transparent border-none outline-none text-xs sm:text-sm font-medium text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 resize-none max-h-32 py-1"
              />

              {isGenerating ? (
                <button
                  type="button"
                  onClick={handleStopGenerating}
                  className="px-2.5 sm:px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 font-semibold text-xs transition-colors flex items-center gap-1.5 shrink-0"
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
            <p className="text-center text-[10px] text-slate-400 dark:text-slate-500 mt-1.5 font-medium">
              RapidAid Clinical Decision Support • Confidential & Secure
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
              className="absolute inset-0 bg-slate-950/40 backdrop-blur-xs"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 10 }}
              className="relative w-full max-w-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-2xl z-10"
            >
              <div className="p-4 sm:p-5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50/80 dark:bg-slate-950/80">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-teal-100 dark:bg-teal-950 border border-teal-200 dark:border-teal-800 flex items-center justify-center text-teal-800 dark:text-teal-300 shrink-0">
                    <BookOpen className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Clinical Knowledge & Reference Library</h3>
                      <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-teal-100 dark:bg-teal-950 text-teal-800 dark:text-teal-300 border dark:border-teal-800 font-bold">VERIFIED</span>
                    </div>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">Authoritative biomedical guidelines and evidence repositories</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowResources(false)}
                  className="p-1.5 hover:bg-slate-200/60 dark:hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
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
                    className="p-3.5 rounded-xl bg-slate-50/80 dark:bg-slate-800/60 hover:bg-teal-50/60 dark:hover:bg-teal-950/40 border border-slate-200 dark:border-slate-700 hover:border-teal-300 dark:hover:border-teal-700 transition-all flex items-start justify-between group block shadow-xs"
                  >
                    <div>
                      <span className="text-[9px] font-mono font-bold text-teal-800 dark:text-teal-300 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-1.5 py-0.5 rounded uppercase tracking-wider">{res.category}</span>
                      <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover:text-teal-900 dark:group-hover:text-teal-200 transition-colors mt-1.5">{res.name}</h4>
                      <p className="text-[11px] text-slate-600 dark:text-slate-400 mt-1 leading-relaxed font-medium">{res.description}</p>
                    </div>
                    <ExternalLink className="w-4 h-4 text-slate-400 dark:text-slate-500 group-hover:text-teal-700 dark:group-hover:text-teal-300 shrink-0 ml-3 mt-1" />
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
