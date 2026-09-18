import React from 'react';
import { motion } from 'motion/react';
import Markdown from 'react-markdown';
import { 
  ShieldAlert, 
  FileText, 
  CheckCircle2, 
  Volume2, 
  VolumeX, 
  Copy, 
  Check 
} from 'lucide-react';
import { Transcription } from '../types';

interface ChatMessageItemProps {
  msg: Transcription;
  index: number;
  isSpeaking: boolean;
  isCopied: boolean;
  onSpeak: (text: string, id: string) => void;
  onCopy: (text: string, id: string) => void;
  onOpenSoap: () => void;
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = React.memo(({
  msg,
  index,
  isSpeaking,
  isCopied,
  onSpeak,
  onCopy,
  onOpenSoap
}) => {
  const messageKey = msg.id || `${index}`;

  return (
    <div
      id={`msg-${msg.id}`}
      className={`flex scroll-mt-20 ${msg.isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`relative ${msg.analysis ? 'w-full' : 'max-w-[92%] sm:max-w-[85%] md:max-w-[80%]'}`}>
        {msg.isUser ? (
          // Patient Message Bubble
          <div className="p-3.5 sm:p-4 rounded-2xl bg-slate-900 dark:bg-slate-800 border border-slate-800 dark:border-slate-700 text-white shadow-xs">
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
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-xs mb-3">
                {/* Card Header with Clinical Urgency Pill */}
                <div className={`px-4 sm:px-5 py-3 sm:py-3.5 flex flex-wrap items-center justify-between gap-2 border-b ${
                  msg.analysis.urgency === 'Emergency' ? 'bg-red-50/90 dark:bg-red-950/80 border-red-200 dark:border-red-900 text-red-800 dark:text-red-300' :
                  msg.analysis.urgency === 'High' ? 'bg-amber-50/90 dark:bg-amber-950/80 border-amber-200 dark:border-amber-900 text-amber-800 dark:text-amber-300' :
                  'bg-teal-50/90 dark:bg-teal-950/80 border-teal-200 dark:border-teal-900 text-teal-900 dark:text-teal-200'
                }`}>
                  <div className="flex items-center gap-2.5">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                      msg.analysis.urgency === 'Emergency' ? 'bg-red-100 dark:bg-red-900/80 text-red-700 dark:text-red-300' :
                      msg.analysis.urgency === 'High' ? 'bg-amber-100 dark:bg-amber-900/80 text-amber-700 dark:text-amber-300' :
                      'bg-teal-100 dark:bg-teal-900/80 text-teal-800 dark:text-teal-300'
                    }`}>
                      <ShieldAlert className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-xs font-bold uppercase tracking-wider">Symptom Summary & Care Level</h4>
                        <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-white/60 dark:bg-slate-800/80 border border-current/20">GUIDANCE</span>
                      </div>
                      <p className="text-[11px] font-medium opacity-85">
                         Reported: {msg.analysis.symptoms.join(', ')}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
                    <button
                      onClick={onOpenSoap}
                      className="px-2 py-1 rounded bg-white/90 dark:bg-slate-900/90 hover:bg-white dark:hover:bg-slate-800 text-slate-800 dark:text-slate-200 text-[10px] font-mono font-bold flex items-center gap-1 border border-current/20 transition-colors shadow-xs"
                      title="Open Encounter Progress Note"
                    >
                      <FileText className="w-3 h-3 text-teal-800 dark:text-teal-400" />
                      <span>Clinical Chart</span>
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
                      <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                        Possible Causes
                      </p>
                      <span className="text-[9px] font-mono text-slate-400 dark:text-slate-500">HOW LIKELY</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      {msg.analysis.potentialConditions.map((cond, cIdx) => (
                        <div key={cIdx} className="p-3 rounded-xl bg-slate-50/80 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-bold text-slate-900 dark:text-slate-100">{cond.name}</span>
                            <span className="text-[9px] font-mono font-bold text-teal-800 dark:text-teal-300 bg-teal-50 dark:bg-teal-950/80 border border-teal-200 dark:border-teal-800 px-1.5 py-0.5 rounded">
                              {cond.likelihood}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-600 dark:text-slate-400 font-medium leading-relaxed">{cond.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recommended Clinical Roadmap */}
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2.5">
                      Recommended Next Steps
                    </p>
                    <div className="space-y-1.5">
                      {msg.analysis.recommendations.map((rec, rIdx) => (
                        <div key={rIdx} className="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-300 font-medium p-2 rounded-lg bg-slate-50/50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800">
                          <CheckCircle2 className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400 shrink-0 mt-0.5" />
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
              <div className="p-3.5 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 shadow-xs relative">
                <div className="prose prose-sm prose-slate dark:prose-invert max-w-none prose-p:leading-relaxed prose-headings:font-bold prose-headings:text-slate-900 dark:prose-headings:text-slate-100 prose-ul:my-2 prose-li:my-0.5 text-xs sm:text-sm overflow-x-auto">
                  <Markdown>{msg.text}</Markdown>
                </div>

                {/* Message Footer / Clinical Telemetry */}
                <div className="mt-3 sm:mt-4 pt-2.5 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 dark:text-slate-500">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-teal-700 text-white flex items-center justify-center text-[9px] font-bold">
                      +
                    </div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">RapidAid Health AI</span>
                    {msg.fromVoice && (
                      <span className="px-1.5 py-0.2 rounded bg-teal-50 dark:bg-teal-950/80 border border-teal-200 dark:border-teal-800 text-teal-800 dark:text-teal-300 text-[9px] font-mono font-bold">
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
                      onClick={() => onSpeak(msg.text!, messageKey)}
                      className={`p-1.5 rounded-lg transition-colors ${
                        isSpeaking
                          ? 'bg-teal-50 dark:bg-teal-950/80 text-teal-700 dark:text-teal-300 font-bold'
                          : 'hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'
                      }`}
                      title={isSpeaking ? "Stop Audio Readout" : "Audio Readout"}
                    >
                      {isSpeaking ? (
                        <VolumeX className="w-3.5 h-3.5 text-teal-700 dark:text-teal-300 animate-pulse" />
                      ) : (
                        <Volume2 className="w-3.5 h-3.5" />
                      )}
                    </button>

                    {/* Copy Clinical Advice */}
                    <button
                      onClick={() => onCopy(msg.text!, messageKey)}
                      className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                      title="Copy Clinical Text"
                    >
                      {isCopied ? (
                        <Check className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400" />
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
    </div>
  );
}, (prevProps, nextProps) => {
  // Memoization comparison: only re-render if text, analysis, or speaking/copied state changed
  return (
    prevProps.msg.id === nextProps.msg.id &&
    prevProps.msg.text === nextProps.msg.text &&
    prevProps.msg.analysis === nextProps.msg.analysis &&
    prevProps.isSpeaking === nextProps.isSpeaking &&
    prevProps.isCopied === nextProps.isCopied
  );
});
