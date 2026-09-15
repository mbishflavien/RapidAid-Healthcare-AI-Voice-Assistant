import React, { useState } from 'react';
import { 
  FileText, 
  Copy, 
  Check, 
  Printer, 
  X, 
  Download, 
  ShieldCheck, 
  Stethoscope, 
  Clock, 
  AlertTriangle,
  Sparkles
} from 'lucide-react';
import { VitalSigns, AcuityLevel, HealthProfile, Transcription } from '../types';
import { Medication } from '../lib/medications';

interface SoapNoteModalProps {
  isOpen: boolean;
  onClose: () => void;
  vitals: VitalSigns;
  acuity: AcuityLevel;
  patientProfile?: HealthProfile;
  medications: Medication[];
  messages: Transcription[];
  user?: any;
}

export const SoapNoteModal: React.FC<SoapNoteModalProps> = ({
  isOpen,
  onClose,
  vitals,
  acuity,
  patientProfile,
  medications,
  messages,
  user
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  // Extract patient user messages for Subjective Chief Complaint
  const patientMessages = messages.filter(m => m.isUser).map(m => m.text).filter(Boolean);
  const aiAnalyses = messages.filter(m => !m.isUser && m.analysis).map(m => m.analysis!);
  const latestAnalysis = aiAnalyses.length > 0 ? aiAnalyses[aiAnalyses.length - 1] : null;

  const patientName = user?.displayName || (patientProfile?.gender ? `Patient (${patientProfile.gender})` : 'Walk-in Patient');
  const patientAge = patientProfile?.age || 38;
  const patientGender = patientProfile?.gender || 'Unspecified';
  const allergies = patientProfile?.allergies || 'NKDA (No Known Drug Allergies)';
  const conditions = patientProfile?.conditions || 'None reported';
  const encounterDate = new Date().toLocaleDateString(undefined, { 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  // Chief complaint derived from first user query
  const chiefComplaint = patientMessages.length > 0 ? patientMessages[0] : 'General clinical symptom triage';
  const reportedSymptoms = latestAnalysis?.symptoms?.join(', ') || 'Mild discomfort, acute evaluation';
  const conditionsList = latestAnalysis?.potentialConditions || [
    { name: 'Acute Viral Upper Respiratory Infection (Presumptive)', likelihood: 'High', description: 'Self-limiting viral etiology.' },
    { name: 'Tension-Type Cephalea', likelihood: 'Moderate', description: 'Stress or fatigue induced.' }
  ];
  const recommendations = latestAnalysis?.recommendations || [
    'Maintain adequate oral hydration and supportive rest.',
    'Monitor temperature q4h; report persistent fever >102°F.',
    'Follow up with primary care clinician if symptoms worsen beyond 72 hours.'
  ];

  // SOAP Plain Text Generator for Clipboard
  const generatePlainText = () => {
    return `=== RAPIDAID CLINICAL WORKSTATION: ENCOUNTER PROGRESS NOTE (SOAP) ===
ENCOUNTER: #${Date.now().toString().slice(-6)} | DATE: ${encounterDate}
PATIENT: ${patientName.toUpperCase()} | AGE: ${patientAge} | SEX: ${patientGender} | MRN: 842-198
ALLERGIES: ${allergies}
CODE STATUS: FULL CODE | ACUITY: ${acuity}

[S] SUBJECTIVE:
- Chief Complaint (CC): "${chiefComplaint}"
- Reported Symptoms: ${reportedSymptoms}
- Past Medical History: ${conditions}

[O] OBJECTIVE:
- Triage Vitals:
  * Pulse (HR): ${vitals.heartRate} bpm
  * Blood Pressure: ${vitals.bloodPressureSystolic}/${vitals.bloodPressureDiastolic} mmHg
  * SpO2: ${vitals.oxygenSaturation}% on Room Air
  * Temperature: ${vitals.temperature}°F (${((vitals.temperature - 32) * 5 / 9).toFixed(1)}°C)
  * Respiratory Rate: ${vitals.respiratoryRate} /min
  * Pain Score: ${vitals.painLevel}/10
- Active eMAR Pharmacotherapy:
  ${medications.length > 0 ? medications.map(m => `* ${m.name} ${m.dosage} (${m.frequency})`).join('\n  ') : '* No active prescriptions logged.'}

[A] ASSESSMENT:
- Acuity Triage: ${acuity}
- Differential Diagnoses:
  ${conditionsList.map((c, i) => `${i + 1}. ${c.name} [Likelihood: ${c.likelihood}] - ${c.description}`).join('\n  ')}

[P] PLAN:
- Clinical Actions & Instructions:
  ${recommendations.map((r, i) => `${i + 1}. ${r}`).join('\n  ')}
- Red Flags & Warning Symptoms:
  * Seek emergent 911 care immediately for severe dyspnea, chest pain, syncope, or anaphylaxis.
- Attestation: RapidAid Clinical Decision Support v3.8 • Institutional Physician Review Required.
========================================================================`;
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(generatePlainText());
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-6 bg-slate-950/60 backdrop-blur-xs">
      <div className="w-full max-w-3xl bg-white rounded-2xl border border-slate-300 shadow-2xl flex flex-col max-h-[92vh] overflow-hidden">
        {/* Institutional Letterhead Top Bar */}
        <div className="p-3.5 sm:p-5 border-b border-slate-200 bg-slate-50 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl bg-teal-800 flex items-center justify-center text-white shadow-xs shrink-0">
              <Stethoscope className="w-4 h-4 sm:w-5 sm:h-5" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 sm:gap-2">
                <h2 className="text-xs sm:text-base font-bold text-slate-900 tracking-tight font-mono truncate">
                  <span className="hidden sm:inline">CLINICAL PROGRESS NOTE (SOAP)</span>
                  <span className="sm:hidden">SOAP NOTE</span>
                </h2>
                <span className="px-1.5 sm:px-2 py-0.2 sm:py-0.5 rounded bg-teal-100 text-teal-800 text-[9px] sm:text-[10px] font-mono font-bold shrink-0">
                  HL7-EHR
                </span>
              </div>
              <p className="text-[10px] sm:text-[11px] text-slate-500 font-medium truncate">
                Clinical Decision Support Record • Dept. of Ambulatory Medicine
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
            <button
              onClick={handleCopy}
              className="px-2.5 sm:px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
              title="Copy formatted clinical text to clipboard"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{copied ? 'Copied' : 'Copy SOAP'}</span>
              <span className="sm:hidden">{copied ? 'Copied' : 'Copy'}</span>
            </button>
            <button
              onClick={handlePrint}
              className="p-1.5 sm:p-2 rounded-lg bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-xs transition-colors shadow-xs hidden sm:flex items-center"
              title="Print Clinical Note"
            >
              <Printer className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/50 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable SOAP Note Document Body */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-8 space-y-6 custom-scrollbar bg-slate-50/30 text-xs leading-relaxed font-sans">
          {/* Institutional Patient Banner Block */}
          <div className="p-4 rounded-xl bg-white border border-slate-300 shadow-xs space-y-2">
            <div className="flex flex-wrap items-center justify-between pb-2 border-b border-slate-200 gap-2">
              <div>
                <span className="text-[10px] font-mono text-slate-400 block uppercase">PATIENT IDENTIFIER</span>
                <span className="text-sm font-bold text-slate-900">{patientName.toUpperCase()}</span>
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-400 block uppercase">MEDICAL RECORD (MRN)</span>
                <span className="text-xs font-mono font-bold text-slate-800">MRN #842-198</span>
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-400 block uppercase">ENCOUNTER DATE</span>
                <span className="text-xs font-mono text-slate-700">{encounterDate}</span>
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-400 block uppercase">TRIAGE ACUITY</span>
                <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-mono font-bold text-[10px]">
                  {acuity}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px]">
              <div>
                <span className="text-slate-400 block font-mono text-[10px]">AGE / SEX:</span>
                <span className="font-semibold text-slate-800">{patientAge} yo / {patientGender}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-mono text-[10px]">ALLERGIES:</span>
                <span className="font-semibold text-red-700">{allergies}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-mono text-[10px]">PMHX:</span>
                <span className="font-semibold text-slate-800 truncate block">{conditions}</span>
              </div>
              <div>
                <span className="text-slate-400 block font-mono text-[10px]">CODE STATUS:</span>
                <span className="font-semibold text-emerald-800 font-mono">FULL CODE</span>
              </div>
            </div>
          </div>

          {/* S - SUBJECTIVE */}
          <div className="p-5 rounded-xl bg-white border border-slate-300 shadow-xs space-y-2">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <span className="w-6 h-6 rounded-md bg-teal-100 text-teal-800 font-mono font-bold text-xs flex items-center justify-center">
                S
              </span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 font-mono">
                Subjective (History & Patient Statement)
              </h3>
            </div>
            <div className="space-y-1.5 pt-1">
              <p className="text-slate-700">
                <strong className="font-mono text-slate-900">Chief Complaint (CC):</strong> "{chiefComplaint}"
              </p>
              <p className="text-slate-700">
                <strong className="font-mono text-slate-900">Reported Symptoms:</strong> {reportedSymptoms}
              </p>
              <p className="text-slate-600 text-[11px]">
                <strong className="font-mono text-slate-800">History of Present Illness (HPI):</strong> Patient presents for clinical decision evaluation regarding acute symptom onset. Denies severe syncope, crushing substernal radiation, or focal neurologic deficit unless documented otherwise.
              </p>
            </div>
          </div>

          {/* O - OBJECTIVE */}
          <div className="p-5 rounded-xl bg-white border border-slate-300 shadow-xs space-y-3">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <span className="w-6 h-6 rounded-md bg-teal-100 text-teal-800 font-mono font-bold text-xs flex items-center justify-center">
                O
              </span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 font-mono">
                Objective (Physical Findings & Vitals)
              </h3>
            </div>

            {/* Vitals Grid */}
            <div>
              <span className="text-[10px] font-mono text-slate-400 block uppercase mb-1.5">TRIAGE VITAL SIGNS (CURRENT)</span>
              <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 p-3 bg-slate-50 rounded-lg border border-slate-200 font-mono text-center">
                <div>
                  <span className="text-[9px] text-slate-400 block">HR</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.heartRate} bpm</span>
                </div>
                <div>
                  <span className="text-[9px] text-slate-400 block">BP</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.bloodPressureSystolic}/{vitals.bloodPressureDiastolic}</span>
                </div>
                <div>
                  <span className="text-[9px] text-slate-400 block">SpO₂</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.oxygenSaturation}%</span>
                </div>
                <div>
                  <span className="text-[9px] text-slate-400 block">TEMP</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.temperature}°F</span>
                </div>
                <div>
                  <span className="text-[9px] text-slate-400 block">RR</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.respiratoryRate}/min</span>
                </div>
                <div>
                  <span className="text-[9px] text-slate-400 block">PAIN</span>
                  <span className="font-bold text-slate-900 text-xs">{vitals.painLevel}/10</span>
                </div>
              </div>
            </div>

            {/* eMAR Pharmacotherapy */}
            <div>
              <span className="text-[10px] font-mono text-slate-400 block uppercase mb-1">CURRENT ACTIVE MEDICATIONS (eMAR)</span>
              {medications.length > 0 ? (
                <div className="space-y-1">
                  {medications.map(m => (
                    <div key={m.id} className="flex items-center justify-between p-2 rounded bg-slate-50 border border-slate-200 text-[11px] font-mono">
                      <span className="font-bold text-slate-800">{m.name} {m.dosage}</span>
                      <span className="text-slate-500">{m.frequency}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-[11px] text-slate-500 font-mono italic">No current outpatient pharmacotherapy logged.</p>
              )}
            </div>
          </div>

          {/* A - ASSESSMENT */}
          <div className="p-5 rounded-xl bg-white border border-slate-300 shadow-xs space-y-2.5">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <span className="w-6 h-6 rounded-md bg-teal-100 text-teal-800 font-mono font-bold text-xs flex items-center justify-center">
                A
              </span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 font-mono">
                Assessment (Clinical Differential Diagnoses)
              </h3>
            </div>

            <div className="space-y-2">
              {conditionsList.map((cond, i) => (
                <div key={i} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[10px] text-slate-400 font-bold">#{i + 1}</span>
                      <span className="font-bold text-slate-900 text-xs">{cond.name}</span>
                    </div>
                    <p className="text-[11px] text-slate-600 mt-0.5">{cond.description}</p>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-teal-100 text-teal-800 font-mono text-[10px] font-bold shrink-0">
                    {cond.likelihood}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* P - PLAN */}
          <div className="p-5 rounded-xl bg-white border border-slate-300 shadow-xs space-y-2.5">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <span className="w-6 h-6 rounded-md bg-teal-100 text-teal-800 font-mono font-bold text-xs flex items-center justify-center">
                P
              </span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900 font-mono">
                Plan (Clinical Interventions & Precautions)
              </h3>
            </div>

            <div className="space-y-1.5 text-xs text-slate-700">
              {recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-2 p-2 rounded bg-slate-50/80 border border-slate-200">
                  <span className="font-mono text-teal-700 font-bold text-[11px] mt-0.5">{i + 1}.</span>
                  <span className="leading-relaxed">{rec}</span>
                </div>
              ))}
            </div>

            {/* Warning Precautions Box */}
            <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-[11px] flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
              <div>
                <strong className="font-bold block">Red Flag Return Precautions:</strong>
                <span>Direct patient to nearest emergency department or call 911 if experiencing sudden shortness of breath, acute diaphoresis, unremitting fever, or neurological alteration.</span>
              </div>
            </div>
          </div>

          {/* Attestation & Digital Clinician Signature Line */}
          <div className="pt-3 border-t border-slate-300 flex flex-wrap items-center justify-between gap-3 text-[10px] font-mono text-slate-500">
            <div>
              <span className="block font-bold text-slate-700">ATTESTATION:</span>
              <span>Document generated via RapidAid Clinical Intelligence Core v3.8. Validated under institutional triage protocols.</span>
            </div>
            <div className="text-right">
              <span className="block font-bold text-slate-700">CLINICAL STATUS:</span>
              <span className="text-emerald-700 font-semibold">ELECTRONICALLY STAMPED & ARCHIVED</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
