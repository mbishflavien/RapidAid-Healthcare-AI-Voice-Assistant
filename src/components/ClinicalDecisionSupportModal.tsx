import React, { useState } from 'react';
import { 
  ShieldAlert, 
  X, 
  CheckSquare, 
  Square, 
  Activity, 
  AlertTriangle, 
  FilePlus2, 
  Stethoscope,
  Info,
  CheckCircle2
} from 'lucide-react';
import { VitalSigns } from '../types';

interface ClinicalDecisionSupportModalProps {
  isOpen: boolean;
  onClose: () => void;
  vitals: VitalSigns;
  onInjectNote: (note: string) => void;
}

export const ClinicalDecisionSupportModal: React.FC<ClinicalDecisionSupportModalProps> = ({
  isOpen,
  onClose,
  vitals,
  onInjectNote
}) => {
  const [activeTab, setActiveTab] = useState<'qsofa' | 'redflags' | 'gcs'>('qsofa');

  // qSOFA state
  const [qsofaRR, setQsofaRR] = useState(vitals.respiratoryRate >= 22);
  const [qsofaMentation, setQsofaMentation] = useState(false);
  const [qsofaBP, setQsofaBP] = useState(vitals.bloodPressureSystolic <= 100);

  // Red flags state
  const [redFlags, setRedFlags] = useState<{ [key: string]: boolean }>({
    chestPainRadiation: false,
    shortnessOfBreathRest: false,
    thunderclapHeadache: false,
    focalNeuroDeficit: false,
    stridorAirway: false,
    uncontrolledBleeding: false,
    syncopeLossOfConsciousness: false
  });

  // GCS State
  const [gcsEye, setGcsEye] = useState<number>(4);
  const [gcsVerbal, setGcsVerbal] = useState<number>(5);
  const [gcsMotor, setGcsMotor] = useState<number>(6);

  if (!isOpen) return null;

  // qSOFA score calculation
  const qsofaScore = (qsofaRR ? 1 : 0) + (qsofaMentation ? 1 : 0) + (qsofaBP ? 1 : 0);
  const isHighRiskSepsis = qsofaScore >= 2;

  // GCS total
  const gcsTotal = gcsEye + gcsVerbal + gcsMotor;

  // Red flags count
  const activeRedFlagsCount = Object.values(redFlags).filter(Boolean).length;

  const toggleRedFlag = (key: string) => {
    setRedFlags(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleInjectQsofa = () => {
    const text = `[CLINICAL DECISION SUPPORT: qSOFA Sepsis Screening] Score: ${qsofaScore}/3. Criteria: RR≥22: ${qsofaRR ? 'YES' : 'NO'}, Altered Mentation: ${qsofaMentation ? 'YES' : 'NO'}, SBP≤100: ${qsofaBP ? 'YES' : 'NO'}. Interpretation: ${isHighRiskSepsis ? 'HIGH RISK for severe sepsis/poor outcome - requires urgent escalation.' : 'Low risk under current quick screening criteria.'}`;
    onInjectNote(text);
    onClose();
  };

  const handleInjectRedFlags = () => {
    const flaggedList = Object.entries(redFlags)
      .filter(([_, v]) => v)
      .map(([k]) => k.replace(/([A-Z])/g, ' $1').toLowerCase());
    const text = `[CLINICAL RED FLAGS SCREENING] ${activeRedFlagsCount > 0 ? `POSITIVE FOR: ${flaggedList.join(', ')}. URGENT EVALUATION REQUIRED.` : 'All primary red flags negative (no acute neurological deficit, radiation chest pain, or airway compromise).'}`;
    onInjectNote(text);
    onClose();
  };

  const handleInjectGCS = () => {
    const text = `[CLINICAL SCORING: Glasgow Coma Scale (GCS)] Total: ${gcsTotal}/15 (Eye: ${gcsEye}/4, Verbal: ${gcsVerbal}/5, Motor: ${gcsMotor}/6). Interpretation: ${gcsTotal === 15 ? 'Normal / Fully Alert' : gcsTotal >= 13 ? 'Mild Brain Injury / Alteration' : gcsTotal >= 9 ? 'Moderate Impairment' : 'Severe Impairment (Airway protection indicated)'}.`;
    onInjectNote(text);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-slate-950/60 backdrop-blur-xs">
      <div className="w-full max-w-2xl bg-white dark:bg-slate-900 rounded-2xl border border-slate-300 dark:border-slate-800 shadow-2xl flex flex-col max-h-[90vh] overflow-hidden transition-colors">
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-teal-800 text-white flex items-center justify-center shadow-xs">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm sm:text-base font-bold text-slate-900 dark:text-slate-100 font-mono">
                  CLINICAL DECISION SUPPORT (CDS)
                </h3>
                <span className="px-1.5 py-0.2 rounded bg-teal-100 dark:bg-teal-950/80 text-teal-800 dark:text-teal-300 text-[10px] font-mono font-bold border dark:border-teal-800">
                  TRIAGE TOOLS
                </span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">
                Standardized bedside risk calculators & clinical triage matrices
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg hover:bg-slate-200/60 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex border-b border-slate-200 dark:border-slate-800 bg-slate-100/70 dark:bg-slate-950/60 p-1 gap-1">
          <button
            onClick={() => setActiveTab('qsofa')}
            className={`flex-1 py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all text-center ${
              activeTab === 'qsofa' ? 'bg-white dark:bg-slate-800 text-teal-900 dark:text-teal-200 shadow-xs' : 'text-slate-600 dark:text-slate-400 hover:bg-white/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <span className="sm:hidden">qSOFA</span>
            <span className="hidden sm:inline">qSOFA Sepsis</span>
          </button>
          <button
            onClick={() => setActiveTab('redflags')}
            className={`flex-1 py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all text-center ${
              activeTab === 'redflags' ? 'bg-white dark:bg-slate-800 text-teal-900 dark:text-teal-200 shadow-xs' : 'text-slate-600 dark:text-slate-400 hover:bg-white/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <span className="sm:hidden">Red Flags ({activeRedFlagsCount})</span>
            <span className="hidden sm:inline">Red Flag Checklist ({activeRedFlagsCount})</span>
          </button>
          <button
            onClick={() => setActiveTab('gcs')}
            className={`flex-1 py-1.5 sm:py-2 px-2 sm:px-3 rounded-lg text-[11px] sm:text-xs font-mono font-bold transition-all text-center ${
              activeTab === 'gcs' ? 'bg-white dark:bg-slate-800 text-teal-900 dark:text-teal-200 shadow-xs' : 'text-slate-600 dark:text-slate-400 hover:bg-white/60 dark:hover:bg-slate-800/60'
            }`}
          >
            <span className="sm:hidden">GCS ({gcsTotal})</span>
            <span className="hidden sm:inline">Glasgow Coma ({gcsTotal}/15)</span>
          </button>
        </div>

        {/* Tab Content */}
        <div className="p-3.5 sm:p-6 overflow-y-auto space-y-4 custom-scrollbar text-xs">
          {/* qSOFA Tool */}
          {activeTab === 'qsofa' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1 text-slate-700 dark:text-slate-300">
                <span className="font-bold text-slate-900 dark:text-slate-100 block font-mono">quick SOFA (Sequential Organ Failure Assessment)</span>
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  Identifies patients outside the ICU who are at high risk of in-hospital mortality from suspected infection. Score ≥ 2 indicates high risk.
                </p>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-3 p-3 rounded-xl border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={qsofaRR}
                    onChange={(e) => setQsofaRR(e.target.checked)}
                    className="w-4 h-4 accent-teal-700 rounded"
                  />
                  <div className="flex-1">
                    <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">Respiratory Rate ≥ 22 /min</span>
                    <span className="block text-[11px] text-slate-500 dark:text-slate-400">Current triage RR: {vitals.respiratoryRate}/min</span>
                  </div>
                  <span className="font-mono text-slate-400 dark:text-slate-500 font-bold">+1 pt</span>
                </label>

                <label className="flex items-center gap-3 p-3 rounded-xl border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={qsofaMentation}
                    onChange={(e) => setQsofaMentation(e.target.checked)}
                    className="w-4 h-4 accent-teal-700 rounded"
                  />
                  <div className="flex-1">
                    <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">Altered Mental Status (GCS &lt; 15)</span>
                    <span className="block text-[11px] text-slate-500 dark:text-slate-400">Disorientation, confusion, or somnolence</span>
                  </div>
                  <span className="font-mono text-slate-400 dark:text-slate-500 font-bold">+1 pt</span>
                </label>

                <label className="flex items-center gap-3 p-3 rounded-xl border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 cursor-pointer transition-colors">
                  <input
                    type="checkbox"
                    checked={qsofaBP}
                    onChange={(e) => setQsofaBP(e.target.checked)}
                    className="w-4 h-4 accent-teal-700 rounded"
                  />
                  <div className="flex-1">
                    <span className="font-bold text-slate-800 dark:text-slate-200 font-mono">Systolic Blood Pressure ≤ 100 mmHg</span>
                    <span className="block text-[11px] text-slate-500 dark:text-slate-400">Current triage SBP: {vitals.bloodPressureSystolic} mmHg</span>
                  </div>
                  <span className="font-mono text-slate-400 dark:text-slate-500 font-bold">+1 pt</span>
                </label>
              </div>

              {/* Score Display */}
              <div className={`p-4 rounded-xl border flex items-center justify-between ${
                isHighRiskSepsis ? 'bg-red-50 dark:bg-red-950/70 border-red-300 dark:border-red-800 text-red-900 dark:text-red-200' : 'bg-teal-50 dark:bg-teal-950/70 border-teal-200 dark:border-teal-800 text-teal-900 dark:text-teal-200'
              }`}>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold font-mono">qSOFA SCORE: {qsofaScore} / 3</span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                      isHighRiskSepsis ? 'bg-red-700 text-white' : 'bg-teal-700 text-white'
                    }`}>
                      {isHighRiskSepsis ? 'CRITICAL HIGH RISK' : 'LOW RISK'}
                    </span>
                  </div>
                  <p className="text-[11px] mt-0.5">
                    {isHighRiskSepsis 
                      ? 'Patient meets criteria for high risk of sepsis complications. Blood cultures, lactate, and immediate medical escalation recommended.' 
                      : 'Criteria not currently met for high risk. Continue standard monitoring.'}
                  </p>
                </div>
                <button
                  onClick={handleInjectQsofa}
                  className="px-3 py-1.5 rounded-lg bg-teal-800 hover:bg-teal-900 text-white font-semibold text-xs transition-colors shrink-0 shadow-xs flex items-center gap-1.5"
                >
                  <FilePlus2 className="w-3.5 h-3.5" />
                  <span>Insert Score</span>
                </button>
              </div>
            </div>
          )}

          {/* Red Flags Checklist */}
          {activeTab === 'redflags' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1">
                <span className="font-bold text-slate-900 dark:text-slate-100 block font-mono">Clinical Red Flag Screen</span>
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  Critical exclusionary symptoms requiring immediate emergency department evaluation or physician intervention.
                </p>
              </div>

              <div className="space-y-1.5">
                {[
                  { key: 'chestPainRadiation', label: 'Crushing or radiating substernal chest discomfort (jaw, left arm)', desc: 'Rule out Acute Coronary Syndrome (ACS)' },
                  { key: 'shortnessOfBreathRest', label: 'Acute dyspnea or tachypnea at rest with cyanosis', desc: 'Rule out pulmonary embolism or acute pulmonary edema' },
                  { key: 'thunderclapHeadache', label: 'Sudden onset "worst headache of life" (thunderclap)', desc: 'Rule out Subarachnoid Hemorrhage (SAH)' },
                  { key: 'focalNeuroDeficit', label: 'Sudden unilateral weakness, facial droop, or speech slurring', desc: 'Rule out Acute Ischemic Stroke (BEFAST)' },
                  { key: 'stridorAirway', label: 'Inspiratory stridor, drooling, or severe airway compromise', desc: 'Rule out epiglottitis or severe foreign body obstruction' },
                  { key: 'uncontrolledBleeding', label: 'Pulsatile, arterial, or uncontrolled hemorrhage', desc: 'Direct compression & surgical consultation' },
                  { key: 'syncopeLossOfConsciousness', label: 'Unexplained syncope with head trauma or arrhythmia', desc: 'Rule out cardiogenic collapse or intracranial bleed' }
                ].map((item) => (
                  <label
                    key={item.key}
                    onClick={() => toggleRedFlag(item.key)}
                    className={`flex items-start gap-3 p-3 rounded-xl border cursor-pointer transition-all ${
                      redFlags[item.key]
                        ? 'bg-red-50/80 dark:bg-red-950/60 border-red-300 dark:border-red-800 text-red-950 dark:text-red-200'
                        : 'bg-white dark:bg-slate-800/80 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'
                    }`}
                  >
                    <div className="mt-0.5">
                      {redFlags[item.key] ? (
                        <CheckSquare className="w-4 h-4 text-red-600 dark:text-red-400" />
                      ) : (
                        <Square className="w-4 h-4 text-slate-400 dark:text-slate-500" />
                      )}
                    </div>
                    <div className="flex-1">
                      <span className="font-bold text-xs block">{item.label}</span>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">{item.desc}</span>
                    </div>
                  </label>
                ))}
              </div>

              <div className="pt-2 flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-slate-700 dark:text-slate-300">
                  Active Red Flags: <strong className={activeRedFlagsCount > 0 ? 'text-red-700 dark:text-red-400' : 'text-emerald-700 dark:text-emerald-400'}>{activeRedFlagsCount}</strong>
                </span>
                <button
                  onClick={handleInjectRedFlags}
                  className="px-3.5 py-1.5 rounded-lg bg-teal-800 hover:bg-teal-900 text-white font-semibold text-xs transition-colors shadow-xs flex items-center gap-1.5"
                >
                  <FilePlus2 className="w-3.5 h-3.5" />
                  <span>Insert Screening Result</span>
                </button>
              </div>
            </div>
          )}

          {/* GCS Tool */}
          {activeTab === 'gcs' && (
            <div className="space-y-4">
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1">
                <span className="font-bold text-slate-900 dark:text-slate-100 block font-mono">Glasgow Coma Scale (GCS)</span>
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  Standard neurological assessment quantifying consciousness across Eye (1-4), Verbal (1-5), and Motor (1-6) responses.
                </p>
              </div>

              <div className="space-y-3">
                {/* Eye Opening */}
                <div>
                  <label className="font-bold text-slate-800 dark:text-slate-200 font-mono block mb-1">
                    Eye Opening (1 - 4): Currently {gcsEye}/4
                  </label>
                  <select
                    value={gcsEye}
                    onChange={(e) => setGcsEye(parseInt(e.target.value))}
                    className="w-full p-2 rounded-lg border border-slate-300 dark:border-slate-700 font-mono text-xs bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                  >
                    <option value={4}>4 - Spontaneous opening</option>
                    <option value={3}>3 - To sound / verbal command</option>
                    <option value={2}>2 - To pressure / pain</option>
                    <option value={1}>1 - None</option>
                  </select>
                </div>

                {/* Verbal Response */}
                <div>
                  <label className="font-bold text-slate-800 dark:text-slate-200 font-mono block mb-1">
                    Verbal Response (1 - 5): Currently {gcsVerbal}/5
                  </label>
                  <select
                    value={gcsVerbal}
                    onChange={(e) => setGcsVerbal(parseInt(e.target.value))}
                    className="w-full p-2 rounded-lg border border-slate-300 dark:border-slate-700 font-mono text-xs bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                  >
                    <option value={5}>5 - Orientated and conversing</option>
                    <option value={4}>4 - Confused conversation</option>
                    <option value={3}>3 - Inappropriate words</option>
                    <option value={2}>2 - Incomprehensible sounds</option>
                    <option value={1}>1 - None</option>
                  </select>
                </div>

                {/* Motor Response */}
                <div>
                  <label className="font-bold text-slate-800 dark:text-slate-200 font-mono block mb-1">
                    Motor Response (1 - 6): Currently {gcsMotor}/6
                  </label>
                  <select
                    value={gcsMotor}
                    onChange={(e) => setGcsMotor(parseInt(e.target.value))}
                    className="w-full p-2 rounded-lg border border-slate-300 dark:border-slate-700 font-mono text-xs bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100"
                  >
                    <option value={6}>6 - Obeys commands</option>
                    <option value={5}>5 - Localizing to pain</option>
                    <option value={4}>4 - Normal flexion (withdrawal)</option>
                    <option value={3}>3 - Abnormal flexion (decorticate)</option>
                    <option value={2}>2 - Extension (decerebrate)</option>
                    <option value={1}>1 - None</option>
                  </select>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 flex items-center justify-between">
                <div>
                  <span className="text-base font-bold font-mono text-slate-900 dark:text-slate-100">
                    TOTAL GCS: {gcsTotal} / 15
                  </span>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                    {gcsTotal === 15 ? 'Fully Alert & Oriented (Normal)' : gcsTotal >= 13 ? 'Mild Alteration' : gcsTotal >= 9 ? 'Moderate Neurological Deficit' : 'Severe Coma (GCS ≤ 8 Intubate)'}
                  </p>
                </div>
                <button
                  onClick={handleInjectGCS}
                  className="px-3.5 py-1.5 rounded-lg bg-teal-800 hover:bg-teal-900 text-white font-semibold text-xs transition-colors shadow-xs flex items-center gap-1.5"
                >
                  <FilePlus2 className="w-3.5 h-3.5" />
                  <span>Insert GCS</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
