import React, { useState } from 'react';
import { 
  Heart, 
  Activity, 
  Thermometer, 
  Gauge, 
  Wind, 
  AlertTriangle, 
  CheckCircle2, 
  Edit3, 
  ChevronDown, 
  ChevronUp, 
  FilePlus2,
  ShieldCheck,
  User,
  Clock,
  X
} from 'lucide-react';
import { VitalSigns, AcuityLevel, HealthProfile } from '../types';

interface ClinicalPatientBannerProps {
  vitals: VitalSigns;
  onUpdateVitals: (newVitals: VitalSigns) => void;
  acuity: AcuityLevel;
  onUpdateAcuity: (newAcuity: AcuityLevel) => void;
  patientProfile?: HealthProfile;
  user?: any;
  onOpenProfile: () => void;
  onInjectVitals: (vitalsSummary: string) => void;
  onClose?: () => void;
}

export const ClinicalPatientBanner: React.FC<ClinicalPatientBannerProps> = ({
  vitals,
  onUpdateVitals,
  acuity,
  onUpdateAcuity,
  patientProfile,
  user,
  onOpenProfile,
  onInjectVitals,
  onClose
}) => {
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [showVitalsModal, setShowVitalsModal] = useState(false);

  // Vitals edit form state
  const [tempVitals, setTempVitals] = useState<VitalSigns>(vitals);

  // Check for abnormal clinical vitals
  const isFebrile = vitals.temperature >= 100.4;
  const isHypothermic = vitals.temperature < 96.0;
  const isTachycardic = vitals.heartRate > 100;
  const isBradycardic = vitals.heartRate < 60;
  const isHypoxic = vitals.oxygenSaturation < 95;
  const isHypertensive = vitals.bloodPressureSystolic >= 140 || vitals.bloodPressureDiastolic >= 90;
  const isTachypneic = vitals.respiratoryRate > 20;

  const handleSaveVitals = (e: React.FormEvent) => {
    e.preventDefault();
    onUpdateVitals({
      ...tempVitals,
      lastRecorded: Date.now()
    });
    setShowVitalsModal(false);
  };

  const handleInjectVitalsSummary = () => {
    const summary = `Current Clinical Triage Vitals: HR ${vitals.heartRate} bpm, BP ${vitals.bloodPressureSystolic}/${vitals.bloodPressureDiastolic} mmHg, SpO2 ${vitals.oxygenSaturation}%, Temp ${vitals.temperature}°F, RR ${vitals.respiratoryRate}/min, Pain ${vitals.painLevel}/10. Patient Acuity: ${acuity}.`;
    onInjectVitals(summary);
  };

  const getAcuityColor = (lvl: AcuityLevel) => {
    switch (lvl) {
      case 'ESI-1': return 'bg-red-700 text-white border-red-800';
      case 'ESI-2': return 'bg-orange-600 text-white border-orange-700';
      case 'ESI-3': return 'bg-amber-500 text-slate-950 border-amber-600 font-bold';
      case 'ESI-4': return 'bg-teal-700 text-white border-teal-800';
      case 'ESI-5': return 'bg-slate-700 text-white border-slate-800';
    }
  };

  const getAcuityLabel = (lvl: AcuityLevel) => {
    switch (lvl) {
      case 'ESI-1': return 'ESI-1: Resuscitation (Immediate)';
      case 'ESI-2': return 'ESI-2: Emergent (<15m)';
      case 'ESI-3': return 'ESI-3: Urgent (<30m)';
      case 'ESI-4': return 'ESI-4: Less Urgent (<60m)';
      case 'ESI-5': return 'ESI-5: Non-Urgent (<120m)';
    }
  };

  // Derive patient name & MRN
  const patientName = user?.displayName || (patientProfile?.gender ? `Patient (${patientProfile.gender})` : 'Walk-in Patient');
  const patientAge = patientProfile?.age ? `${patientProfile.age} yo` : '38 yo';
  const patientGender = patientProfile?.gender || 'Unspecified';
  const allergiesText = patientProfile?.allergies?.trim() || 'NKDA (No Known Drug Allergies)';
  const bloodType = patientProfile?.bloodType || 'O+ (Pos)';

  return (
    <div className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-900 dark:text-slate-100 text-xs shadow-xs relative z-15 select-none transition-colors">
      {/* Top Telemetry Strip: Hospital Ward, Patient Demographics & Acuity */}
      <div className="px-3 sm:px-6 py-2 bg-slate-100/90 dark:bg-slate-950/80 border-b border-slate-200/80 dark:border-slate-800">
        <div className="flex items-center justify-between gap-2">
          {/* Patient Demographic Identity */}
          <div className="flex items-center gap-1.5 sm:gap-2 min-w-0 flex-1">
            <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-emerald-600 animate-pulse shrink-0" />
            <div className="flex items-center gap-1 sm:gap-1.5 min-w-0">
              <span className="font-bold text-slate-900 dark:text-slate-100 uppercase tracking-tight text-[11px] sm:text-xs truncate">
                {patientName}
              </span>
              <span className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono text-[9px] sm:text-[10px] font-semibold shrink-0">
                #842-198
              </span>
            </div>

            {/* Desktop Demographics inline */}
            <div className="hidden md:flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-400 font-medium ml-2">
              <span>{patientAge}</span>
              <span>•</span>
              <span className="capitalize">{patientGender}</span>
              <span>•</span>
              <span className="font-mono text-slate-700 dark:text-slate-300">{bloodType}</span>
              <span className="px-1.5 py-0.2 rounded bg-emerald-100 dark:bg-emerald-950/80 text-emerald-800 dark:text-emerald-300 border dark:border-emerald-800 font-mono text-[9px] font-bold">
                FULL CODE
              </span>
            </div>
          </div>

          {/* Right Station Controls: Acuity Badge, Vitals Editor, Expand/Collapse */}
          <div className="flex items-center gap-1 sm:gap-1.5 shrink-0">
            {/* Acuity (ESI) Dropdown */}
            <div className="flex items-center gap-1">
              <label className="text-[9px] sm:text-[10px] uppercase font-mono font-bold text-slate-500 dark:text-slate-400 hidden xl:inline">
                ACUITY:
              </label>
              <select
                value={acuity}
                onChange={(e) => onUpdateAcuity(e.target.value as AcuityLevel)}
                className={`px-1.5 sm:px-2 py-1 rounded text-[9px] sm:text-[10px] font-mono font-bold uppercase tracking-wider border cursor-pointer max-w-[105px] xs:max-w-[130px] sm:max-w-none ${getAcuityColor(acuity)}`}
                title="Emergency Severity Index Triage Level"
              >
                <option value="ESI-1">ESI-1 (Immediate)</option>
                <option value="ESI-2">ESI-2 (Emergent)</option>
                <option value="ESI-3">ESI-3 (Urgent)</option>
                <option value="ESI-4">ESI-4 (Less Urgent)</option>
                <option value="ESI-5">ESI-5 (Non-Urgent)</option>
              </select>
            </div>

            {/* Quick Record Vitals Button */}
            <button
              onClick={() => {
                setTempVitals(vitals);
                setShowVitalsModal(true);
              }}
              className="p-1.5 sm:px-2 sm:py-1 rounded bg-white dark:bg-slate-800 hover:bg-teal-50 dark:hover:bg-teal-950/60 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:text-teal-800 dark:hover:text-teal-300 text-[10px] sm:text-[11px] font-semibold flex items-center justify-center gap-1 transition-colors shadow-xs min-h-[32px] min-w-[32px]"
              title="Log / Update Patient Vital Signs"
            >
              <Edit3 className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400" />
              <span className="hidden sm:inline">Log Vitals</span>
            </button>

            {/* Collapse/Expand Vitals Toggle */}
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1.5 rounded text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800 transition-colors min-h-[32px] min-w-[32px] flex items-center justify-center"
              title={isCollapsed ? "Expand Triage Vitals Bar" : "Collapse Triage Vitals Bar"}
            >
              {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
            </button>

            {onClose && (
              <button
                onClick={onClose}
                className="p-1.5 rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-800 transition-colors min-h-[32px] min-w-[32px] flex items-center justify-center"
                title="Hide Patient Telemetry Banner"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Sub-row: Mobile Demographics & Allergy Chip */}
        <div className="flex items-center justify-between gap-2 mt-1.5 pt-1.5 border-t border-slate-200/60 dark:border-slate-800/80 text-[10px] sm:text-[11px]">
          <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400 font-medium truncate">
            <span className="md:hidden">{patientAge} • {patientGender}</span>
            <span className="md:hidden">•</span>
            <span className="md:hidden font-mono text-slate-700 dark:text-slate-300">{bloodType}</span>
            {isCollapsed && (
              <div className="flex items-center gap-1.5 font-mono text-[10px] text-slate-500 dark:text-slate-400 bg-white/70 dark:bg-slate-900/70 px-1.5 py-0.2 rounded border border-slate-200/70 dark:border-slate-800 truncate">
                <span>HR <strong>{vitals.heartRate}</strong></span>
                <span>•</span>
                <span>BP <strong>{vitals.bloodPressureSystolic}/{vitals.bloodPressureDiastolic}</strong></span>
                <span>•</span>
                <span>SpO₂ <strong>{vitals.oxygenSaturation}%</strong></span>
              </div>
            )}
          </div>

          {/* Critical Allergy Pill */}
          <div className="shrink-0 max-w-[140px] xs:max-w-[180px] sm:max-w-xs">
            {allergiesText.toLowerCase().includes('nkda') || allergiesText.toLowerCase().includes('none') ? (
              <span className="px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono text-[9px] sm:text-[10px] font-semibold border border-slate-300 dark:border-slate-700 block truncate">
                NKDA
              </span>
            ) : (
              <span className="px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-950/80 text-red-800 dark:text-red-300 font-mono text-[9px] sm:text-[10px] font-bold border border-red-300 dark:border-red-800 flex items-center gap-1 animate-pulse truncate" title={allergiesText}>
                <AlertTriangle className="w-2.5 h-2.5 sm:w-3 sm:h-3 text-red-700 dark:text-red-400 shrink-0" />
                <span className="truncate">{allergiesText}</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Triage Vital Signs Telemetry Strip - Responsive Grid on Mobile & Desktop */}
      {!isCollapsed && (
        <div className="px-3 sm:px-6 py-2.5 bg-white dark:bg-slate-900 border-b border-slate-200/70 dark:border-slate-800">
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 sm:gap-3 text-xs">
            {/* Heart Rate (Pulse) */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                isTachycardic || isBradycardic 
                  ? 'bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300' 
                  : 'bg-rose-50 dark:bg-rose-950/60 text-rose-700 dark:text-rose-400'
              }`}>
                <Heart className={`w-3 h-3 sm:w-3.5 sm:h-3.5 ${isTachycardic ? 'animate-ping' : ''}`} />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  HR
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className={`text-xs sm:text-sm font-mono font-bold ${
                    isTachycardic || isBradycardic ? 'text-amber-700 dark:text-amber-400' : 'text-slate-900 dark:text-slate-100'
                  }`}>
                    {vitals.heartRate}
                  </span>
                  <span className="text-[8px] sm:text-[9px] text-slate-400 font-mono">bpm</span>
                </div>
              </div>
            </div>

            {/* Blood Pressure (NIBP) */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                isHypertensive 
                  ? 'bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300' 
                  : 'bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-400'
              }`}>
                <Gauge className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  BP
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className={`text-xs sm:text-sm font-mono font-bold ${
                    isHypertensive ? 'text-amber-700 dark:text-amber-400' : 'text-slate-900 dark:text-slate-100'
                  }`}>
                    {vitals.bloodPressureSystolic}/{vitals.bloodPressureDiastolic}
                  </span>
                </div>
              </div>
            </div>

            {/* SpO2 Oxygen Saturation */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                isHypoxic 
                  ? 'bg-red-100 dark:bg-red-950/80 text-red-800 dark:text-red-300' 
                  : 'bg-teal-50 dark:bg-teal-950/60 text-teal-800 dark:text-teal-300'
              }`}>
                <Activity className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  SpO₂
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className={`text-xs sm:text-sm font-mono font-bold ${
                    isHypoxic ? 'text-red-700 dark:text-red-400 font-black' : 'text-slate-900 dark:text-slate-100'
                  }`}>
                    {vitals.oxygenSaturation}%
                  </span>
                </div>
              </div>
            </div>

            {/* Temperature */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                isFebrile 
                  ? 'bg-red-100 dark:bg-red-950/80 text-red-800 dark:text-red-300' 
                  : isHypothermic 
                  ? 'bg-blue-100 dark:bg-blue-950/70 text-blue-800 dark:text-blue-300' 
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
              }`}>
                <Thermometer className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  TEMP
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className={`text-xs sm:text-sm font-mono font-bold ${
                    isFebrile ? 'text-red-700 dark:text-red-400' : isHypothermic ? 'text-blue-700 dark:text-blue-400' : 'text-slate-900 dark:text-slate-100'
                  }`}>
                    {vitals.temperature}°F
                  </span>
                </div>
              </div>
            </div>

            {/* Respiratory Rate */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                isTachypneic 
                  ? 'bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300' 
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
              }`}>
                <Wind className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  RR
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className={`text-xs sm:text-sm font-mono font-bold ${
                    isTachypneic ? 'text-amber-700 dark:text-amber-400' : 'text-slate-900 dark:text-slate-100'
                  }`}>
                    {vitals.respiratoryRate}
                  </span>
                  <span className="text-[8px] sm:text-[9px] text-slate-400 font-mono">/m</span>
                </div>
              </div>
            </div>

            {/* Pain Scale */}
            <div className="p-1.5 sm:p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/80 flex items-center gap-1.5 sm:gap-2">
              <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center shrink-0 ${
                vitals.painLevel >= 7 
                  ? 'bg-red-100 dark:bg-red-950/80 text-red-800 dark:text-red-300' 
                  : vitals.painLevel >= 4 
                  ? 'bg-amber-100 dark:bg-amber-950/70 text-amber-800 dark:text-amber-300' 
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
              }`}>
                <AlertTriangle className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0">
                <span className="text-[9px] uppercase font-mono font-bold text-slate-400 dark:text-slate-500 block leading-none">
                  PAIN
                </span>
                <div className="flex items-baseline gap-0.5 mt-0.5">
                  <span className="text-xs sm:text-sm font-mono font-bold text-slate-900 dark:text-slate-100">
                    {vitals.painLevel}/10
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Action to attach vitals to consultation message */}
          <div className="flex items-center justify-end mt-2 pt-2 border-t border-slate-100 dark:border-slate-800">
            <button
              onClick={handleInjectVitalsSummary}
              className="w-full sm:w-auto px-3 py-1.5 rounded-lg bg-teal-50 dark:bg-teal-950/60 hover:bg-teal-100 dark:hover:bg-teal-900/60 text-teal-800 dark:text-teal-300 border border-teal-200 dark:border-teal-800 text-xs font-mono font-semibold flex items-center justify-center gap-1.5 transition-colors shadow-xs"
              title="Add current vitals into clinical intake message"
            >
              <FilePlus2 className="w-3.5 h-3.5 text-teal-700 dark:text-teal-400" />
              <span>Attach Vitals to Note</span>
            </button>
          </div>
        </div>
      )}

      {/* Modal: Interactive Vital Signs Entry Form */}
      {showVitalsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-slate-900/60 backdrop-blur-xs">
          <div className="w-full max-w-md max-h-[92vh] flex flex-col bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl overflow-hidden transition-colors">
            <div className="p-3.5 sm:p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50/90 dark:bg-slate-950/90 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-teal-700 dark:text-teal-400" />
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">Record Patient Vital Signs</h3>
              </div>
              <button
                onClick={() => setShowVitalsModal(false)}
                className="p-1.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg min-h-[36px] min-w-[36px] flex items-center justify-center"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSaveVitals} className="p-4 sm:p-5 space-y-4 overflow-y-auto">
              <div className="grid grid-cols-2 gap-3.5">
                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    Heart Rate (BPM)
                  </label>
                  <input
                    type="number"
                    min="30"
                    max="220"
                    value={tempVitals.heartRate}
                    onChange={(e) => setTempVitals({ ...tempVitals, heartRate: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    SpO₂ Saturation (%)
                  </label>
                  <input
                    type="number"
                    min="50"
                    max="100"
                    value={tempVitals.oxygenSaturation}
                    onChange={(e) => setTempVitals({ ...tempVitals, oxygenSaturation: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    Systolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    min="60"
                    max="250"
                    value={tempVitals.bloodPressureSystolic}
                    onChange={(e) => setTempVitals({ ...tempVitals, bloodPressureSystolic: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    Diastolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    min="40"
                    max="150"
                    value={tempVitals.bloodPressureDiastolic}
                    onChange={(e) => setTempVitals({ ...tempVitals, bloodPressureDiastolic: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    Temperature (°F)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="90"
                    max="108"
                    value={tempVitals.temperature}
                    onChange={(e) => setTempVitals({ ...tempVitals, temperature: parseFloat(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 block mb-1">
                    Resp Rate (/min)
                  </label>
                  <input
                    type="number"
                    min="6"
                    max="60"
                    value={tempVitals.respiratoryRate}
                    onChange={(e) => setTempVitals({ ...tempVitals, respiratoryRate: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] font-mono font-semibold text-slate-700 dark:text-slate-300 flex justify-between mb-1">
                  <span>Numeric Pain Rating Scale (0 to 10)</span>
                  <span className="font-bold text-teal-800 dark:text-teal-400">{tempVitals.painLevel}/10</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  value={tempVitals.painLevel}
                  onChange={(e) => setTempVitals({ ...tempVitals, painLevel: parseInt(e.target.value) || 0 })}
                  className="w-full accent-teal-700 dark:accent-teal-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                  <span>0 (No Pain)</span>
                  <span>5 (Moderate)</span>
                  <span>10 (Worst Possible)</span>
                </div>
              </div>

              {/* Quick Preset Buttons */}
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-2">
                <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">Presets:</span>
                <div className="flex gap-1.5">
                  <button
                    type="button"
                    onClick={() => setTempVitals({
                      heartRate: 72,
                      bloodPressureSystolic: 120,
                      bloodPressureDiastolic: 80,
                      oxygenSaturation: 99,
                      temperature: 98.6,
                      respiratoryRate: 16,
                      painLevel: 1,
                      lastRecorded: Date.now()
                    })}
                    className="px-2 py-0.8 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-[10px] font-mono text-slate-700 dark:text-slate-300"
                  >
                    Normative
                  </button>
                  <button
                    type="button"
                    onClick={() => setTempVitals({
                      heartRate: 112,
                      bloodPressureSystolic: 138,
                      bloodPressureDiastolic: 86,
                      oxygenSaturation: 96,
                      temperature: 101.4,
                      respiratoryRate: 22,
                      painLevel: 6,
                      lastRecorded: Date.now()
                    })}
                    className="px-2 py-0.8 rounded bg-amber-50 dark:bg-amber-950/60 hover:bg-amber-100 dark:hover:bg-amber-900/60 text-amber-800 dark:text-amber-300 text-[10px] font-mono border border-amber-200 dark:border-amber-800"
                  >
                    Febrile / Urgent
                  </button>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowVitalsModal(false)}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold transition-colors shadow-xs"
                >
                  Commit Vitals Record
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );

};
