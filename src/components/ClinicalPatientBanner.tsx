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
  Clock
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
}

export const ClinicalPatientBanner: React.FC<ClinicalPatientBannerProps> = ({
  vitals,
  onUpdateVitals,
  acuity,
  onUpdateAcuity,
  patientProfile,
  user,
  onOpenProfile,
  onInjectVitals
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
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
    <div className="bg-white border-b border-slate-200 text-slate-900 text-xs shadow-xs relative z-15 select-none">
      {/* Top Telemetry Strip: Hospital Ward, Patient Demographics & Acuity */}
      <div className="px-4 sm:px-6 py-2 bg-slate-100/90 border-b border-slate-200/80 flex flex-wrap items-center justify-between gap-y-2">
        {/* Patient Demographic Identity */}
        <div className="flex items-center flex-wrap gap-2.5 sm:gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 animate-pulse" />
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-slate-900 uppercase tracking-tight text-xs">
                {patientName}
              </span>
              <span className="px-1.5 py-0.2 rounded bg-slate-200 text-slate-700 font-mono text-[10px] font-semibold">
                MRN #842-198
              </span>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-2 text-[11px] text-slate-600 font-medium">
            <span>{patientAge}</span>
            <span>•</span>
            <span className="capitalize">{patientGender}</span>
            <span>•</span>
            <span className="font-mono text-slate-700">ABO: {bloodType}</span>
            <span>•</span>
            <span className="px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 font-mono text-[9px] font-bold">
              FULL CODE
            </span>
          </div>

          {/* Critical Allergy Pill */}
          <div className="flex items-center">
            {allergiesText.toLowerCase().includes('nkda') || allergiesText.toLowerCase().includes('none') ? (
              <span className="px-2 py-0.5 rounded bg-slate-200/80 text-slate-700 font-mono text-[10px] font-semibold border border-slate-300">
                ALLERGIES: NKDA
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded bg-red-100 text-red-800 font-mono text-[10px] font-bold border border-red-300 flex items-center gap-1 animate-pulse">
                <AlertTriangle className="w-3 h-3 text-red-700" />
                <span>ALLERGIES: {allergiesText}</span>
              </span>
            )}
          </div>
        </div>

        {/* Right Station Controls: Acuity Badge, Time & Vitals Editor */}
        <div className="flex items-center gap-2 ml-auto">
          {/* Acuity (ESI) Dropdown */}
          <div className="flex items-center gap-1">
            <label className="text-[10px] uppercase font-mono font-bold text-slate-500 hidden md:inline">
              ACUITY:
            </label>
            <select
              value={acuity}
              onChange={(e) => onUpdateAcuity(e.target.value as AcuityLevel)}
              className={`px-2 py-0.8 rounded text-[10px] font-mono font-bold uppercase tracking-wider border cursor-pointer ${getAcuityColor(acuity)}`}
              title="Emergency Severity Index Triage Level"
            >
              <option value="ESI-1">ESI-1: Resuscitation (Immediate)</option>
              <option value="ESI-2">ESI-2: Emergent (&lt;15m)</option>
              <option value="ESI-3">ESI-3: Urgent (&lt;30m)</option>
              <option value="ESI-4">ESI-4: Less Urgent (&lt;60m)</option>
              <option value="ESI-5">ESI-5: Non-Urgent (&lt;120m)</option>
            </select>
          </div>

          {/* Quick Record Vitals Button */}
          <button
            onClick={() => {
              setTempVitals(vitals);
              setShowVitalsModal(true);
            }}
            className="px-2 py-1 rounded bg-white hover:bg-teal-50 border border-slate-300 text-slate-700 hover:text-teal-800 text-[11px] font-semibold flex items-center gap-1 transition-colors shadow-xs"
            title="Log / Update Patient Vital Signs"
          >
            <Edit3 className="w-3 h-3 text-teal-700" />
            <span className="hidden sm:inline">Log Vitals</span>
          </button>

          {/* Collapse/Expand Vitals Toggle */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200/60 transition-colors"
            title={isCollapsed ? "Expand Triage Vitals Bar" : "Collapse Triage Vitals Bar"}
          >
            {isCollapsed ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Triage Vital Signs Telemetry Strip */}
      {!isCollapsed && (
        <div className="px-4 sm:px-6 py-2.5 bg-white flex flex-wrap items-center justify-between gap-3 overflow-x-auto">
          <div className="flex items-center gap-3 sm:gap-6 text-xs flex-nowrap">
            {/* Heart Rate (Pulse) */}
            <div className="flex items-center gap-2">
              <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
                isTachycardic || isBradycardic ? 'bg-amber-100 text-amber-800' : 'bg-rose-50 text-rose-700'
              }`}>
                <Heart className={`w-3.5 h-3.5 ${isTachycardic ? 'animate-ping' : ''}`} />
              </div>
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  HR (PULSE)
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className={`text-sm font-mono font-bold ${
                    isTachycardic || isBradycardic ? 'text-amber-700' : 'text-slate-900'
                  }`}>
                    {vitals.heartRate}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">bpm</span>
                </div>
              </div>
            </div>

            {/* Blood Pressure (NIBP) */}
            <div className="flex items-center gap-2 border-l border-slate-200 pl-3 sm:pl-5">
              <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
                isHypertensive ? 'bg-amber-100 text-amber-800' : 'bg-blue-50 text-blue-700'
              }`}>
                <Gauge className="w-3.5 h-3.5" />
              </div>
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  BP (NIBP)
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className={`text-sm font-mono font-bold ${
                    isHypertensive ? 'text-amber-700' : 'text-slate-900'
                  }`}>
                    {vitals.bloodPressureSystolic}/{vitals.bloodPressureDiastolic}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">mmHg</span>
                </div>
              </div>
            </div>

            {/* SpO2 Oxygen Saturation */}
            <div className="flex items-center gap-2 border-l border-slate-200 pl-3 sm:pl-5">
              <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
                isHypoxic ? 'bg-red-100 text-red-800' : 'bg-teal-50 text-teal-800'
              }`}>
                <Activity className="w-3.5 h-3.5" />
              </div>
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  SpO₂
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className={`text-sm font-mono font-bold ${
                    isHypoxic ? 'text-red-700 font-black' : 'text-slate-900'
                  }`}>
                    {vitals.oxygenSaturation}%
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">RA</span>
                </div>
              </div>
            </div>

            {/* Temperature */}
            <div className="flex items-center gap-2 border-l border-slate-200 pl-3 sm:pl-5">
              <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
                isFebrile ? 'bg-red-100 text-red-800' : isHypothermic ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-700'
              }`}>
                <Thermometer className="w-3.5 h-3.5" />
              </div>
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  TEMP (ORAL)
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className={`text-sm font-mono font-bold ${
                    isFebrile ? 'text-red-700' : 'text-slate-900'
                  }`}>
                    {vitals.temperature}°F
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    ({((vitals.temperature - 32) * 5 / 9).toFixed(1)}°C)
                  </span>
                </div>
              </div>
            </div>

            {/* Respiratory Rate */}
            <div className="hidden lg:flex items-center gap-2 border-l border-slate-200 pl-3 sm:pl-5">
              <div className={`w-7 h-7 rounded-md flex items-center justify-center ${
                isTachypneic ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'
              }`}>
                <Wind className="w-3.5 h-3.5" />
              </div>
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  RESP RATE
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className={`text-sm font-mono font-bold ${
                    isTachypneic ? 'text-amber-700' : 'text-slate-900'
                  }`}>
                    {vitals.respiratoryRate}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">/min</span>
                </div>
              </div>
            </div>

            {/* Pain Scale */}
            <div className="hidden sm:flex items-center gap-2 border-l border-slate-200 pl-3 sm:pl-5">
              <div>
                <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block leading-none">
                  PAIN (0-10)
                </span>
                <div className="flex items-baseline gap-1 mt-0.5">
                  <span className="text-sm font-mono font-bold text-slate-900">
                    {vitals.painLevel}/10
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {vitals.painLevel === 0 ? 'None' : vitals.painLevel <= 3 ? 'Mild' : vitals.painLevel <= 6 ? 'Mod' : 'Severe'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Action to send vitals to chat */}
          <div className="flex items-center gap-2 ml-auto shrink-0">
            <button
              onClick={handleInjectVitalsSummary}
              className="px-2.5 py-1 rounded bg-teal-50 hover:bg-teal-100 text-teal-800 border border-teal-200 text-[11px] font-mono font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
              title="Add current vitals into clinical intake message"
            >
              <FilePlus2 className="w-3 h-3 text-teal-700" />
              <span>Attach Vitals to Note</span>
            </button>
          </div>
        </div>
      )}

      {/* Modal: Interactive Vital Signs Entry Form */}
      {showVitalsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs">
          <div className="w-full max-w-md bg-white rounded-xl border border-slate-200 shadow-2xl overflow-hidden">
            <div className="p-4 border-b border-slate-200 bg-slate-50/90 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-teal-700" />
                <h3 className="text-sm font-bold text-slate-900">Record Patient Vital Signs</h3>
              </div>
              <button
                onClick={() => setShowVitalsModal(false)}
                className="p-1 text-slate-400 hover:text-slate-700 rounded-md"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSaveVitals} className="p-5 space-y-4">
              <div className="grid grid-cols-2 gap-3.5">
                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    Heart Rate (BPM)
                  </label>
                  <input
                    type="number"
                    min="30"
                    max="220"
                    value={tempVitals.heartRate}
                    onChange={(e) => setTempVitals({ ...tempVitals, heartRate: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    SpO₂ Saturation (%)
                  </label>
                  <input
                    type="number"
                    min="50"
                    max="100"
                    value={tempVitals.oxygenSaturation}
                    onChange={(e) => setTempVitals({ ...tempVitals, oxygenSaturation: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    Systolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    min="60"
                    max="250"
                    value={tempVitals.bloodPressureSystolic}
                    onChange={(e) => setTempVitals({ ...tempVitals, bloodPressureSystolic: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    Diastolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    min="40"
                    max="150"
                    value={tempVitals.bloodPressureDiastolic}
                    onChange={(e) => setTempVitals({ ...tempVitals, bloodPressureDiastolic: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    Temperature (°F)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="90"
                    max="108"
                    value={tempVitals.temperature}
                    onChange={(e) => setTempVitals({ ...tempVitals, temperature: parseFloat(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>

                <div>
                  <label className="text-[11px] font-mono font-semibold text-slate-700 block mb-1">
                    Resp Rate (/min)
                  </label>
                  <input
                    type="number"
                    min="6"
                    max="60"
                    value={tempVitals.respiratoryRate}
                    onChange={(e) => setTempVitals({ ...tempVitals, respiratoryRate: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-1.5 rounded-lg border border-slate-300 font-mono text-xs focus:ring-2 focus:ring-teal-600 focus:border-teal-600 outline-none"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] font-mono font-semibold text-slate-700 flex justify-between mb-1">
                  <span>Numeric Pain Rating Scale (0 to 10)</span>
                  <span className="font-bold text-teal-800">{tempVitals.painLevel}/10</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  value={tempVitals.painLevel}
                  onChange={(e) => setTempVitals({ ...tempVitals, painLevel: parseInt(e.target.value) || 0 })}
                  className="w-full accent-teal-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0 (No Pain)</span>
                  <span>5 (Moderate)</span>
                  <span>10 (Worst Possible)</span>
                </div>
              </div>

              {/* Quick Preset Buttons */}
              <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
                <span className="text-[10px] font-mono text-slate-400">Presets:</span>
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
                    className="px-2 py-0.8 rounded bg-slate-100 hover:bg-slate-200 text-[10px] font-mono text-slate-700"
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
                    className="px-2 py-0.8 rounded bg-amber-50 hover:bg-amber-100 text-amber-800 text-[10px] font-mono border border-amber-200"
                  >
                    Febrile / Urgent
                  </button>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-200 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowVitalsModal(false)}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition-colors"
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
