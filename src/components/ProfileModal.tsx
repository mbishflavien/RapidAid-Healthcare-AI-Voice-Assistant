import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, User, Activity, ShieldAlert, Droplets, Pill, Calendar, HeartPulse, ShieldCheck, CheckCircle2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface ProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ProfileModal: React.FC<ProfileModalProps> = ({ isOpen, onClose }) => {
  const { userData, updateHealthProfile } = useAuth();
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<string>('');
  const [conditions, setConditions] = useState<string>('');
  const [allergies, setAllergies] = useState<string>('');
  const [medications, setMedications] = useState<string>('');
  const [bloodType, setBloodType] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (userData?.healthProfile) {
      setAge(userData.healthProfile.age?.toString() || '');
      setGender(userData.healthProfile.gender || '');
      setConditions(userData.healthProfile.conditions || '');
      setAllergies(userData.healthProfile.allergies || '');
      setMedications(userData.healthProfile.medications || '');
      setBloodType(userData.healthProfile.bloodType || '');
    }
  }, [userData]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await updateHealthProfile({
        age: age ? parseInt(age) : undefined,
        gender,
        conditions,
        allergies,
        medications,
        bloodType
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs">
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            className="w-full max-w-xl bg-white rounded-2xl border border-slate-200 shadow-2xl flex flex-col overflow-hidden max-h-[92vh]"
          >
            {/* Clinical Modal Header */}
            <div className="p-5 border-b border-slate-200 bg-slate-50/80 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700 shadow-xs">
                  <User className="w-5 h-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-base font-bold text-slate-900 tracking-tight">Patient Health Record (EHR)</h2>
                    <span className="px-2 py-0.5 rounded-full bg-slate-200 text-slate-700 text-[10px] font-mono font-semibold">
                      MRN-CONFIDENTIAL
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 font-medium">
                    Clinical demographics, allergy profile, and baseline medical history
                  </p>
                </div>
              </div>
              <button 
                onClick={onClose}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/50 transition-colors"
                title="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Form Fields */}
            <form onSubmit={handleSubmit} className="p-6 space-y-4 overflow-y-auto custom-scrollbar">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
                <div>
                  <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1 mb-1">
                    <Calendar className="w-3.5 h-3.5 text-teal-600" />
                    Age (Years)
                  </label>
                  <input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    placeholder="e.g. 42"
                    min="1"
                    max="125"
                    className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-semibold text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                  />
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1 mb-1">
                    <Activity className="w-3.5 h-3.5 text-teal-600" />
                    Biological Sex
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-semibold text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                  >
                    <option value="">Select</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other / Intersex</option>
                  </select>
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1 mb-1">
                    <Droplets className="w-3.5 h-3.5 text-rose-600" />
                    Blood Group
                  </label>
                  <input
                    type="text"
                    value={bloodType}
                    onChange={(e) => setBloodType(e.target.value)}
                    placeholder="e.g. O+, A-"
                    className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-semibold text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1.5 mb-1">
                  <HeartPulse className="w-3.5 h-3.5 text-teal-600" />
                  Pre-existing Chronic Conditions
                </label>
                <textarea
                  value={conditions}
                  onChange={(e) => setConditions(e.target.value)}
                  placeholder="e.g., Hypertension, Type 2 Diabetes, Asthma, Previous Myocardial Infarction..."
                  rows={2}
                  className="w-full bg-white border border-slate-200 rounded-lg p-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all resize-none leading-relaxed"
                />
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1.5 mb-1 text-rose-700">
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
                  Verified Drug & Environmental Allergies (Critical)
                </label>
                <textarea
                  value={allergies}
                  onChange={(e) => setAllergies(e.target.value)}
                  placeholder="e.g., Penicillin (Anaphylaxis), Sulfa drugs, Peanuts, Latex..."
                  rows={2}
                  className="w-full bg-white border border-rose-200 rounded-lg p-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-rose-600 focus:ring-1 focus:ring-rose-600 transition-all resize-none leading-relaxed"
                />
              </div>

              <div>
                <label className="text-[11px] font-semibold text-slate-700 flex items-center gap-1.5 mb-1">
                  <Pill className="w-3.5 h-3.5 text-teal-600" />
                  Current Regimen / Prescriptions
                </label>
                <textarea
                  value={medications}
                  onChange={(e) => setMedications(e.target.value)}
                  placeholder="e.g., Lisinopril 10mg q.d., Atorvastatin 20mg q.h.s..."
                  rows={2}
                  className="w-full bg-white border border-slate-200 rounded-lg p-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all resize-none leading-relaxed"
                />
              </div>

              {/* Security & Action Bar */}
              <div className="pt-3 border-t border-slate-200 flex items-center justify-between">
                <div className="flex items-center gap-2 text-slate-500">
                  <ShieldCheck className="w-4 h-4 text-teal-600" />
                  <span className="text-[11px] font-medium">HIPAA/AES-256 Protected</span>
                </div>
                <div className="flex items-center gap-2">
                  <AnimatePresence>
                    {success && (
                      <motion.div 
                        initial={{ opacity: 0, x: 6 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0 }}
                        className="flex items-center gap-1 text-emerald-600 text-xs font-semibold mr-2"
                      >
                        <CheckCircle2 className="w-4 h-4" />
                        <span>Chart Updated</span>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <button
                    type="button"
                    onClick={onClose}
                    className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-4 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white font-semibold text-xs transition-colors shadow-xs disabled:opacity-50"
                  >
                    {loading ? 'Saving Chart...' : 'Save Patient Chart'}
                  </button>
                </div>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
