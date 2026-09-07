import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, User, Activity, ShieldAlert, Droplets, Pill, Calendar, HeartPulse } from 'lucide-react';
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
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-[200]"
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-2xl z-[201] p-1"
          >
            <div className="bg-white rounded-[3rem] border border-slate-200/60 shadow-2xl p-10 relative flex flex-col max-h-[90vh] overflow-hidden">
              <button 
                onClick={onClose}
                className="absolute top-8 right-8 p-3 rounded-2xl bg-slate-50 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all border border-transparent hover:border-slate-200"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-5 mb-10">
                <div className="w-16 h-16 rounded-[1.5rem] bg-slate-900 text-blue-400 flex items-center justify-center border border-slate-800 shadow-xl">
                  <User className="w-8 h-8" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-slate-900 tracking-tight leading-tight">Patient Identity Protocol</h2>
                  <p className="text-[11px] text-slate-400 font-black uppercase tracking-[0.2em] mt-1">Refine AI Diagnostic Accuracy</p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-8 overflow-y-auto pr-4 custom-scrollbar pb-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-3">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Age Profile</label>
                    <div className="relative group">
                      <Calendar className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                      <input
                        type="number"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        placeholder="Years"
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-none"
                      />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Gender ID</label>
                    <div className="relative group">
                      <Activity className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                      <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all appearance-none shadow-none cursor-pointer"
                      >
                        <option value="">Select ID</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Blood Delta</label>
                    <div className="relative group">
                      <Droplets className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                      <input
                        type="text"
                        value={bloodType}
                        onChange={(e) => setBloodType(e.target.value)}
                        placeholder="Type"
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-4 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-none"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Clinical History</label>
                  <div className="relative group">
                    <HeartPulse className="absolute left-5 top-5 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                    <textarea
                      value={conditions}
                      onChange={(e) => setConditions(e.target.value)}
                      placeholder="Chronic conditions, surgeries, or underlying diagnostics..."
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-[1.5rem] py-5 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-none"
                    />
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Immune Sensitivity (Allergies)</label>
                  <div className="relative group">
                    <ShieldAlert className="absolute left-5 top-5 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                    <textarea
                      value={allergies}
                      onChange={(e) => setAllergies(e.target.value)}
                      placeholder="Verified drug, environmental, or nutritional allergies..."
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-[1.5rem] py-5 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-none"
                    />
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] ml-4">Pharmaceutical Regime</label>
                  <div className="relative group">
                    <Pill className="absolute left-5 top-5 w-4 h-4 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                    <textarea
                      value={medications}
                      onChange={(e) => setMedications(e.target.value)}
                      placeholder="Active prescriptions and dosage schedules..."
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-[1.5rem] py-5 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-none"
                    />
                  </div>
                </div>

                <div className="pt-6 flex items-center justify-between border-t border-slate-100">
                  <div className="flex flex-col gap-1 max-w-[50%]">
                    <span className="text-[10px] font-black text-slate-900 uppercase tracking-widest leading-none">Security Encryption</span>
                    <p className="text-[10px] text-slate-400 font-medium leading-relaxed">
                      Encrypted end-to-end. Stored in your private medical partition.
                    </p>
                  </div>
                  <div className="flex items-center gap-6">
                    <AnimatePresence>
                      {success && (
                        <motion.div 
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0 }}
                          className="flex items-center gap-2 text-green-600"
                        >
                          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                          <span className="text-[10px] font-black uppercase tracking-widest">Profile Synced</span>
                        </motion.div>
                      )}
                    </AnimatePresence>
                    <button
                      type="submit"
                      disabled={loading}
                      className="bg-slate-900 text-white font-black text-xs uppercase tracking-[0.2em] px-10 py-5 rounded-[1.75rem] shadow-xl shadow-slate-900/10 hover:bg-slate-800 active:scale-95 transition-all disabled:opacity-50 relative overflow-hidden group/btn"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover/btn:translate-x-full transition-transform duration-1000" />
                      {loading ? 'Syncing...' : 'Update Protocol'}
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
