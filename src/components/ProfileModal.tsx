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
            <div className="bg-white rounded-[2.5rem] border border-slate-200 shadow-2xl p-8 relative flex flex-col max-h-[90vh]">
              <button 
                onClick={onClose}
                className="absolute top-6 right-6 p-2 rounded-xl bg-slate-50 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-4 mb-8">
                <div className="w-12 h-12 rounded-2xl bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100">
                  <User className="w-6 h-6" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight">Health Profile</h2>
                  <p className="text-xs text-slate-400 font-medium">Tailor AI suggestions to your medical background</p>
                </div>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6 overflow-y-auto pr-2 custom-scrollbar">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Age</label>
                    <div className="relative">
                      <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                      <input
                        type="number"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        placeholder="e.g. 35"
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-inner"
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Gender</label>
                    <div className="relative">
                      <Activity className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                      <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all appearance-none shadow-inner"
                      >
                        <option value="">Select</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Blood Type</label>
                    <div className="relative">
                      <Droplets className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-300" />
                      <input
                        type="text"
                        value={bloodType}
                        onChange={(e) => setBloodType(e.target.value)}
                        placeholder="e.g. O+"
                        className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-inner"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Pre-existing Conditions</label>
                  <div className="relative">
                    <HeartPulse className="absolute left-4 top-4 w-4 h-4 text-slate-300" />
                    <textarea
                      value={conditions}
                      onChange={(e) => setConditions(e.target.value)}
                      placeholder="List any chronic conditions (e.g., Hypertension, Diabetes)"
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-inner"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Allergies</label>
                  <div className="relative">
                    <ShieldAlert className="absolute left-4 top-4 w-4 h-4 text-slate-300" />
                    <textarea
                      value={allergies}
                      onChange={(e) => setAllergies(e.target.value)}
                      placeholder="List any known allergies"
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-inner"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] ml-2">Current Medications</label>
                  <div className="relative">
                    <Pill className="absolute left-4 top-4 w-4 h-4 text-slate-300" />
                    <textarea
                      value={medications}
                      onChange={(e) => setMedications(e.target.value)}
                      placeholder="List any medications currently taken"
                      rows={2}
                      className="w-full bg-slate-50 border border-slate-200 rounded-2xl py-3.5 pl-12 pr-4 text-slate-900 text-sm focus:outline-none focus:border-blue-500 focus:bg-white transition-all resize-none shadow-inner"
                    />
                  </div>
                </div>

                <div className="pt-4 flex items-center justify-between">
                  <p className="text-[10px] text-slate-300 italic max-w-[60%]">
                    Your information is stored securely and used only to personalize your health consultations.
                  </p>
                  <div className="flex items-center gap-4">
                    <AnimatePresence>
                      {success && (
                        <motion.span 
                          initial={{ opacity: 0, x: 10 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0 }}
                          className="text-blue-600 text-xs font-bold uppercase tracking-widest"
                        >
                          Saved!
                        </motion.span>
                      )}
                    </AnimatePresence>
                    <button
                      type="submit"
                      disabled={loading}
                      className="bg-gradient-to-br from-blue-600 to-blue-500 text-white font-bold px-8 py-3.5 rounded-2xl shadow-lg shadow-blue-500/20 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50"
                    >
                      {loading ? 'Saving...' : 'Update Profile'}
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
