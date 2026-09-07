import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Pill, Plus, Trash2, Clock, Calendar, AlertCircle, X, CheckSquare } from 'lucide-react';
import { Medication, addMedication, deleteMedication } from '../lib/medications';

interface MedicationPanelProps {
  userId: string;
  medications: Medication[];
  onClose: () => void;
}

export const MedicationPanel: React.FC<MedicationPanelProps> = ({ userId, medications, onClose }) => {
  const [showAddForm, setShowAddForm] = useState(false);
  const [name, setName] = useState('');
  const [dosage, setDosage] = useState('');
  const [frequency, setFrequency] = useState('Daily');
  const [time, setTime] = useState('08:00');
  const [times, setTimes] = useState<string[]>(['08:00']);

  const handleAddMedication = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !dosage) return;

    await addMedication(userId, {
      userId,
      name,
      dosage,
      frequency,
      times
    });

    setName('');
    setDosage('');
    setShowAddForm(false);
  };

  const addTimeSlot = () => setTimes([...times, '12:00']);
  const removeTimeSlot = (index: number) => setTimes(times.filter((_, i) => i !== index));
  const updateTimeSlot = (index: number, val: string) => {
    const newTimes = [...times];
    newTimes[index] = val;
    setTimes(newTimes);
  };

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="p-6 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center text-blue-600 border border-blue-100">
            <Pill className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">Medications</h2>
            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">Active Prescriptions</p>
          </div>
        </div>
        <button 
          onClick={() => setShowAddForm(true)}
          className="p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition-all shadow-lg shadow-blue-600/20"
        >
          <Plus className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
        {showAddForm && (
          <motion.form 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            onSubmit={handleAddMedication}
            className="p-6 rounded-[2rem] bg-slate-50 border border-slate-200 space-y-4"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-black text-slate-400 uppercase tracking-widest">New Entry</h3>
              <button onClick={() => setShowAddForm(false)} type="button">
                <X className="w-4 h-4 text-slate-400 hover:text-red-500" />
              </button>
            </div>
            
            <div className="space-y-3">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Medication Name (e.g. Lisinopril)"
                className="w-full bg-white border border-slate-200 rounded-xl py-3 px-4 text-sm font-bold focus:outline-none focus:border-blue-500 transition-all"
                required
              />
              <input
                type="text"
                value={dosage}
                onChange={(e) => setDosage(e.target.value)}
                placeholder="Dosage (e.g. 10mg)"
                className="w-full bg-white border border-slate-200 rounded-xl py-3 px-4 text-sm font-bold focus:outline-none focus:border-blue-500 transition-all"
                required
              />
              
              <div className="space-y-2">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Frequency</label>
                <select
                  value={frequency}
                  onChange={(e) => setFrequency(e.target.value)}
                  className="w-full bg-white border border-slate-200 rounded-xl py-3 px-4 text-sm font-bold focus:outline-none focus:border-blue-500 transition-all appearance-none"
                >
                  <option value="Daily">Daily</option>
                  <option value="Twice a day">Twice a day</option>
                  <option value="Weekly">Weekly</option>
                  <option value="As needed">As needed</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-1">Reminder Times</label>
                <div className="space-y-2">
                  {times.map((t, idx) => (
                    <div key={idx} className="flex gap-2">
                      <input
                        type="time"
                        value={t}
                        onChange={(e) => updateTimeSlot(idx, e.target.value)}
                        className="flex-1 bg-white border border-slate-200 rounded-xl py-2 px-4 text-sm font-bold"
                      />
                      {times.length > 1 && (
                        <button 
                          type="button" 
                          onClick={() => removeTimeSlot(idx)}
                          className="px-3 rounded-xl bg-red-50 text-red-500 hover:bg-red-100 border border-red-100 transition-all"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                  <button 
                    type="button"
                    onClick={addTimeSlot}
                    className="w-full py-2 rounded-xl border border-dashed border-slate-300 text-[10px] font-bold text-slate-400 hover:border-blue-400 hover:text-blue-500 transition-all"
                  >
                    + Add Time Slot
                  </button>
                </div>
              </div>

              <button 
                type="submit"
                className="w-full py-3 bg-slate-900 text-white rounded-xl text-xs font-black uppercase tracking-[0.2em] shadow-lg shadow-slate-900/10 hover:bg-slate-800 transition-all mt-4"
              >
                Register Medication
              </button>
            </div>
          </motion.form>
        )}

        {medications.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
            <div className="w-16 h-16 rounded-[1.5rem] bg-slate-50 flex items-center justify-center text-slate-200">
              <CheckSquare className="w-8 h-8" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-bold text-slate-400">No active prescriptions</p>
              <p className="text-xs text-slate-400 max-w-[180px]">Add your medications to receive personalized reminders and AI analysis.</p>
            </div>
          </div>
        ) : (
          medications.map(med => (
            <motion.div 
              layout
              key={med.id}
              className="p-5 rounded-[2rem] bg-white border border-slate-100 shadow-sm hover:border-blue-200 transition-all group relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-1 h-full bg-blue-600" />
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <h4 className="font-bold text-slate-900">{med.name}</h4>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-black text-blue-600 uppercase tracking-widest bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
                      {med.dosage}
                    </span>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                      {med.frequency}
                    </span>
                  </div>
                </div>
                <button 
                  onClick={() => deleteMedication(userId, med.id)}
                  className="p-2 text-slate-200 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              <div className="mt-4 pt-4 border-t border-slate-50 flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5 text-slate-300" />
                  <div className="flex gap-1.5">
                    {med.times.map((t, i) => (
                      <span key={i} className="text-[10px] font-bold text-slate-500 bg-slate-50 px-2 py-0.5 rounded-lg border border-slate-100">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          ))
        )}
      </div>

      <div className="p-6 bg-slate-50 border-t border-slate-100">
        <div className="flex items-start gap-4">
          <AlertCircle className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
          <p className="text-[10px] text-slate-500 leading-relaxed font-medium">
            AI reminders are approximate. Always maintain a physical log and follow your doctor's exact prescription schedule.
          </p>
        </div>
      </div>
    </div>
  );
};
