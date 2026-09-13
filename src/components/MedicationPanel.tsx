import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Pill, Plus, Trash2, Clock, AlertCircle, X, ShieldCheck, FileText } from 'lucide-react';
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
  const [frequency, setFrequency] = useState('Daily (q.d.)');
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
    <div className="fixed inset-0 z-50 flex items-center justify-end bg-slate-900/40 backdrop-blur-xs p-0 sm:p-4">
      <motion.div 
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 50 }}
        className="w-full sm:max-w-md h-full sm:h-[90vh] bg-white sm:rounded-2xl border border-slate-200 shadow-2xl flex flex-col overflow-hidden"
      >
        {/* Clinical Pharmacy Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-50/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700 shadow-xs">
              <Pill className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-900 tracking-tight">Active Medications</h2>
                <span className="px-2 py-0.5 rounded-full bg-teal-100 text-teal-800 text-[10px] font-mono font-semibold">
                  eMAR
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium">
                Electronic Medication Administration Record • {medications.length} Prescribed
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <button 
              onClick={() => setShowAddForm(true)}
              className="px-3 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold flex items-center gap-1 shadow-xs transition-colors"
              title="Add Medication Order"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add Rx</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/50 transition-colors"
              title="Close Panel"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 custom-scrollbar bg-slate-50/40">
          {showAddForm && (
            <motion.form 
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              onSubmit={handleAddMedication}
              className="p-5 rounded-xl bg-white border border-teal-200 shadow-sm space-y-3.5"
            >
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <span className="text-xs font-bold text-teal-900 uppercase tracking-wider flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-teal-600" />
                  New Medication Entry
                </span>
                <button onClick={() => setShowAddForm(false)} type="button">
                  <X className="w-4 h-4 text-slate-400 hover:text-slate-600" />
                </button>
              </div>
              
              <div className="space-y-3">
                <div>
                  <label className="text-[11px] font-semibold text-slate-600 block mb-1">Medication / Generic Name</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g., Lisinopril, Metformin, Amoxicillin"
                    className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                    required
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-[11px] font-semibold text-slate-600 block mb-1">Dosage / Strength</label>
                    <input
                      type="text"
                      value={dosage}
                      onChange={(e) => setDosage(e.target.value)}
                      placeholder="e.g., 10 mg, 500 mg"
                      className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                      required
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-slate-600 block mb-1">Frequency (Sig)</label>
                    <select
                      value={frequency}
                      onChange={(e) => setFrequency(e.target.value)}
                      className="w-full bg-white border border-slate-200 rounded-lg py-2 px-3 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-600 focus:ring-1 focus:ring-teal-600 transition-all"
                    >
                      <option value="Daily (q.d.)">Daily (q.d.)</option>
                      <option value="Twice daily (b.i.d.)">Twice daily (b.i.d.)</option>
                      <option value="Three times daily (t.i.d.)">Three times daily (t.i.d.)</option>
                      <option value="Four times daily (q.i.d.)">Four times daily (q.i.d.)</option>
                      <option value="Every 8 hours (q8h)">Every 8 hours (q8h)</option>
                      <option value="Every 12 hours (q12h)">Every 12 hours (q12h)</option>
                      <option value="As needed (p.r.n.)">As needed (p.r.n.)</option>
                      <option value="Weekly (q.w.)">Weekly (q.w.)</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-600 block mb-1">Scheduled Administration Times</label>
                  <div className="space-y-2">
                    {times.map((t, idx) => (
                      <div key={idx} className="flex gap-2">
                        <input
                          type="time"
                          value={t}
                          onChange={(e) => updateTimeSlot(idx, e.target.value)}
                          className="flex-1 bg-white border border-slate-200 rounded-lg py-1.5 px-3 text-xs font-mono font-medium"
                        />
                        {times.length > 1 && (
                          <button 
                            type="button" 
                            onClick={() => removeTimeSlot(idx)}
                            className="px-2.5 rounded-lg bg-red-50 text-red-600 hover:bg-red-100 border border-red-200 transition-all text-xs"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    ))}
                    <button 
                      type="button"
                      onClick={addTimeSlot}
                      className="w-full py-1.5 rounded-lg border border-dashed border-slate-300 text-[11px] font-semibold text-slate-500 hover:border-teal-600 hover:text-teal-700 transition-all"
                    >
                      + Add Dose Time
                    </button>
                  </div>
                </div>

                <div className="pt-2 flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setShowAddForm(false)}
                    className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 rounded-lg bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold transition-colors shadow-xs"
                  >
                    Confirm Rx Order
                  </button>
                </div>
              </div>
            </motion.form>
          )}

          {medications.length === 0 ? (
            <div className="py-12 text-center space-y-3 bg-white rounded-xl border border-slate-200 p-6">
              <div className="w-12 h-12 rounded-xl bg-slate-100 border border-slate-200 flex items-center justify-center mx-auto text-slate-400">
                <Pill className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-bold text-slate-800">No Active Prescriptions Recorded</h3>
              <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                Add your current medications or ask RapidAid during clinical triage to automatically register medication reminders.
              </p>
              <button
                onClick={() => setShowAddForm(true)}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-teal-700 text-white text-xs font-semibold hover:bg-teal-800 transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Medication
              </button>
            </div>
          ) : (
            medications.map((med, idx) => (
              <motion.div
                key={med.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs hover:border-slate-300 transition-all group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-700 shrink-0 mt-0.5">
                      <Pill className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-sm font-bold text-slate-900">{med.name}</h4>
                        <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded font-semibold border border-slate-200">
                          {med.dosage}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-slate-600 font-medium">
                          Sig: {med.frequency}
                        </span>
                        <span className="text-slate-300">•</span>
                        <span className="text-[10px] font-mono text-slate-400">
                          Rx #{String(idx + 101).padStart(5, '0')}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button 
                    onClick={() => deleteMedication(userId, med.id)}
                    className="p-1.5 text-slate-400 hover:text-red-600 rounded-md hover:bg-red-50 transition-colors"
                    title="Discontinue Medication"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1.5 text-slate-500">
                    <Clock className="w-3.5 h-3.5 text-teal-600" />
                    <span className="text-[11px] font-medium">Schedule:</span>
                    <div className="flex flex-wrap gap-1">
                      {med.times.map((t, i) => (
                        <span key={i} className="text-[10px] font-mono font-semibold text-slate-700 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                  <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">
                    <ShieldCheck className="w-3 h-3" />
                    Active
                  </span>
                </div>
              </motion.div>
            ))
          )}
        </div>

        {/* Clinical Disclaimer Footer */}
        <div className="p-4 bg-slate-50 border-t border-slate-200">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <p className="text-[11px] text-slate-600 leading-relaxed font-medium">
              <strong className="text-slate-800">Pharmacy Protocol:</strong> Always cross-reference your physical prescription label with attending clinician or pharmacist instructions before altering dosages.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
