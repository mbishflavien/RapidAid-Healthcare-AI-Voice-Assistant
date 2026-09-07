import React, { useState } from 'react';
import { signInWithPopup, signInWithEmailAndPassword, createUserWithEmailAndPassword } from 'firebase/auth';
import { auth, googleProvider } from '../lib/firebase';
import { motion, AnimatePresence } from 'motion/react';
import { X, Mail, Lock, User, LogIn, Github, Activity, AlertCircle } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
      onClose();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
      onClose();
    } catch (err: any) {
      setError(err.message);
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
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-md z-[201] p-1"
          >
            <div className="bg-white rounded-[3rem] overflow-hidden border border-slate-200/60 shadow-2xl p-10 relative">
              <button 
                onClick={onClose}
                className="absolute top-8 right-8 p-3 rounded-2xl bg-slate-50 text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all border border-transparent hover:border-slate-200"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex flex-col items-center mb-10">
                <div className="w-20 h-20 rounded-[2rem] bg-slate-900 flex items-center justify-center shadow-2xl shadow-slate-900/20 mb-6 relative group">
                  <div className="absolute inset-0 bg-blue-500/20 blur-2xl rounded-full scale-0 group-hover:scale-100 transition-transform duration-500" />
                  <Activity className="text-blue-400 w-10 h-10 relative z-10" />
                </div>
                <h2 className="text-3xl font-bold text-slate-900 tracking-tight text-center">
                  {isLogin ? 'Neural Access' : 'Create Profile'}
                </h2>
                <p className="text-[11px] text-slate-400 font-black uppercase tracking-[0.2em] mt-2 text-center">
                  {isLogin ? 'Secure Medical Authentication' : 'Initializing Neural Health Link'}
                </p>
              </div>

              <form onSubmit={handleAuth} className="space-y-5">
                <div className="space-y-2">
                  <div className="relative group">
                    <Mail className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="Email Address"
                      className="w-full bg-slate-50 border border-slate-200 rounded-[1.5rem] py-5 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-none"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="relative group">
                    <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-300 group-focus-within:text-blue-500 transition-colors" />
                    <input
                      type="password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="Security Token / Password"
                      className="w-full bg-slate-50 border border-slate-200 rounded-[1.5rem] py-5 pl-14 pr-5 text-slate-900 text-sm font-bold focus:outline-none focus:border-blue-500 focus:bg-white transition-all shadow-none"
                      required
                    />
                  </div>
                </div>

                {error && (
                  <div className="p-4 rounded-2xl bg-red-50 border border-red-100 text-red-500 text-xs font-bold flex items-center gap-3">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-slate-900 text-white font-black text-xs uppercase tracking-[0.2em] py-5 rounded-[1.75rem] shadow-xl shadow-slate-900/10 hover:bg-slate-800 transition-all disabled:opacity-50 relative overflow-hidden group/btn"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover/btn:translate-x-full transition-transform duration-1000" />
                  {loading ? 'Authenticating...' : isLogin ? 'Initialize Access' : 'Create Protocol'}
                </button>
              </form>

              <div className="mt-10 flex flex-col items-center gap-6">
                <div className="flex items-center gap-4 w-full">
                  <div className="h-[1px] bg-slate-100 flex-1" />
                  <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest leading-none">Security Providers</span>
                  <div className="h-[1px] bg-slate-100 flex-1" />
                </div>

                <button 
                  onClick={handleGoogleSignIn}
                  className="w-full py-5 rounded-2xl bg-white border border-slate-200 flex items-center justify-center gap-4 text-slate-700 text-xs font-black uppercase tracking-widest hover:bg-slate-50 hover:border-slate-300 transition-all shadow-sm"
                >
                  <LogIn className="w-5 h-5 text-blue-500" />
                  Google Identity
                </button>

                <button
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-[11px] text-slate-400 hover:text-blue-600 transition-colors font-black uppercase tracking-widest"
                >
                  {isLogin ? "Request New Protocol (Sign Up)" : "Access Existing Protocol (Log In)"}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
