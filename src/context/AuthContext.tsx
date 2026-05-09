import React, { createContext, useContext, useEffect, useState } from 'react';
import { onAuthStateChanged, User } from 'firebase/auth';
import { doc, getDoc, setDoc, onSnapshot } from 'firebase/firestore';
import { auth, db, handleFirestoreError, OperationType } from '../lib/firebase';

interface HealthProfile {
  age?: number;
  gender?: string;
  conditions?: string;
  allergies?: string;
  medications?: string;
  bloodType?: string;
}

interface UserData {
  uid: string;
  email: string | null;
  displayName: string | null;
  healthProfile?: HealthProfile;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  userData: UserData | null;
  loading: boolean;
  updateHealthProfile: (profile: HealthProfile) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user);
      if (user) {
        // Sync with Firestore
        const userRef = doc(db, 'users', user.uid);
        let userDoc;
        try {
          userDoc = await getDoc(userRef);
        } catch (e) {
          handleFirestoreError(e, OperationType.GET, 'users/' + user.uid);
        }
        
        if (userDoc && !userDoc.exists()) {
          const newData: UserData = {
            uid: user.uid,
            email: user.email,
            displayName: user.displayName,
            createdAt: new Date().toISOString(),
          };
          try {
            await setDoc(userRef, newData);
          } catch (e) {
            handleFirestoreError(e, OperationType.WRITE, 'users/' + user.uid);
          }
          setUserData(newData);
        } else if (userDoc) {
          setUserData(userDoc.data() as UserData);
        }

        // Real-time listener for user data updates
        const unmountListener = onSnapshot(userRef, (doc) => {
          if (doc.exists()) {
            setUserData(doc.data() as UserData);
          }
        }, (error) => {
          handleFirestoreError(error, OperationType.GET, 'users/' + user.uid);
        });
        
        setLoading(false);
        return () => unmountListener();
      } else {
        setUserData(null);
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  const updateHealthProfile = async (profile: HealthProfile) => {
    if (!user) return;
    const userRef = doc(db, 'users', user.uid);
    try {
      await setDoc(userRef, { healthProfile: profile }, { merge: true });
    } catch (e) {
      handleFirestoreError(e, OperationType.UPDATE, 'users/' + user.uid);
    }
  };

  return (
    <AuthContext.Provider value={{ user, userData, loading, updateHealthProfile }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
