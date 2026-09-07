import { collection, addDoc, query, where, orderBy, onSnapshot, doc, updateDoc, deleteDoc, serverTimestamp } from 'firebase/firestore';
import { db, handleFirestoreError, OperationType } from './firebase';

export interface Medication {
  id: string;
  userId: string;
  name: string;
  dosage: string;
  frequency: string;
  times: string[];
  startDate?: string;
  endDate?: string;
  notes?: string;
  createdAt: number;
  updatedAt?: number;
}

export const subscribeToMedications = (userId: string, onUpdate: (meds: Medication[]) => void) => {
  const q = query(
    collection(db, `users/${userId}/medications`),
    orderBy('createdAt', 'desc')
  );

  return onSnapshot(q, (snapshot) => {
    const meds = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    } as Medication));
    onUpdate(meds);
  }, (error) => {
    handleFirestoreError(error, OperationType.LIST, `users/${userId}/medications`);
  });
};

export const addMedication = async (userId: string, data: Omit<Medication, 'id' | 'createdAt'>) => {
  try {
    return await addDoc(collection(db, `users/${userId}/medications`), {
      ...data,
      userId,
      createdAt: Date.now()
    });
  } catch (e) {
    handleFirestoreError(e, OperationType.CREATE, `users/${userId}/medications`);
  }
};

export const updateMedication = async (userId: string, medId: string, data: Partial<Medication>) => {
  try {
    const medRef = doc(db, `users/${userId}/medications`, medId);
    await updateDoc(medRef, {
      ...data,
      updatedAt: Date.now()
    });
  } catch (e) {
    handleFirestoreError(e, OperationType.UPDATE, `users/${userId}/medications/${medId}`);
  }
};

export const deleteMedication = async (userId: string, medId: string) => {
  try {
    await deleteDoc(doc(db, `users/${userId}/medications`, medId));
  } catch (e) {
    handleFirestoreError(e, OperationType.DELETE, `users/${userId}/medications/${medId}`);
  }
};
