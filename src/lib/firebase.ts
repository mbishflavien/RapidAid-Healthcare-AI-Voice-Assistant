import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getFirestore, doc, getDocFromServer } from 'firebase/firestore';

// Default configuration fallback so CI/CD builds (like Vercel) succeed even if config file is not committed
const defaultFirebaseConfig: Record<string, string> = {
  projectId: "gen-lang-client-0750947491",
  appId: "1:795885399173:web:155a06ff9de5eaccd28dec",
  apiKey: "AIzaSyBhmoz4PJpCZLVCKW01OLkBSjz2Bj5xOUQ",
  authDomain: "gen-lang-client-0750947491.firebaseapp.com",
  firestoreDatabaseId: "ai-studio-32bf3f3a-83a8-4528-9868-d6c5737bfd0c",
  storageBucket: "gen-lang-client-0750947491.firebasestorage.app",
  messagingSenderId: "795885399173"
};

// Safely probe for firebase-applet-config.json with import.meta.glob to prevent Rollup resolve errors when file is not present in Git
let fileConfig: Record<string, any> = {};
const configFiles = import.meta.glob<Record<string, any>>(['/firebase-applet-config.json', '../../firebase-applet-config.json'], { eager: true });
for (const key of Object.keys(configFiles)) {
  if (configFiles[key]) {
    fileConfig = configFiles[key].default || configFiles[key];
    break;
  }
}

const config = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || fileConfig.apiKey || defaultFirebaseConfig.apiKey,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || fileConfig.authDomain || defaultFirebaseConfig.authDomain,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || fileConfig.projectId || defaultFirebaseConfig.projectId,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || fileConfig.storageBucket || defaultFirebaseConfig.storageBucket,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || fileConfig.messagingSenderId || defaultFirebaseConfig.messagingSenderId,
  appId: import.meta.env.VITE_FIREBASE_APP_ID || fileConfig.appId || defaultFirebaseConfig.appId,
  firestoreDatabaseId: import.meta.env.VITE_FIREBASE_FIRESTORE_DATABASE_ID || fileConfig.firestoreDatabaseId || defaultFirebaseConfig.firestoreDatabaseId
};

const app = initializeApp(config);
export const db = getFirestore(app, config.firestoreDatabaseId);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId?: string | null;
    email?: string | null;
    emailVerified?: boolean | null;
    isAnonymous?: boolean | null;
    tenantId?: string | null;
    providerInfo?: {
      providerId?: string | null;
      email?: string | null;
    }[];
  }
}

export function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData?.map(provider => ({
        providerId: provider.providerId,
        email: provider.email,
      })) || []
    },
    operationType,
    path
  }
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}
async function testConnection() {
  try {
    await getDocFromServer(doc(db, 'test', 'connection'));
  } catch (error) {
    // Silence persistent configuration warnings - errors will still be caught during actual operations
  }
}
testConnection();
