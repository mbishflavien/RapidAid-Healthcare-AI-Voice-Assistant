export interface SymptomAnalysis {
  symptoms: string[];
  potentialConditions: { name: string; likelihood: string; description: string }[];
  urgency: 'Low' | 'Medium' | 'High' | 'Emergency';
  recommendations: string[];
  disclaimer?: string;
}

export interface VitalSigns {
  heartRate: number; // bpm
  bloodPressureSystolic: number; // mmHg
  bloodPressureDiastolic: number; // mmHg
  oxygenSaturation: number; // %
  temperature: number; // °F
  respiratoryRate: number; // /min
  painLevel: number; // 0-10
  lastRecorded: number;
}

export type AcuityLevel = 'ESI-1' | 'ESI-2' | 'ESI-3' | 'ESI-4' | 'ESI-5';

export interface SoapNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  icdCodes: { code: string; description: string }[];
  generatedAt: number;
}

export interface Transcription {
  id?: string;
  text?: string;
  analysis?: SymptomAnalysis;
  isUser: boolean;
  timestamp: number;
  fromVoice?: boolean;
}

export interface Session {
  id: string;
  title: string;
  timestamp: number;
  transcriptions?: Transcription[];
}

export interface HealthProfile {
  age?: number;
  gender?: string;
  conditions?: string;
  allergies?: string;
  medications?: string;
  bloodType?: string;
}
