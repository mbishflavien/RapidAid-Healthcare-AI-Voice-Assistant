export interface SymptomAnalysis {
  symptoms: string[];
  potentialConditions: { name: string; likelihood: string; description: string }[];
  urgency: 'Low' | 'Medium' | 'High' | 'Emergency';
  recommendations: string[];
  disclaimer?: string;
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
  transcriptions: Transcription[];
}

export interface HealthProfile {
  age?: number;
  gender?: string;
  conditions?: string;
  allergies?: string;
  medications?: string;
  bloodType?: string;
}
