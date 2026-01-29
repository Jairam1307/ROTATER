
export interface BoundingBox {
  latMin: number;
  latMax: number;
  lonMin: number;
  lonMax: number;
}

export interface ClimateStats {
  date: string;
  temperature: number; // Celsius
  rainfall: number; // mm
  ndvi: number; // 0-1
  anomaly: number;
}

export interface ClimatologyStats {
  parameter: string;
  jan: number;
  feb: number;
  mar: number;
  apr: number;
  may: number;
  jun: number;
  jul: number;
  aug: number;
  sep: number;
  oct: number;
  nov: number;
  dec: number;
  annual: number;
}

export interface WeatherData {
  temp: number;
  condition: string;
  humidity: number;
  windSpeed: number;
  uvIndex: number;
  visibility: number;
}

export interface DisasterEvent {
  date: string;
  type: string;
  severity: 'Moderate' | 'Severe' | 'Extreme';
  impact: string;
  areas: string[];
}

export interface ReliefInfrastructure {
  name: string;
  type: 'Shelter' | 'Medical' | 'Distribution' | 'Response Hub';
  distance: string;
  location: string;
  uri: string;
}

export interface NGO {
  name: string;
  origin: string;
  aidType: string;
  mode: 'On-ground' | 'Remote' | 'Hybrid';
}

export interface ResearchIntelligence {
  summary: string;
  disasters: DisasterEvent[];
  infrastructure: ReliefInfrastructure[];
  ngos: NGO[];
  predictions: Prediction[];
}

export interface Prediction {
  month: string;
  riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
  predictedTemp: number;
  description: string;
}

export interface Calamity {
  year: number;
  type: 'Flood' | 'Drought' | 'Cyclone' | 'Heatwave';
  intensity: string;
  month: string;
}

export interface ViewState {
  view: 'intro' | 'dashboard';
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

export interface GroundingSource {
  title: string;
  uri: string;
}
