export interface TokenEmbedding {
  token: string;
  x: number; // Sentiment / Valence (-10 to 10)
  y?: number; // Intensity / Arousal (-10 to 10)
  z?: number; // Formality / Professionalism (-10 to 10)
  w?: number; // Complexity / Abstractness (-10 to 10)
  vector?: number[]; // High-dimensional representation (e.g., 768d)
  color?: string;
  cluster?: number; // Group index
  isCentroid?: boolean; // If this represents a cluster center
  modelVectors?: Record<string, number[]>; // Vectors for different models (e.g., 'gemini-flash', 'gemini-pro')
}

export type ModelType = 'gemini-flash' | 'gemini-pro';

export type Dimension = 1 | 2 | 3 | 4;

export interface VectorArithmetic {
  formula: string;
  tokens: string[];
  operators: string[];
  resultToken: string | null;
}

export interface ContextualToken extends TokenEmbedding {
  index: number;
  originalX: number;
  originalY: number;
  original?: TokenEmbedding;
}

export interface DimensionInfo {
  name: string;
  description: string;
  minLabel: string;
  maxLabel: string;
}
