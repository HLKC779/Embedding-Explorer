import { TokenEmbedding, DimensionInfo } from './types';

export const DIMENSIONS: Record<number, DimensionInfo> = {
  1: {
    name: 'Emotional Valence',
    description: 'The intrinsic attractiveness (positive) or aversiveness (negative) of an event, object, or situation.',
    minLabel: 'Negative / Terrible',
    maxLabel: 'Positive / Excellent'
  },
  2: {
    name: 'Arousal / Intensity',
    description: 'The physiological and psychological state of being awoken or of sense organs being stimulated to a point of perception.',
    minLabel: 'Calm / Neutral',
    maxLabel: 'Excited / Fury'
  },
  3: {
    name: 'Professionalism / Formality',
    description: 'The level of etiquette, politeness, and structured language used in communication.',
    minLabel: 'Slang / Informal',
    maxLabel: 'Formal / Academic'
  },
  4: {
    name: 'Complexity / Abstractness',
    description: 'The level of conceptual depth, linguistic complexity, or abstract nature of the token.',
    minLabel: 'Simple / Concrete',
    maxLabel: 'Complex / Abstract'
  }
};

// Helper to generate a 16-dimensional vector based on the base coordinates + noise
const v = (x: number, y: number, z: number, w: number, model: 'flash' | 'pro'): number[] => {
  const base = [x, y, z, w];
  // Pro model has more "focused" clusters and less noise, but higher magnitude differences
  const multiplier = model === 'pro' ? 1.2 : 1.0;
  const noiseSeed = model === 'pro' ? 0.1 : 0.5;
  
  return [...base.map(b => b * multiplier), ...Array(12).fill(0).map((_, i) => {
    const val = (x * (i%3) + y * Math.sin(i)) / 2;
    return val * multiplier + (Math.random() - 0.5) * noiseSeed;
  })];
};

export const INITIAL_TOKENS: TokenEmbedding[] = [
  { token: 'Hate', x: -9, y: 8, z: -2, w: -5, color: '#ef4444' },
  { token: 'Dislike', x: -5, y: 3, z: 0, w: -2, color: '#f87171' },
  { token: 'Neutral', x: 0, y: 0, z: 2, w: 0, color: '#94a3b8' },
  { token: 'Like', x: 5, y: 4, z: 1, w: -1, color: '#4ade80' },
  { token: 'Love', x: 9, y: 9, z: -1, w: 2, color: '#22c55e' },
  { token: 'Terrible', x: -10, y: 5, z: 1, w: -3, color: '#b91c1c' },
  { token: 'Excellent', x: 10, y: 6, z: 3, w: 4, color: '#15803d' },
  { token: 'Fury', x: -8, y: 10, z: -4, w: 1, color: '#7f1d1d' },
  { token: 'Euphoria', x: 9, y: 10, z: -3, w: 5, color: '#166534' },
  { token: 'Yo', x: 1, y: 5, z: -10, w: -8, color: '#f59e0b' },
  { token: 'Greetings', x: 2, y: 2, z: 10, w: 3, color: '#6366f1' },
  { token: 'Whatever', x: -1, y: -2, z: -5, w: 0, color: '#64748b' },
  { token: 'Indeed', x: 3, y: 1, z: 8, w: 7, color: '#4338ca' },
  { token: 'Awesome', x: 8, y: 7, z: 2, w: 3, color: '#facc15' },
  { token: 'Boring', x: -3, y: -8, z: 1, w: -4, color: '#cbd5e1' },
  { token: 'Chaotic', x: 0, y: 9, z: -8, w: 8, color: '#ec4899' },
  { token: 'Serene', x: 7, y: -7, z: 4, w: -6, color: '#06b6d4' },
  { token: 'Paradox', x: 2, y: 4, z: 5, w: 10, color: '#8b5cf6' },
  { token: 'Coffee', x: 4, y: 6, z: 2, w: 1, color: '#78350f' },
  { token: 'Cold', x: -2, y: -5, z: 0, w: -2, color: '#3b82f6' },
  { token: 'Hot', x: 2, y: 5, z: 0, w: -1, color: '#ef4444' },
  { token: 'Ice', x: -5, y: -8, z: -2, w: -3, color: '#93c5fd' },
  { token: 'Iced Coffee', x: 2, y: 1, z: 2, w: -1, color: '#92400e' }
].map(t => ({
  ...t,
  vector: v(t.x, t.y || 0, t.z || 0, t.w || 0, 'flash'),
  modelVectors: {
    'gemini-flash': v(t.x, t.y || 0, t.z || 0, t.w || 0, 'flash'),
    'gemini-pro': v(t.x, t.y || 0, t.z || 0, t.w || 0, 'pro'),
  }
}));
