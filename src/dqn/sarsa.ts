import { Mat } from 'recurrent-js';

export interface SarsaExperience  {
  s0: Mat;      // last state after acting (from t-1)
  a0: number;   // last action Index after acting (from t-1)
  r0: number;   // current reward after learning (from t)
  s1: Mat;      // current state while acting (from t)
  a1: number;   // current action Index while acting (from t)
}
