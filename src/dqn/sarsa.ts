import { Mat } from 'recurrent-js';

export interface SarsaExperience  {
  s0: Mat;
  a0: number;
  r0: number;
  s1: Mat;
  a1: number;
}
