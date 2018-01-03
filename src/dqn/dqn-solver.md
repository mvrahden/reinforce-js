# DQN-Solver Code Example


```typescript
// TypeScript
import { DQNSolver, DQNOpt, Env } from 'reinforce-js';

// JavaScript
// const DQNSolver = require('reinforce-js').DQNSolver;
// const DQNOpt = require('reinforce-js').DQNOpt;

const width = 400;
const height = 400;
const env = new Env(width, height);

const opt = new DQNOpt();
opt.setGamma(0.9);
opt.setEpsilon(0.15);
opt.setAlpha(0.005);
opt.setExperienceAddEvery(5);
opt.setExperienceSize(10000);
opt.setLearningStepsPerIteration(5);
opt.setTDErrorClamp(1.0);
opt.setNumHiddenUnits(R.randi(20, 100));

// outfit solver with environment complexity and specs
const dqnSolver = new DQNSolver(env, opt);
```