# TD-Solver Code Example


```typescript
// TypeScript
import { TDSolver, TDOpt, TDEnv } from 'reinforce-js';

// JavaScript
// const TDSolver = require('reinforce-js').TDSolver;
// const TDEnv = require('reinforce-js').TDEnv;
// const TDOpt = require('reinforce-js').TDOpt;

const width = 400;
const height = 400;
const numberOfStates = 20;
const numberOfActions = 4;
const env = new TDEnv(width, height, numberOfStates, numberOfActions);

const opt = new TDOpt();
opt.setUpdate('qlearn'); // or 'sarsa'
opt.setGamma(0.9);
opt.setEpsilon(0.2);
opt.setAlpha(0.1);
opt.setLambda(0);
opt.setReplacingTraces(true);
opt.setnumberOfPlanningSteps(50);

opt.setSmoothPolicyUpdate(true);
opt.setBeta(0.1);

// outfit solver with environment complexity and specs
const tdSolver = new TDSolver(env, opt);
```