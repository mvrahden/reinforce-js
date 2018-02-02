# DQN-Solver Code Example

This is a working example of how to use the `DQNSolver`-Class in your Project. For further information on:
- how the solver is constructed,
- what all the single values stand for and
- how the solver can action on it's environment,

please also have a look [here](../../examples/dqn-solver.md).

```typescript
// TypeScript
import { DQNSolver, DQNOpt, DQNEnv } from 'reinforce-js';

// JavaScript
// const DQNSolver = require('reinforce-js').DQNSolver;
// const DQNEnv = require('reinforce-js').DQNEnv;
// const DQNOpt = require('reinforce-js').DQNOpt;

const width = 400;
const height = 400;
const env = new DQNEnv(width, height);

const opt = new DQNOpt();
opt.setTrainingMode(true);
opt.setNumberOfHiddenUnits(100);
opt.setEpsilonDecay(1.0, 0.1, 1e6);
opt.setEpsilon(0.05);
opt.setGamma(0.9);
opt.setAlpha(0.005);
opt.setLossClipping(true);
opt.setLossClamp(1.0);
opt.setRewardClipping(true);
opt.setRewardClamp(1.0);
opt.setExperienceSize(1e6);
opt.setReplayInterval(5);
opt.setReplaySteps(5);

// outfit solver with environment complexity and specs
const dqnSolver = new DQNSolver(env, opt);
```