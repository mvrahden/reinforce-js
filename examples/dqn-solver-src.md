# DQN-Solver Code Example

This is a working example of how to use the `DQNSolver`-Class in your Project. For further information on:
- how the solver is constructed,
- what all the single values stand for and
- how the solver can action on it's environment,

please also have a look [here](./dqn-solver.md).

To see a live example visit [learning-agents](https://mvrahden.github.io/learning-agents/).

```typescript
// TypeScript
import { DQNSolver, DQNOpt, DQNEnv } from 'reinforce-js';

// JavaScript
// const DQNSolver = require('reinforce-js').DQNSolver;
// const DQNEnv = require('reinforce-js').DQNEnv;
// const DQNOpt = require('reinforce-js').DQNOpt;

const width = 400;
const height = 400;
const numberOfStates = 20;
const numberOfActions = 4;
const env = new DQNEnv(width, height, numberOfStates, numberOfActions);

const opt = new DQNOpt();
opt.setTrainingMode(true);
opt.setNumberOfHiddenUnits([100]);
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

/*
Outfit solver with environment complexity and specs.
After configuration it's ready to train its untrained Q-Network and learn from SARSA experiences.
*/
const dqnSolver = new DQNSolver(env, opt);

/*
Determine a state, e.g.:
*/
const state = [ /* Array with numerical values and length of 20 as configured via numberOfStates */ ];

/*
Now inject state and receive the preferred action as index from 0 to 3 as configured via numberOfActions.
*/
const action = dqnSolver.decide(state);

/*
Now calculate some Reward and let the Solver learn from it, e.g.:
*/
const reward = 0.9;

dqnSolver.learn(reward);
```
