# reinforce-js
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)

[dqn-solver]: examples/dqn-solver.md
[dqn-solver-src]: examples/dqn-solver-src.md
[td-solver-src]: examples/td-solver-src.md

**The reinforce-js library** &ndash; is a collection of various machine learning solver. The library is an object-oriented approach and tries to deliver simplified interfaces that make using the algorithms pretty easy (baked with [Typescript](https://github.com/Microsoft/TypeScript)). More over it is an extension of _Andrej Karpathy's_ reinforcement learning library that implements several common RL algorithms.
In particular, the library currently includes:

* **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks ([DQNSolver Details][dqn-solver] and related [Google DeepMind Paper](https://www.nature.com/articles/nature14236))
* **Dynamic Programming** methods
* (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
* **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures for dealing with continuous action spaces. (_very alpha, likely buggy or at the very least finicky and inconsistent_)

## For Production Use

### What does the Library offer?

Currently exposed Classes:

* **Solver** - Generic Solver Interface
* **Env** - Generic *Environment* for a Solver
* **Opt** - Generic *Options* for a Solver

#### DQN-Solver

[Code-Example][dqn-solver-src] and [General Information][dqn-solver]

* `DQNSolver` - Concrete **Deep Q-Learning Solver**
  * This class is containing the main Deep Q-Learning algorithm of the *DeepMind* paper. On instantiation it needs to be configured with two configuration objects. It is an algorithm, which has minimum knowledge of its environment. The behavior of the algorithm can be tuned via its hyperparamters (`DQNOpt`).
  * The Deep Q-Learning algorithm is designed to have a certain universality, since its reasoning is just depending on a environmental perception and an environmental feedback.
  * The [learning-agents](https://mvrahden.github.io/learning-agents)-implementation shows that the DQNSolver can also be designed in such a way, that its agent has a maximum autonomy by establishing its own reward-scheme.
* `DQNOpt` - Concrete options of `DQNSolver`
  * This class is for the configuration of the DQNSolver. It holds all the **hyperparameter** for the DQNSolver. For the detailed initialization please see the [General Information][dqn-solver].
* `DQNEnv` - Concrete environment of `DQNSolver`
  * This class is for the configuration of the DQNSolver. It holds the boundary-measures of the environment, in which the DQNSolver should operate. For the detailed initialization please see the [General Information][dqn-solver].
* **Example Application**: [Learning Agents](https://mvrahden.github.io/learning-agents) (GitHub Page)

#### TD-Solver 

[Code-Example][td-solver-src]

* **TDSolver** - Concrete *Temporal Difference* Solver
* **TDOpt** - Concrete *Options* for TDSolver creation
* **TDEnv** - Concrete *Environment* for TDSolver creation

Planned to be implemented:

* **DPSolver** - Concrete *Dynamic Programming* Solver
* **DPOpt** - Concrete *Options* for DPSolver creation
* **SimpleReinforcementSolver** - Concrete *Simple Reinforcement* Solver
* **SimpleReinforcementOpt** - Concrete *Options* for SimpleReinforcementSolver creation
* **RecurrentReinforcementSolver** - Concrete *Recurrent Reinforcement* Solver
* **RecurrentReinforcementOpt** - Concrete *Options* for RecurrentReinforcementSolver creation
* **DeterministPGSolver** - Concrete *Deterministic Policy Gradient* Solver
* **DeterministPGOpt** - Concrete *Options* for DeterministPGSolver creation

### How to install as a dependency:

Download available `@npm`: [reinforce-js](https://www.npmjs.com/package/reinforce-js)

Install via command line:

```
npm install --save reinforce-js@latest
```

The project directly ships with the transpiled Javascript code.
And for TypeScript development it also contains Map-files and Declaration-files.

### How to import?

These classes can be imported from this `npm` module, e.g.:
```typescript
import { DQNSolver, DQNOpt, DQNEnv } from 'reinforce-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
const DQNSolver = require('reinforce-js').DQNSolver;
const DQNOpt = require('reinforce-js').DQNOpt;
const DQNEnv = require('reinforce-js').DQNEnv;
```

### Example Application

For the DQN-Solver please visit [Learning Agents](https://mvrahden.github.io/learning-agents) (GitHub Page).

## For Contributors

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. Let's follow primarily the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the included *tslint-google.json* configuration file.

### Dependencies

This Library relies on the object-oriented _Deep Recurrent Neural Network_ library:

* **GitHub**: [recurrent-js](https://github.com/mvrahden/recurrent-js)
* **npm**: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

### Work in Progress
Please be aware that this repository is still _under construction_. Changes are likely to happen.
There are still classes to be added, e.g. *DPSolver*, *SimpleReinforcementSolver*, *RecurrentReinforcementSolver*, *DeterministPG* and their individual *Opts* and *Envs*

## License

As of License-File: [MIT](LICENSE)
