# REINFORCE-ts
**REINFORCE-ts** is an object-oriented Typescript port of _Andrej Karpathy's_ Reinforcement Learning library that implements several common RL algorithms.
In particular, the library currently includes:

* **Dynamic Programming** methods
* (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
* **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks
* **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures for dealing with continuous action spaces. (_very alpha, likely buggy or at the very least finicky and inconsistent_)

For further Information see the [reinforce-js](https://github.com/karpathy/reinforcejs) repository.

# Work in Progress!
Please be aware that this repository is still _under construction_.
There are still classes to be added, e.g. *DPSolver*, *TDSolver*, *SimpleReinforcementSolver*, *RecurrentReinforcementSolver*, *DeterministPG* and their individual *Opts* and *Envs*

# Dependencies

This Library relies on the object-oriented _Deep Recurrent Neural Network_ library:

* **GitHub**: [recurrent-ts](https://github.com/mvrahden/recurrent-ts)
* **npm**: [recurrent-ts](https://www.npmjs.com/package/recurrent-ts)

# Use as `npm`-Project Dependency

## To install as dependency:

Download available `@npm`: [reinforce-ts](https://www.npmjs.com/package/reinforce-ts)

Install via command line:

```
npm install --save reinforce-ts
```

## To use the Library in Production:

Currently exposed Classes:

* `Solver` - Generic Interface
* `DQNSolver` - Concrete Solver
* `Env` - Environment a Solver
* `Opt` - Options a Solver

These classes can be directly imported from this `npm` module, e.g.:
```typescript
import { Solver, Env } from 'reinforce-ts';
```

# Contribute

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. Let's follow primarily the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the included *tslint-google.json* configuration file.

# License

As of License-File: *MIT*
