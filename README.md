# REINFORCE-ts
**REINFORCE-ts** is an object-oriented Typescript port of _Andrej Karpathy's_ Reinforcement Learning library that implements several common RL algorithms.
In particular, the library currently includes:

* **Dynamic Programming** methods
* (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
* **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks
* **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures for dealing with continuous action spaces. (_very alpha, likely buggy or at the very least finicky and inconsistent_)

For further Information see the [reinforce-js](https://github.com/karpathy/reinforcejs) repository.

# Work in Progress!
Please be aware that this repository is still _under construction_. Changes are likely to happen.
There are still classes to be added, e.g. *DPSolver*, *TDSolver*, *SimpleReinforcementSolver*, *RecurrentReinforcementSolver*, *DeterministPG* and their individual *Opts* and *Envs*

# Dependencies

This Library relies on the object-oriented _Deep Recurrent Neural Network_ library:

* **GitHub**: [recurrent-ts](https://github.com/mvrahden/recurrent-ts)
* **npm**: [recurrent-ts](https://www.npmjs.com/package/recurrent-ts)

# Use as Project Dependency

## To install as a dependency:

Download available `@npm`: [reinforce-ts](https://www.npmjs.com/package/reinforce-ts)

Install via command line:

```
npm install --save reinforce-ts
```

## Use the Library in Production:

Currently exposed Classes (more to be expected soon):

* `Solver` - Generic Solver Interface
* `Env` - Generic *Environment* for a Solver (serves as a DQNEnv currently; changes to be expected)
* `Opt` - Generic *Options* for a Solver (serves as a DQNOpt currently; changes to be expected)
* `DQNSolver` - Concrete *Deep Q-Learning* Solver

To be implemented:

- `DQNEnv` - Concrete *Environment* for DQNSolver creation
- `DQNOpt` - Concrete *Options* for DQNSolver creation
- `DPSolver` - Concrete *Temporal Difference* Solver
- `DPEnv` - Concrete *Environment* for DPSolver creation
- `DPOpt` - Concrete *Options* for DPSolver creation
- `TDSolver` - Concrete *Temporal Difference* Solver
- `TDEnv` - Concrete *Environment* for TDSolver creation
- `TDOpt` - Concrete *Options* for TDSolver creation
- `SimpleReinforcementSolver` - Concrete *Simple Reinforcement* Solver
- `SimpleReinforcementEnv` - Concrete *Environment* for SimpleReinforcementSolver creation
- `SimpleReinforcementOpt` - Concrete *Options* for SimpleReinforcementSolver creation
- `RecurrentReinforcementSolver` - Concrete *Recurrent Reinforcement* Solver
- `RecurrentReinforcementEnv` - Concrete *Environment* for RecurrentReinforcementSolver creation
- `RecurrentReinforcementOpt` - Concrete *Options* for RecurrentReinforcementSolver creation
- `DeterministPGSolver` - Concrete *Deterministic Policy Gradient* Solver
- `DeterministPGEnv` - Concrete *Environment* for DeterministPGSolver creation
- `DeterministPGOpt` - Concrete *Options* for DeterministPGSolver creation

These classes can be directly imported from this `npm` module, e.g.:
```typescript
import { Solver, Env } from 'reinforce-ts';
```

## Further Info

The transpiled Javascript-target is `ES6`, with a `CommonJS` module format.

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
