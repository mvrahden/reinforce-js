import { DQNSolver, DQNOpt, DQNEnv } from '.';
import { TDSolver, TDOpt, TDEnv } from '.';

describe('Given the imports, the example code should execute just fine', () => {
  // Environmental Variables
  const width = 400;
  const height = 400;
  const numberOfStates = 20;
  const numberOfActions = 4;

  // Common Hyperparameter
  const alpha = 0.005;
  const epsilon = 0.05;
  const gamma = 0.9;

  describe('DQNSolver:', () => {
    it('Sample code', () => {
      const env = new DQNEnv(width, height, numberOfStates, numberOfActions);

      const opt = new DQNOpt();
      opt.setTrainingMode(true);
      opt.setNumberOfHiddenUnits([100]);  // mind the Array
      opt.setEpsilonDecay(1.0, 0.1, 1e6);
      opt.setAlpha(alpha);
      opt.setEpsilon(epsilon);
      opt.setGamma(gamma);
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
    });
    
  });

  describe('TDSolver:', () => {
    it('Sample code', () => {
      const env = new TDEnv(width, height, numberOfStates, numberOfActions);

      const opt = new TDOpt();
      opt.setUpdate('qlearn'); // or 'sarsa'
      opt.setAlpha(alpha);
      opt.setEpsilon(epsilon);
      opt.setGamma(gamma);
      opt.setLambda(0);
      opt.setReplacingTraces(true);
      opt.setNumberOfPlanningSteps(50);

      opt.setSmoothPolicyUpdate(true);
      opt.setBeta(0.1);

      // outfit solver with environment complexity and specs
      const tdSolver = new TDSolver(env, opt);
    });
  });
});
