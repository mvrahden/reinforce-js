import { Net, Graph, Mat, RandMat, R } from 'recurrent-js';

import { Solver } from '../Solver';
import { Env } from '../Env';
import { DQNOpt } from './DQNOpt';
import { SarsaExperience } from './sarsa';

export class DQNSolver extends Solver {
  
  // Env
  public numberOfStates: number;
  public numberOfActions: number;

  // Opts
  public numberOfHiddenUnits: number;

  public readonly epsilonMax: number;
  public readonly epsilonMin: number;
  public readonly epsilonDecayPeriod: number;
  public readonly epsilon: number;
  
  public readonly gamma: number;
  public readonly alpha: number;
  public readonly doLossClipping: boolean;
  public readonly lossClamp: number;
  public readonly doRewardClipping: any;
  public readonly rewardClamp: any;
  public readonly experienceSize: number;
  public readonly keepExperienceInterval: number;
  public readonly replaySteps: number;
  
  // Local
  protected net: Net;
  protected previousGraph: Graph;
  protected shortTermMemory: SarsaExperience = { s0: null, a0: null, r0: null, s1: null, a1: null };
  protected longTermMemory: Array<SarsaExperience>;
  protected isInTrainingMode: boolean;
  protected learnTick: number;
  protected memoryTick: number;

  constructor(env: Env, opt: DQNOpt) {
    super(env, opt);
    this.numberOfHiddenUnits = opt.get('numberOfHiddenUnits');
    
    this.epsilonMax = opt.get('epsilonMax');
    this.epsilonMin = opt.get('epsilonMin');
    this.epsilonDecayPeriod = opt.get('epsilonDecayPeriod');
    this.epsilon = opt.get('epsilon');
    
    this.experienceSize = opt.get('experienceSize');
    this.gamma = opt.get('gamma');
    this.alpha = opt.get('alpha');
    this.doLossClipping = opt.get('doLossClipping');
    this.lossClamp = opt.get('lossClamp');
    this.doRewardClipping = opt.get('doRewardClipping');
    this.rewardClamp = opt.get('rewardClamp');
    
    this.keepExperienceInterval = opt.get('keepExperienceInterval');
    this.replaySteps = opt.get('replaySteps');
    
    this.isInTrainingMode = opt.get('trainingMode');

    this.reset();
  }

  public reset(): void {
    this.numberOfHiddenUnits = this.opt.get('numberOfHiddenUnits');
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numberOfActions');

    const netOpts = {
      inputSize: this.numberOfStates,
      hiddenSize: this.numberOfHiddenUnits,
      outputSize: this.numberOfActions
    };
    this.net = new Net(netOpts);

    this.memoryTick = 0;
    this.learnTick = 0;
    
    this.shortTermMemory.s0 = null;
    this.shortTermMemory.a0 = null;
    this.shortTermMemory.r0 = null;
    this.shortTermMemory.s1 = null;
    this.shortTermMemory.a1 = null;

    this.longTermMemory = [];
  }

  /**
   * Sets the training mode of the agent to the given state.
   * given: true => training mode on, linearly decaying epsilon 
   * given: false => deployment mode on, constant epsilon
   * (Epsilon is the threshold for controlling the epsilon greedy policy (or exploration via random actions))
   * @param trainingMode true if training mode should be switched on, false if training mode should be switched off
   */
  public setTrainingModeTo(trainingMode: boolean): void {
    this.isInTrainingMode = trainingMode;
  }

  /**
   * Returns the current state of training mode
   * @returns true if is in training mode, else false.
   */
  public getTrainingMode(): boolean {
    return this.isInTrainingMode;
  }

  /**
   * Transforms Agent to (ready-to-stringify) JSON object
   */
  public toJSON(): object {
    const j = {
      ns: this.numberOfStates,
      nh: this.numberOfHiddenUnits,
      na: this.numberOfActions,
      net: Net.toJSON(this.net)
    };
    return j;
  }

  /**
   * Loads an Agent from a (already parsed) JSON object
   * @param json with properties `nh`, `ns`, `na` and `net`
   */
  public fromJSON(json: { ns, nh, na, net }): void {
    this.numberOfStates = json.ns;
    this.numberOfHiddenUnits = json.nh;
    this.numberOfActions = json.na;
    this.net = Net.fromJSON(json.net);
  }

  /**
   * Decide an action according to current state
   * @param state current state
   * @returns index of argmax action
   */
  public decide(state: Array<number>): number {
    const stateVector = new Mat(this.numberOfStates, 1);
    stateVector.setFrom(state);

    const actionIndex = this.epsilonGreedyActionPolicy(stateVector);

    this.shiftStateMemory(stateVector, actionIndex);

    return actionIndex;
  }

  protected epsilonGreedyActionPolicy(stateVector: Mat): number {
    let actionIndex: number = 0;
    if (Math.random() < this.currentEpsilon()) { // greedy Policy Filter
      actionIndex = R.randi(0, this.numberOfActions);
    } else {
      // Q function
      const actionVector = this.forwardQ(stateVector);
      actionIndex = R.maxi(actionVector.w); // returns index of argmax action 
    }
    return actionIndex;
  }

  /**
   * Determines the current epsilon.
   */
  protected currentEpsilon(): number {
    if (this.isInTrainingMode) {
      if (this.learnTick < this.epsilonDecayPeriod) {
        return this.epsilonMax - (this.epsilonMax - this.epsilonMin) / this.epsilonDecayPeriod * this.learnTick;
      } else {
        return this.epsilonMin;
      }
    } else {
      return this.epsilon;
    }
  }

  /**
   * Determine Outputs based on Forward Pass
   * @param stateVector Matrix with states
   * @return Matrix (Vector) with predicted actions values
   */
  protected forwardQ(stateVector: Mat | null): Mat {
    const graph = new Graph(false);
    const a2Mat = this.determineActionVector(graph, stateVector);
    return a2Mat;
  }

  /**
   * Determine Outputs based on Forward Pass
   * @param stateVector Matrix with states
   * @return Matrix (Vector) with predicted actions values
   */
  protected backwardQ(stateVector: Mat | null): Mat {
    const graph = new Graph(true);  // with backprop option
    const a2Mat = this.determineActionVector(graph, stateVector);
    return a2Mat;
  }


  protected determineActionVector(graph: Graph, stateVector: Mat): Mat {
    const a2mat = this.net.forward(stateVector, graph);
    this.backupGraph(graph); // back this up. Kind of hacky isn't it
    return a2mat;
  }

  protected backupGraph(graph: Graph): void {
    this.previousGraph = graph;
  }

  protected shiftStateMemory(stateVector: Mat, actionIndex: number): void {
    this.shortTermMemory.s0 = this.shortTermMemory.s1;
    this.shortTermMemory.a0 = this.shortTermMemory.a1;
    this.shortTermMemory.s1 = stateVector;
    this.shortTermMemory.a1 = actionIndex;
  }

  /**
   * perform an update on Q function
   * @param r current reward passed to learn
   */
  public learn(r: number): void {
    if (this.shortTermMemory.r0 && this.alpha > 0) {
      r = this.clipReward(r);
      this.learnFromSarsaTuple(this.shortTermMemory); // a measure of surprise
      this.addToReplayMemory();
      this.limitedSampledReplayLearning();
    }
    this.shortTermMemory.r0 = r; // store reward for next update
  }

  /**
   * Clips Reward, If doRewardClipping is activated
   * @param r current reward
   */
  protected clipReward(r: number): number {
    return this.doRewardClipping ? Math.sign(r) * Math.min(Math.abs(r), this.rewardClamp) : r;
  }

  /**
   * Learn from sarsa tuple
   * @param {SarsaExperience} sarsa Object containing states, actions and reward of t & t-1
   */
  protected learnFromSarsaTuple(sarsa: SarsaExperience): void {
    const q1Max = this.getTargetQ(sarsa.s1, sarsa.r0);
    const lastActionVector = this.backwardQ(sarsa.s0);
    const q0Max = lastActionVector.w[sarsa.a0];

    // Loss_i(w_i) = [(r0 + gamma * Q'(s',a') - Q(s,a)) ^ 2]
    let loss = q0Max - q1Max;

    loss = this.clipLoss(loss);
    lastActionVector.dw[sarsa.a0] = loss;
    this.previousGraph.backward();

    // discount all weights of net by: w[i] = w[i] - (alpha * dw[i]);
    this.net.update(this.alpha);
  }

  /**
   * Clipp loss to interval of [-delta, delta], e.g. [-1, 1] (Derivative Huber Loss)
   * @returns {number} limited tdError
   */
  protected clipLoss(loss: number): number {
    if(this.doLossClipping) {
      if (loss > this.lossClamp) {
        loss = this.lossClamp;
      }
      else if (loss < -this.lossClamp) {
        loss = -this.lossClamp;
      }
    }
    return loss;
  }

  protected getTargetQ(s1: Mat, r0: number): number {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')
    const targetActionVector = this.forwardQ(s1);
    const targetActionIndex = R.maxi(targetActionVector.w);
    const qMax = r0 + this.gamma * targetActionVector.w[targetActionIndex];
    return qMax;
  }

  protected addToReplayMemory(): void {
    if (this.learnTick % this.keepExperienceInterval === 0) {
      this.addShortTermToLongTermMemory();
    }
    this.learnTick++;
  }

  protected addShortTermToLongTermMemory(): void {
    const sarsa = this.extractSarsaExperience();
    this.longTermMemory[this.memoryTick] = sarsa;
    this.memoryTick++;
    if (this.memoryTick > this.experienceSize - 1) { // roll over
      this.memoryTick = 0;
    }
  }

  protected extractSarsaExperience(): SarsaExperience {
    const s0 = new Mat(this.shortTermMemory.s0.rows, this.shortTermMemory.s0.cols);
    s0.setFrom(this.shortTermMemory.s0.w);
    const s1 = new Mat(this.shortTermMemory.s1.rows, this.shortTermMemory.s1.cols);
    s1.setFrom(this.shortTermMemory.s1.w);
    const sarsa = {
      s0,
      a0: this.shortTermMemory.a0,
      r0: this.shortTermMemory.r0,
      s1,
      a1: this.shortTermMemory.a1
    };
    return sarsa;
  }

  /**
   * Sample some additional experience (minibatches) from replay memory and learn from it
   */
  protected limitedSampledReplayLearning(): void {
    for (let i = 0; i < this.replaySteps; i++) {
      const ri = R.randi(0, this.longTermMemory.length); // todo: priority sweeps?
      const sarsa = this.longTermMemory[ri];
      this.learnFromSarsaTuple(sarsa);
    }
  }
}
