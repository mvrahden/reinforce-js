import { Net, Graph, Mat, RandMat, R } from 'recurrent-js';

import { Solver } from '../Solver';
import { Env } from '../Env';
import { DQNOpt } from './DQNOpt';
import { SARSA } from './sarsa';

export class DQNSolver extends Solver {
  // Opts
  public readonly alpha: number;
  public readonly epsilon: number;
  public readonly gamma: number;

  public readonly experienceSize: number;
  public readonly experienceAddEvery: number;
  public readonly learningStepsPerIteration: number;
  public readonly tdErrorClamp: number;
  public numberOfHiddenUnits: number;

  // Env
  public numberOfStates: number;
  public numberOfActions: number;

  // Local
  protected net: Net;
  protected previousGraph: Graph;
  protected a1: number | null = null;  // current action Index after acting (from t)
  protected a0: number | null = null;  // last action Index after acting (from t-1)
  protected s1: Mat | null = null;     // current state while acting (from t)
  protected s0: Mat | null = null;     // last state after acting (from t-1)
  protected r0: number | null = null;  // current reward after learning (from t)
  protected learnTick: number;
  protected experienceTick: number;
  protected memory: Array<SARSA>;
  protected tderror: number;

  constructor(env: Env, opt: DQNOpt) {
    super(env, opt);
    this.alpha = opt.get('alpha');
    this.epsilon = opt.get('epsilon');
    this.gamma = opt.get('gamma');
    this.experienceSize = opt.get('experienceSize');
    this.experienceAddEvery = opt.get('experienceAddEvery');
    this.learningStepsPerIteration = opt.get('learningStepsPerIteration');
    this.tdErrorClamp = opt.get('tdErrorClamp');
    this.numberOfHiddenUnits = opt.get('numberOfHiddenUnits');

    this.reset();
  }

  public reset(): void {
    this.numberOfHiddenUnits = this.opt.get('numberOfHiddenUnits');
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numberOfActions');

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    const netOpts = {
      inputSize: this.numberOfStates,
      hiddenSize: this.numberOfHiddenUnits,
      outputSize: this.numberOfActions
    };
    this.net = new Net(netOpts);

    this.memory = []; // experience
    this.experienceTick = 0; // where to insert
    this.learnTick = 0;

    // sarsa
    this.s0 = null;
    this.a0 = null;
    this.r0 = null;
    this.s1 = null;
    this.a1 = null;

    this.tderror = 0; // for visualization only...
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
   * Determine action on StateList
   * @param stateList 
   * @returns Index of argmax action
   */
  public act(stateList: Array<number>): number {
    const stateVector = new Mat(this.numberOfStates, 1);
    stateVector.setFrom(stateList);

    const actionIndex = this.epsilonGreedyActionPolicy(stateVector);

    this.shiftStateMemory(stateVector, actionIndex);

    return actionIndex;
  }

  private epsilonGreedyActionPolicy(stateVector: Mat): number {
    let actionIndex: number = 0;
    if (Math.random() < this.epsilon) { // greedy Policy Filter
      actionIndex = R.randi(0, this.numberOfActions);
    }
    else {
      // Q function
      const actionVector = this.forwardQ(stateVector);
      actionIndex = R.maxi(actionVector.w); // returns index of argmax action 
    }
    return actionIndex;
  }

  /**
   * Determine Outputs based on Forward Pass
   * @param stateVector Matrix with states
   * @return Matrix (Vector) with predicted actions values
   */
  private forwardQ(stateVector: Mat | null): Mat {
    const graph = new Graph(false);
    const a2Mat = this.determineActionVector(graph, stateVector);
    return a2Mat;
  }

  /**
   * Determine Outputs based on Forward Pass
   * @param stateVector Matrix with states
   * @return Matrix (Vector) with predicted actions values
   */
  private backwardQ(stateVector: Mat | null): Mat {
    const graph = new Graph(true);  // with backprop option
    const a2Mat = this.determineActionVector(graph, stateVector);
    return a2Mat;
  }


  private determineActionVector(graph: Graph, stateVector: Mat) {
    const a2mat = this.net.forward(stateVector, graph);
    // TODO: Hyperbolic activation of a2Mat
    const h2mat = graph.sigmoid(a2mat);

    this.backupGraph(graph); // back this up. Kind of hacky isn't it
    return h2mat;
  }

  private backupGraph(graph: Graph): void {
    this.previousGraph = graph;
  }

  private shiftStateMemory(stateVector: Mat, actionIndex: number): void {
    this.s0 = this.s1;
    this.s1 = stateVector;
    this.a0 = this.a1;
    this.a1 = actionIndex;
  }

  /**
   * perform an update on Q function
   * @param r1 current reward passed to learn
   */
  public learn(r1: number): void {
    if (this.r0 != null && this.alpha > 0) {
      // SARSA: learn from this tuple to get a sense of how "surprising" it is to the agent
      this.tderror = this.learnFromTuple({s0: this.s0, a0: this.a0, r0: this.r0, s1: this.s1, a1: this.a1}); // a measure of surprise

      this.addToReplayMemory();

      // sample some additional experience from replay memory and learn from it
      this.limitedSampledReplayLearning();
    }
    this.r0 = r1; // store reward for next update
  }

  /**
   * SARSA learn from tuple
   * @param {Mat|null} s0 last stateVector (from last action)
   * @param {number} a0 last action Index (from last action)
   * @param {number} r0 last reward (from after last action)
   * @param {Mat|null} s1 current StateVector (from current action)
   * @param {number|null} a1 current action Index (from current action)
   */
  private learnFromTuple(sarsa: SARSA): number {
    const q1Max = this.getTargetQ(sarsa.s1, sarsa.r0);
    const lastActionVector = this.backwardQ(sarsa.s0);
    let tdError = lastActionVector.w[sarsa.a0] - q1Max;  // Last Action Intensity - ( r0 + gamma * Current Action Intensity)

    tdError = this.huberLoss(tdError);
    lastActionVector.dw[sarsa.a0] = tdError;
    this.previousGraph.backward();

    // discount all weights of all Matrices by: w[i] = w[i] - (alpha * dw[i]);
    this.net.update(this.alpha);
    return tdError;
  }

  /**
   * Limit tdError to interval of [-tdErrorClapm, tdErrorClapm], e.g. [-1, 1]
   * @returns {number} limited tdError
   */
  private huberLoss(tdError: number): number {
    if (Math.abs(tdError) > this.tdErrorClamp) {
      if (tdError > this.tdErrorClamp) {
        tdError = this.tdErrorClamp;
      }
      else if (tdError < -this.tdErrorClamp) {
        tdError = -this.tdErrorClamp;
      }
    }
    return tdError;
  }

  private getTargetQ(s1: Mat, r0: number): number {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')
    const targetActionVector = this.forwardQ(s1);
    const targetActionIndex = R.maxi(targetActionVector.w);
    const qMax = r0 + this.gamma * targetActionVector.w[targetActionIndex];
    return qMax;
  }

  private addToReplayMemory(): void {
    if (this.learnTick % this.experienceAddEvery === 0) {
      this.memory[this.experienceTick] = {s0: this.s0, a0: this.a0, r0: this.r0, s1: this.s1, a1: this.a1};
      this.experienceTick++;
      if (this.experienceTick > this.experienceSize) {
        this.experienceTick = 0;
      } // roll over when we run out
    }
    this.learnTick++;
  }

  private limitedSampledReplayLearning(): void {
    for (let i = 0; i < this.learningStepsPerIteration; i++) {
      const ri = R.randi(0, this.memory.length); // todo: priority sweeps?
      const sarsa = this.memory[ri];
      this.learnFromTuple(sarsa);
    }
  }
}
