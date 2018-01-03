import { Net, Graph, Mat, RandMat, R } from 'recurrent-js';

import { Solver } from '../Solver';
import { Env } from '../Env';
import { DQNOpt } from './DQNOpt';

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
  public net: Net;
  public previousGraph: Graph;
  public a1: number | null = null;
  public a0: number | null = null;
  public s1: Mat | null = null;
  public s0: Mat | null = null;
  public r0: number | null = null;
  public learnTick: number;
  public experienceTick: number;
  public experience: Array<any>;
  public tderror: number;

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
    this.numberOfStates = this.env.get('numStates');
    this.numberOfActions = this.env.get('maxNumActions');

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    const netOpts = {
      inputSize: this.numberOfStates,
      hiddenSize: this.numberOfHiddenUnits,
      outputSize: this.numberOfActions
    };
    this.net = new Net(netOpts);

    this.experience = []; // experience
    this.experienceTick = 0; // where to insert
    this.learnTick = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
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

  private epsilonGreedyActionPolicy(s: Mat): number {
    let actionIndex: number = 0;
    if (Math.random() < this.epsilon) {
      actionIndex = R.randi(0, this.numberOfActions);
    }
    else {
      // greedy wrt Q function
      const actionMat = this.forwardQ(s, false);
      actionIndex = R.maxi(actionMat.w); // returns index of argmax action
    }
    return actionIndex;
  }

  /**
   * Determine Outputs based on Forward Pass
   * @param net Network
   * @param s Matrix with states
   * @param needsBackprop 
   * @return Matrix with predicted actions values
   */
  private forwardQ(s: Mat | null, needsBackprop: boolean): Mat {
    const graph = new Graph(needsBackprop);
    const a1mat = graph.add(graph.mul(this.net.W1, s), this.net.b1);
    const h1mat = graph.tanh(a1mat);
    const a2Mat = graph.add(graph.mul(this.net.W2, h1mat), this.net.b2);
    this.backupGraph(graph); // back this up. Kind of hacky isn't it
    return a2Mat;
  }

  private backupGraph(graph: Graph): void {
    this.previousGraph = graph;
  }

  private shiftStateMemory(s: Mat, actionIndex: number): void {
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s; // add new state
    this.a1 = actionIndex;
  }

  /**
   * perform an update on Q function
   * @param r1 reward passed to learn
   */
  public learn(r1: number): void {
    if (!(this.r0 == null) && this.alpha > 0) {

      // learn from this tuple to get a sense of how "surprising" it is to the agent
      this.tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1); // a measure of surprise

      // decide: keep this experience in the replay memory?
      this.replayMemoryGate();

      // sample some additional experience from replay memory and learn from it
      this.sampledReplayLearning();
    }
    this.r0 = r1; // store reward for next update
  }

  private learnFromTuple(s0: Mat | null, a0: number, r0: number, s1: Mat | null, a1: number | null): number {

    const qMax = this.getTargetQ(s1, r0);
    const predictor = this.forwardQ(s0, true);
    let tdError = predictor.w[a0] - qMax;

    if (Math.abs(tdError) > this.tdErrorClamp) {  // huber loss to robustify
      if (tdError > this.tdErrorClamp) {
        tdError = this.tdErrorClamp;
      }
      if (tdError < -this.tdErrorClamp) {
        tdError = -this.tdErrorClamp;
      }
    }
    predictor.dw[a0] = tdError;
    this.previousGraph.backward();

    // update net
    this.net.update(this.alpha);
    return tdError;
  }

  private getTargetQ(s1: Mat, r0: number): number {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')
    const tMat = this.forwardQ(s1, false);
    const qMax = r0 + this.gamma * tMat.w[R.maxi(tMat.w)];
    return qMax;
  }

  private replayMemoryGate(): void {
    if (this.learnTick % this.experienceAddEvery === 0) {
      this.experience[this.experienceTick] = [this.s0, this.a0, this.r0, this.s1, this.a1];
      this.experienceTick++;
      if (this.experienceTick > this.experienceSize) {
        this.experienceTick = 0;
      } // roll over when we run out
    }
    this.learnTick++;
  }

  private sampledReplayLearning(): void {
    for (let i = 0; i < this.learningStepsPerIteration; i++) {
      const ri = R.randi(0, this.experience.length); // todo: priority sweeps?
      const e = this.experience[ri];
      this.learnFromTuple(e[0], e[1], e[2], e[3], e[4]);
    }
  }
}
