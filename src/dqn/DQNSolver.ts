import { Net, Graph, Mat, RandMat, R } from 'recurrent-js';

import { Solver } from '../Solver';
import { Env } from '../Env';
import { Opt } from '../Opt';

export class DQNSolver extends Solver {
  protected net: Net;
  protected lastGraph: Graph;
  protected a1: number | null = null;
  protected a0: number | null = null;
  protected s1: Mat | null = null;
  protected s0: Mat | null = null;
  protected r0: number | null = null;
  protected t: number;
  protected expi: number;
  protected exp: Array<any>;
  protected tderror: number;
  protected numHiddenUnits: number;
  protected nh: number;
  protected tdErrorClamp: number;
  protected learningStepsPerIteration: number;
  protected experienceSize: number;
  protected experienceAddEvery: number;
  
  protected numberOfStates: number;
  protected numberOfMaxActions: number;

  protected alpha: number;
  protected epsilon: number;
  protected gamma: number;
  protected env: Env;

  constructor(env: Env, opt: Opt) {
    super();
    this.gamma = this.getopt(opt, 'gamma', 0.75); // future reward discount factor
    this.epsilon = this.getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
    this.alpha = this.getopt(opt, 'alpha', 0.01); // value function learning rate

    this.experienceAddEvery = this.getopt(opt, 'experienceAddEvery', 25); // number of time steps before we add another experience to replay memory
    this.experienceSize = this.getopt(opt, 'experienceSize', 5000); // size of experience replay
    this.learningStepsPerIteration = this.getopt(opt, 'learningStepsPerIteration', 10);
    this.tdErrorClamp = this.getopt(opt, 'TDErrorClamp', 1.0);

    this.numHiddenUnits = this.getopt(opt, 'numHiddenUnits', 100);

    this.env = env;
    this.reset();
  }

  public reset():void {
    this.nh = this.numHiddenUnits; // number of hidden units
    this.numberOfStates = this.env.getNumStates();
    this.numberOfMaxActions = this.env.getMaxNumActions();

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    this.net = new Net();
    this.net.W1 = new RandMat(this.nh, this.numberOfStates, 0, 0.01);
    this.net.b1 = new Mat(this.nh, 1);
    this.net.W2 = new RandMat(this.numberOfMaxActions, this.nh, 0, 0.01);
    this.net.b2 = new Mat(this.numberOfMaxActions, 1);

    this.exp = []; // experience
    this.expi = 0; // where to insert

    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;

    this.tderror = 0; // for visualization only...
  }

  public toJSON ():object {
    // save function
    const j = {
      nh: this.nh,
      ns: this.numberOfStates,
      na: this.numberOfMaxActions,
      net: Net.toJSON(this.net)
    };
    return j;
  }

  public fromJSON (json:{nh, ns, na, net}):void {
    // load function
    this.nh = json.nh;
    this.numberOfStates = json.ns;
    this.numberOfMaxActions = json.na;
    this.net = Net.fromJSON(json.net);
  }

  /**
   * 
   * @param stateList 
   * @returns Index of argmax action
   */
  public act (stateList:Array<number>):number {
    // convert to a Mat column vector
    const s = new Mat(this.numberOfStates, 1);
    s.setFrom(stateList);

    // epsilon greedy policy
    const actionIndex: number = this.greedyActionPolicy(s);

    this.shiftStateMemory(s, actionIndex);

    return actionIndex;
  }

    private greedyActionPolicy(s: Mat): number {
      let actionIndex: number = 0;
      if (Math.random() < this.epsilon) {
        actionIndex = R.randi(0, this.numberOfMaxActions);
      }
      else {
        // greedy wrt Q function
        const actionMat = this.forwardQ(s, false);
        actionIndex = R.maxi(actionMat.w); // returns index of argmax action
      }
      return actionIndex;
    }

    /**
     * Determine Outputs based on Forward Feed
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

    private backupGraph(graph: Graph) {
      this.lastGraph = graph;
    }

    private shiftStateMemory(s: Mat, actionIndex: number) {
      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = s; // add new state
      this.a1 = actionIndex;
    }
  
  /**
   * perform an update on Q function
   * @param r1 
   */
  public learn (r1:number): void {
    if(!(this.r0 == null) && this.alpha > 0) {

      // learn from this tuple to get a sense of how "surprising" it is to the agent
      this.tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1); // a measure of surprise

      // decide if we should keep this experience in the replay
      if (this.t % this.experienceAddEvery === 0) {
        this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
        this.expi += 1;
        if (this.expi > this.experienceSize) { this.expi = 0; } // roll over when we run out
      }
      this.t += 1;

      // sample some additional experience from replay memory and learn from it
      for (let k = 0; k < this.learningStepsPerIteration; k++) {
        const ri = R.randi(0, this.exp.length); // todo: priority sweeps?
        const e = this.exp[ri];
        this.learnFromTuple(e[0], e[1], e[2], e[3], e[4]);
      }
    }
    this.r0 = r1; // store for next update
  }

    private learnFromTuple (s0:Mat | null, a0:number, r0:number, s1:Mat | null, a1:number | null): number {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')

      // compute the target Q value
      const qmax = this.getTargetQ(s1, r0);

      // now predict
      const pred = this.forwardQ(s0, true);

      let tderror = pred.w[a0] - qmax;

      if(Math.abs(tderror) > this.tdErrorClamp) {  // huber loss to robustify
        if (tderror > this.tdErrorClamp) {
          tderror = this.tdErrorClamp;
        }
        if (tderror < -this.tdErrorClamp) {
          tderror = -this.tdErrorClamp;
        }
      }
      pred.dw[a0] = tderror;
      this.lastGraph.backward(); // compute gradients on net params

      // update net
      Net.update(this.net, this.alpha);
      return tderror;
    }
  
  private getTargetQ(s1: Mat, r0: number) {
    const tmat = this.forwardQ(s1, false);
    const qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];
    return qmax;
  }

  public getTDError(): number {
    return this.tderror;
  }
}
