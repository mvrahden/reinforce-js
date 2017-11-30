import { Net, Graph, Mat, RandMat, R } from 'recurrent-ts';

import { Solver } from '../Solver';
import { Env } from '../Env';
import { Opt } from '../Opt';

export class DQNSolver extends Solver {
  protected net: Net;
  protected lastG: Graph;
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
  
  protected ns: number;
  protected na: number;
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
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    this.net = new Net();
    this.net.W1 = new RandMat(this.nh, this.ns, 0, 0.01);
    this.net.b1 = new Mat(this.nh, 1);
    this.net.W2 = new RandMat(this.na, this.nh, 0, 0.01);
    this.net.b2 = new Mat(this.na, 1);

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
      ns: this.ns,
      na: this.na,
      net: Net.toJSON(this.net)
    };
    return j;
  }

  public fromJSON (json:{nh, ns, na, net}):void {
    // load function
    this.nh = json.nh;
    this.ns = json.ns;
    this.na = json.na;
    this.net = Net.fromJSON(json.net);
  }

  /**
   * 
   * @param stateList 
   * @returns Index of argmax action
   */
  public act (stateList:Array<number>):number {
    // convert to a Mat column vector
    const s = new Mat(this.ns, 1);
    s.setFrom(stateList);

    // epsilon greedy policy
    let a = 0;
    if(Math.random() < this.epsilon) {
      a = R.randi(0, this.na);
    } else {
      // greedy wrt Q function
      const amat = this.forwardQ(this.net, s, false);
      a = R.maxi(amat.w); // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  }

    private forwardQ(net: Net, s: Mat | null, needsBackprop: boolean) {
      const graph = new Graph(needsBackprop);
      const a1mat = graph.add(graph.mul(net.W1, s), net.b1);
      const h1mat = graph.tanh(a1mat);
      const a2mat = graph.add(graph.mul(net.W2, h1mat), net.b2);
      this.lastG = graph; // back this up. Kind of hacky isn't it
      return a2mat;
    }
  
  /**
   * perform an update on Q function
   * @param r1 
   */
  public learn (r1:number):void {
    if(!(this.r0 == null) && this.alpha > 0) {

      // learn from this tuple to get a sense of how "surprising" it is to the agent
      const tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
      this.tderror = tderror; // a measure of surprise

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

    private learnFromTuple (s0:Mat | null, a0:number, r0:number, s1:Mat | null, a1:number | null) {
      // want: Q(s,a) = r + gamma * max_a' Q(s',a')

      // compute the target Q value
      const tmat = this.forwardQ(this.net, s1, false);
      const qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

      // now predict
      const pred = this.forwardQ(this.net, s0, true);

      let tderror = pred.w[a0] - qmax;
      const clamp = this.tdErrorClamp;
      if(Math.abs(tderror) > clamp) {  // huber loss to robustify
        if (tderror > clamp) {
          tderror = clamp;
        }
        if (tderror < -clamp) {
          tderror = -clamp;
        }
      }
        pred.dw[a0] = tderror;
      this.lastG.backward(); // compute gradients on net params

      // update net
      Net.update(this.net, this.alpha);
      return tderror;
    }
}
