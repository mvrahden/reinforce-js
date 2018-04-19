import { Utils } from 'recurrent-js';

import { Solver } from './../Solver';
import { TDEnv } from './TDEnv';
import { TDOpt } from './TDOpt';

export class TDSolver extends Solver {
  protected readonly alpha: number;
  protected readonly epsilon: number;
  protected readonly gamma: number;

  protected readonly beta: number;
  protected readonly lambda: number;
  protected readonly numberOfPlanningSteps: number;
  protected readonly update: string;
  protected readonly qInitValue: number;
  protected readonly updateSmoothPolicy: boolean;
  protected readonly replacingTraces: boolean;

  protected numberOfActions: number;
  protected numberOfStates: number;

  protected pq: Array<number> | Float64Array;
  protected saSeen: Array<number>;
  protected envModelR: Array<number> | Float64Array;
  protected envModelS: Array<number> | Float64Array;
  protected eligibilityTraces: Array<number> | Float64Array;
  protected randomPolicies: Array<number> | Float64Array;
  protected Q: Array<number> | Float64Array;
  protected explored: boolean;
  protected a1: number;
  protected s1: number;
  protected r0: number;
  protected a0: number;
  protected s0: number;

  protected env: TDEnv;


  constructor(env: TDEnv, opt: TDOpt) {
    super(env, opt);
    this.env = env;
    this.alpha = opt.get('alpha');
    this.epsilon = opt.get('epsilon');
    this.gamma = opt.get('gamma');

    this.update = opt.get('update');

    this.updateSmoothPolicy = opt.get('smoothPolicyUpdate');
    this.beta = opt.get('beta');

    this.lambda = opt.get('lambda');
    this.replacingTraces = opt.get('replacingTraces');

    this.qInitValue = opt.get('qInitVal');
    this.numberOfPlanningSteps = opt.get('numberOfPlanningSteps');

    this.Q = null; // state action value function
    this.randomPolicies = null; // policy distribution \pi(s,a)
    this.eligibilityTraces = null; // eligibility trace
    this.envModelS = null;; // environment model (s,a) -> (s',r)
    this.envModelR = null;; // environment model (s,a) -> (s',r)
    this.env = env; // store pointer to environment

    this.reset();
  }

  public reset(): void {
    // reset the agent's policy and value function
    this.numberOfStates = this.env.get('numberOfStates');
    this.numberOfActions = this.env.get('numerOfActions');
    this.Q = Utils.zeros(this.numberOfStates * this.numberOfActions);
    if (this.qInitValue !== 0) { Utils.fillConst(this.Q, this.qInitValue); }
    this.randomPolicies = Utils.zeros(this.numberOfStates * this.numberOfActions);
    this.eligibilityTraces = Utils.zeros(this.numberOfStates * this.numberOfActions);

    // model/planning vars
    this.envModelS = Utils.zeros(this.numberOfStates * this.numberOfActions);
    Utils.fillConst(this.envModelS, -1); // init to -1 so we can test if we saw the state before
    this.envModelR = Utils.zeros(this.numberOfStates * this.numberOfActions);
    this.saSeen = [];
    this.pq = Utils.zeros(this.numberOfStates * this.numberOfActions);

    // initialize uniform random policy
    for (let state = 0; state < this.numberOfStates; state++) {
      const allowedActions = this.env.allowedActions(state);
      for (let i = 0; i < allowedActions.length; i++) {
        this.randomPolicies[allowedActions[i] * this.numberOfStates + state] = 1.0 / allowedActions.length;
      }
    }

    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  }

  /**
   * Decide an action according to current state
   * @param state current state
   * @returns decided action
   */
  public decide(state: number): number {
    // act according to epsilon greedy policy
    // TODO: state comes from Gridworld_td --> Environment??
    const allowedActions = this.env.allowedActions(state);
    const probs = new Array<number>();

    for (let i = 0; i < allowedActions.length; i++) {
      probs.push(this.randomPolicies[allowedActions[i] * this.numberOfStates + state]);
    }

    const actionIndex = this.epsilonGreedyActionPolicy(allowedActions, probs);
    
    // shift state memory
    this.shiftStateMemory(state, actionIndex);

    return actionIndex;
  }

  private shiftStateMemory(state: number, actionIndex: number) {
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = state;
    this.a1 = actionIndex;
  }

  private epsilonGreedyActionPolicy(poss: number[], probs: number[]) {
    let actionIndex: number = 0;
    if (Math.random() < this.epsilon) {
      actionIndex = poss[Utils.randi(0, poss.length)]; // random available action
      this.explored = true;
    }
    else {
      actionIndex = poss[Utils.sampleWeighted(probs)];
      this.explored = false;
    }
    return actionIndex;
  }

  public learn(r1: number): void {
    // takes reward for previous action, which came from a call to act()
    if (!(this.r0 == null)) {
      this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
      if (this.numberOfPlanningSteps > 0) {
        this.updateModel(this.s0, this.a0, this.r0, this.s1);
        this.plan();
      }
    }
    this.r0 = r1; // store this for next update
  }

  private learnFromTuple(s0: number, a0: number, r0: number, s1: number, a1: number, lambda: number): void {
    const sa = a0 * this.numberOfStates + s0;
    let target;

    // calculate the target for Q(s,a)
    if (this.update === 'qlearn') {
      // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
      const poss = this.env.allowedActions(s1);
      let qmax = 0;
      for (let i = 0; i < poss.length; i++) {
        const s1a = poss[i] * this.numberOfStates + s1;
        const qval = this.Q[s1a];
        if (i === 0 || qval > qmax) { qmax = qval; }
      }
      target = r0 + this.gamma * qmax;
    } else if (this.update === 'sarsa') {
      // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
      const s1a1 = a1 * this.numberOfStates + s1;
      target = r0 + this.gamma * this.Q[s1a1];
    }

    if (lambda > 0) {
      // perform an eligibility trace update

      // TODO: Where does this come from?
      // if (this.replacing_traces) {
      if (null) {
        this.eligibilityTraces[sa] = 1;
      } else {
        this.eligibilityTraces[sa] += 1;
      }
      const decay = lambda * this.gamma;
      const stateUpdate = Utils.zeros(this.numberOfStates);
      for (let s = 0; s < this.numberOfStates; s++) {
        const poss = this.env.allowedActions(s);
        for (let i = 0; i < poss.length; i++) {
          const a = poss[i];
          const saloop = a * this.numberOfStates + s;
          const esa = this.eligibilityTraces[saloop];
          const update = this.alpha * esa * (target - this.Q[saloop]);
          this.Q[saloop] += update;
          this.updatePriority(s, a, update);
          this.eligibilityTraces[saloop] *= decay;
          const u = Math.abs(update);
          if (u > stateUpdate[s]) { stateUpdate[s] = u; }
        }
      }
      for (let s = 0; s < this.numberOfStates; s++) {
        if (stateUpdate[s] > 1e-5) { // save efficiency here
          this.updatePolicy(s);
        }
      }
      if (this.explored && this.update === 'qlearn') {
        // have to wipe the trace since q learning is off-policy :(
        this.eligibilityTraces = Utils.zeros(this.numberOfStates * this.numberOfActions);
      }
    } else {
      // simpler and faster update without eligibility trace
      // update Q[sa] towards it with some step size
      const update = this.alpha * (target - this.Q[sa]);
      this.Q[sa] += update;
      this.updatePriority(s0, a0, update);
      // update the policy to reflect the change (if appropriate)
      this.updatePolicy(s0);
    }
  }

  private updateModel(s0, a0, r0, s1): void {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    const sa = a0 * this.numberOfStates + s0;
    if (this.envModelS[sa] === -1) {
      // first time we see this state action
      this.saSeen.push(a0 * this.numberOfStates + s0); // add as seen state
    }
    this.envModelS[sa] = s1;
    this.envModelR[sa] = r0;
  }

  private plan(): void {
    // order the states based on current priority queue information
    const spq = [];
    for (let i = 0; i < this.saSeen.length; i++) {
      const sa = this.saSeen[i];
      const sap = this.pq[sa];
      if (sap > 1e-5) { // gain a bit of efficiency
        spq.push({ 'sa': sa, 'p': sap });
      }
    }
    spq.sort((a, b) => { return a.p < b.p ? 1 : -1; });

    // perform the updates
    const nsteps = Math.min(this.numberOfPlanningSteps, spq.length);
    for (let k = 0; k < nsteps; k++) {
      // random exploration
      //let i = randi(0, this.sa_seen.length); // pick random prev seen state action
      //let s0a0 = this.sa_seen[i];
      const s0a0 = spq[k].sa;
      this.pq[s0a0] = 0; // erase priority, since we're backing up this state
      const s0 = s0a0 % this.numberOfStates;
      const a0 = Math.floor(s0a0 / this.numberOfStates);
      const r0 = this.envModelR[s0a0];
      const s1 = this.envModelS[s0a0];
      const a1 = -1; // not used for Q learning
      if (this.update === 'sarsa') {
        // generate random action?...
        const poss = this.env.allowedActions(s1);
        const a1 = poss[Utils.randi(0, poss.length)];
      }
      this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
    }
  }

  private updatePriority(s, a, u): void {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u);
    if (u < 1e-5) { return; } // for efficiency skip small updates
    if (this.numberOfPlanningSteps === 0) { return; } // there is no planning to be done, skip.
    for (let si = 0; si < this.numberOfStates; si++) {
      // note we are also iterating over impossible actions at all states,
      // but this should be okay because their env_model_s should simply be -1
      // as initialized, so they will never be predicted to point to any state
      // because they will never be observed, and hence never be added to the model
      for (let ai = 0; ai < this.numberOfActions; ai++) {
        const siai = ai * this.numberOfStates + si;
        if (this.envModelS[siai] === s) {
          // this state leads to s, add it to priority queue
          this.pq[siai] += u;
        }
      }
    }
  }

  private updatePolicy(s): void {
    const poss = this.env.allowedActions(s);
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    let qmax, nmax;
    const qs = [];
    for (let i = 0; i < poss.length; i++) {
      const a = poss[i];
      const qval = this.Q[a * this.numberOfStates + s];
      qs.push(qval);
      if (i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
      else if (qval === qmax) { nmax += 1; }
    }
    // now update the policy smoothly towards the argmaxy actions
    let psum = 0.0;
    for (let i = 0; i < poss.length; i++) {
      const a = poss[i];
      const target = (qs[i] === qmax) ? 1.0 / nmax : 0.0;
      const ix = a * this.numberOfStates + s;
      if (this.updateSmoothPolicy) {
        // slightly hacky :p
        this.randomPolicies[ix] += this.beta * (target - this.randomPolicies[ix]);
        psum += this.randomPolicies[ix];
      } else {
        // set hard target
        this.randomPolicies[ix] = target;
      }
    }
    if (this.updateSmoothPolicy) {
      // renomalize P if we're using smooth policy updates
      for (let i = 0; i < poss.length; i++) {
        const a = poss[i];
        this.randomPolicies[a * this.numberOfStates + s] /= psum;
      }
    }
  }


  public toJSON(): object {
    throw new Error('Not implemented yet.');
  }

  public fromJSON(json: {}): void {
    throw new Error('Not implemented yet.');
  }

}
