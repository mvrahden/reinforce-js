// import { TDEnv } from './TDEnv';
// import { RL } from '../../src/utils/RL';

// import { Solver } from "../../src/Solver";
// import { TDOpt } from "./TDOpt";

// export class TDAgent extends Solver {
//   private pq: Array<number>;
//   private saSeen: Array<number>;
//   private envModelR: Array<number>;
//   private envModelS: Array<number>;
//   private e: Array<number>;
//   private P: Array<number>;
//   private Q: Array<number>;
//   private beta: number;
//   private lambda: number;
//   private planN: number;
//   private qInitVal: number;
//   private explored: boolean;
//   private replacingTraces: boolean;
//   private smoothPolicyUpdate: boolean;
//   private update: string;
//   private a1: number;
//   private s1: number;
//   private r0: number;
//   private a0: number;
//   private s0: number;
  
  
//   private na: number;
//   private ns: number;
//   private alpha: number;
//   private epsilon: number;
//   private gamma: number;
//   private env: TDEnv;
  
  
//   constructor(env:TDEnv, opt:TDOpt) {
//     super();
//     this.update = this.getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
//     this.gamma = this.getopt(opt, 'gamma', 0.75); // future reward discount factor
//     this.epsilon = this.getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
//     this.alpha = this.getopt(opt, 'alpha', 0.01); // value function learning rate

//     // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
//     this.smoothPolicyUpdate = this.getopt(opt, 'smoothPolicyUpdate', false);
//     this.beta = this.getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

//     // eligibility traces
//     this.lambda = this.getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
//     this.replacingTraces = this.getopt(opt, 'replacingTraces', true);

//     // optional optimistic initial values
//     this.qInitVal = this.getopt(opt, 'qInitVal', 0);

//     this.planN = this.getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

//     this.Q = null; // state action value function
//     this.P = null; // policy distribution \pi(s,a)
//     this.e = null; // eligibility trace
//     this.envModelS = null;; // environment model (s,a) -> (s',r)
//     this.envModelR = null;; // environment model (s,a) -> (s',r)
//     this.env = env; // store pointer to environment
//     this.reset();
//   }

//   public reset(): void {
//     // reset the agent's policy and value function
//     this.ns = this.env.getNumStates();
//     this.na = this.env.getMaxNumActions();
//     this.Q = RL.zeros(this.ns * this.na);
//     if (this.qInitVal !== 0) { RL.setConst(this.Q, this.qInitVal); }
//     this.P = RL.zeros(this.ns * this.na);
//     this.e = RL.zeros(this.ns * this.na);

//     // model/planning vars
//     this.envModelS = RL.zeros(this.ns * this.na);
//     RL.setConst(this.envModelS, -1); // init to -1 so we can test if we saw the state before
//     this.envModelR = RL.zeros(this.ns * this.na);
//     this.saSeen = [];
//     this.pq = RL.zeros(this.ns * this.na);

//     // initialize uniform random policy
//     for (let s = 0; s < this.ns; s++) {
//       const poss = this.env.allowedActions(s);
//       for (let i = 0; i < poss.length; i++) {
//         this.P[poss[i] * this.ns + s] = 1.0 / poss.length;
//       }
//     }
//     // agent memory, needed for streaming updates
//     // (s0,a0,r0,s1,a1,r1,...)
//     this.r0 = null;
//     this.s0 = null;
//     this.s1 = null;
//     this.a0 = null;
//     this.a1 = null;
//   }

//   public act(s: Array<number>): number {
//     // act according to epsilon greedy policy
//     // TODO: state comes from Gridworld_td --> Environment??
//     const poss = this.env.allowedActions(s);
//     const probs = [];
//     for (let i = 0; i < poss.length; i++) {
//       probs.push(this.P[poss[i] * this.ns + s]);
//     }
//     // epsilon greedy policy
//     if (Math.random() < this.epsilon) {
//       const a = poss[RL.randi(0, poss.length)]; // random available action
//       this.explored = true;
//     } else {
//       const a = poss[RL.sampleWeighted(probs)];
//       this.explored = false;
//     }
//     // shift state memory
//     this.s0 = this.s1;
//     this.a0 = this.a1;
//     this.s1 = s;
//     this.a1 = a;
//     return a;
//   }

//   public learn(r1: number):void {
//     // takes reward for previous action, which came from a call to act()
//     if (!(this.r0 == null)) {
//       this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
//       if (this.planN > 0) {
//         this.updateModel(this.s0, this.a0, this.r0, this.s1);
//         this.plan();
//       }
//     }
//     this.r0 = r1; // store this for next update
//   }

//   private learnFromTuple(s0:number, a0:number, r0:number, s1:number, a1:number, lambda:number):void {
//     const sa = a0 * this.ns + s0;
//     let target;

//     // calculate the target for Q(s,a)
//     if(this.update === 'qlearn') {
//       // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
//       const poss = this.env.allowedActions(s1);
//       let qmax = 0;
//       for (let i = 0; i < poss.length; i++) {
//         const s1a = poss[i] * this.ns + s1;
//         const qval = this.Q[s1a];
//         if (i === 0 || qval > qmax) { qmax = qval; }
//       }
//       target = r0 + this.gamma * qmax;
//     } else if(this.update === 'sarsa') {
//       // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
//       const s1a1 = a1 * this.ns + s1;
//       target = r0 + this.gamma * this.Q[s1a1];
//     }

//     if(lambda > 0) {
//       // perform an eligibility trace update

//       // TODO: Where does this come from?
//       // if (this.replacing_traces) {
//       if (null) {
//         this.e[sa] = 1;
//       } else {
//         this.e[sa] += 1;
//       }
//       const edecay = lambda * this.gamma;
//       const stateUpdate = RL.zeros(this.ns);
//       for (let s = 0; s < this.ns; s++) {
//         const poss = this.env.allowedActions(s);
//         for (let i = 0; i < poss.length; i++) {
//           const a = poss[i];
//           const saloop = a * this.ns + s;
//           const esa = this.e[saloop];
//           const update = this.alpha * esa * (target - this.Q[saloop]);
//           this.Q[saloop] += update;
//           this.updatePriority(s, a, update);
//           this.e[saloop] *= edecay;
//           const u = Math.abs(update);
//           if (u > stateUpdate[s]) { stateUpdate[s] = u; }
//         }
//       }
//       for (let s = 0; s < this.ns; s++) {
//         if (stateUpdate[s] > 1e-5) { // save efficiency here
//           this.updatePolicy(s);
//         }
//       }
//       if (this.explored && this.update === 'qlearn') {
//         // have to wipe the trace since q learning is off-policy :(
//         this.e = RL.zeros(this.ns * this.na);
//       }
//     } else {
//       // simpler and faster update without eligibility trace
//       // update Q[sa] towards it with some step size
//       const update = this.alpha * (target - this.Q[sa]);
//       this.Q[sa] += update;
//       this.updatePriority(s0, a0, update);
//       // update the policy to reflect the change (if appropriate)
//       this.updatePolicy(s0);
//     }
//   }

//   private updateModel(s0, a0, r0, s1) {
//     // transition (s0,a0) -> (r0,s1) was observed. Update environment model
//     const sa = a0 * this.ns + s0;
//     if (this.envModelS[sa] === -1) {
//       // first time we see this state action
//       this.saSeen.push(a0 * this.ns + s0); // add as seen state
//     }
//     this.envModelS[sa] = s1;
//     this.envModelR[sa] = r0;
//   }
  
//   private plan():void {
//     // order the states based on current priority queue information
//     const spq = [];
//     for(let i = 0; i<this.saSeen.length; i++) {
//       const sa = this.saSeen[i];
//       const sap = this.pq[sa];
//       if (sap > 1e-5) { // gain a bit of efficiency
//         spq.push({ 'sa': sa, 'p': sap });
//       }
//     }
//     spq.sort((a, b) => { return a.p < b.p ? 1 : -1; });

//     // perform the updates
//     const nsteps = Math.min(this.planN, spq.length);
//     for (let k = 0; k < nsteps; k++) {
//       // random exploration
//       //let i = randi(0, this.sa_seen.length); // pick random prev seen state action
//       //let s0a0 = this.sa_seen[i];
//       const s0a0 = spq[k].sa;
//       this.pq[s0a0] = 0; // erase priority, since we're backing up this state
//       const s0 = s0a0 % this.ns;
//       const a0 = Math.floor(s0a0 / this.ns);
//       const r0 = this.envModelR[s0a0];
//       const s1 = this.envModelS[s0a0];
//       const a1 = -1; // not used for Q learning
//       if (this.update === 'sarsa') {
//         // generate random action?...
//         const poss = this.env.allowedActions(s1);
//         const a1 = poss[RL.randi(0, poss.length)];
//       }
//       this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
//     }
//   }

//   private updatePriority(s, a, u) {
//     // used in planning. Invoked when Q[sa] += update
//     // we should find all states that lead to (s,a) and upgrade their priority
//     // of being update in the next planning step
//     u = Math.abs(u);
//     if(u <1e-5) { return; } // for efficiency skip small updates
//     if(this.planN === 0) { return; } // there is no planning to be done, skip.
//     for(let si = 0; si<this.ns; si++) {
//       // note we are also iterating over impossible actions at all states,
//       // but this should be okay because their env_model_s should simply be -1
//       // as initialized, so they will never be predicted to point to any state
//       // because they will never be observed, and hence never be added to the model
//       for (let ai = 0; ai < this.na; ai++) {
//         const siai = ai * this.ns + si;
//         if (this.envModelS[siai] === s) {
//           // this state leads to s, add it to priority queue
//           this.pq[siai] += u;
//         }
//       }
//     }
//   }

//   private updatePolicy(s):void {
//     const poss = this.env.allowedActions(s);
//     // set policy at s to be the action that achieves max_a Q(s,a)
//     // first find the maxy Q values
//     let qmax, nmax;
//     const qs = [];
//     for(let i = 0; i<poss.length; i++) {
//       const a = poss[i];
//       const qval = this.Q[a * this.ns + s];
//       qs.push(qval);
//       if (i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
//       else if (qval === qmax) { nmax += 1; }
//     }
//     // now update the policy smoothly towards the argmaxy actions
//     let psum = 0.0;
//     for (let i = 0; i < poss.length; i++) {
//       const a = poss[i];
//       const target = (qs[i] === qmax) ? 1.0 / nmax : 0.0;
//       const ix = a * this.ns + s;
//       if (this.smoothPolicyUpdate) {
//         // slightly hacky :p
//         this.P[ix] += this.beta * (target - this.P[ix]);
//         psum += this.P[ix];
//       } else {
//         // set hard target
//         this.P[ix] = target;
//       }
//     }
//     if (this.smoothPolicyUpdate) {
//       // renomalize P if we're using smooth policy updates
//       for (let i = 0; i < poss.length; i++) {
//         const a = poss[i];
//         this.P[a * this.ns + s] /= psum;
//       }
//     }
//   }


//   public toJSON(): object {
//     throw new Error('Not implemented yet.');
//   }

//   public fromJSON(json: {}): void {
//     throw new Error('Not implemented yet.');
//   }
// }
