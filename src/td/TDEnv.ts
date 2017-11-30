// import { Env } from "../../src/Env";

// /**
//  * Not yet generalized
//  * Gridworld-TD sample
//  */
// export class TDEnv extends Env {
//   private gw: number; // width
//   private gh: number; // height
  
//   public allowedActions(s:number):Array<number> {
//     const x = this.stox(s);
//     const y = this.stoy(s);
//     const as = new Array<number>();
//     if (x > 0) { as.push(0); }
//     if (y > 0) { as.push(1); }
//     if (y < this.gh - 1) { as.push(2); }
//     if (x < this.gw - 1) { as.push(3); }
//     return as;
//   }

//   // private functions
//   private stox(s) { return Math.floor(s / this.gh); }
//   private stoy(s) { return s % this.gh; }
//   private xytos(x, y) { return x * this.gh + y; }
// }
