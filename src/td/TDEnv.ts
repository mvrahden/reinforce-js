import { Env } from "../Env";

/**
 * Not yet generalized
 * Gridworld-TD sample
 */
export class TDEnv extends Env {

  public allowedActions(s: number): Array<number> {
    const x = this.stox(s);
    const y = this.stoy(s);
    const allowedActions = new Array<number>();
    if (x > 0) { allowedActions.push(0); }
    if (y > 0) { allowedActions.push(1); }
    if (y < this.height - 1) { allowedActions.push(2); }
    if (x < this.width - 1) { allowedActions.push(3); }
    return allowedActions;
  }

  // private functions
  private stox(s: number): number { return Math.floor(s / this.height); }
  private stoy(s: number): number { return s % this.height; }
  // private xytos(x, y): number { return x * this.height + y; }
}
