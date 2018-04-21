import { Env, Opt } from './.';

export abstract class Solver {
  protected env: Env;
  protected opt: Opt;

  constructor(env: Env, opt: Opt) {
    this.env = env;
    this.opt = opt;
  }

  public getOpt(): any {
    return this.opt;
  }

  public getEnv(): any {
    return this.env;
  }

  /**
   * Decide an action according to current state
   * @param state current state
   * @returns decided action
   */
  public abstract decide(stateList: any): number;
  public abstract learn(r1: number): void;
  public abstract reset(): void;
  public abstract toJSON(): object;
  public abstract fromJSON(json: {}): void;
}
