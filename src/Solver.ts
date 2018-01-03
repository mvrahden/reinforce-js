import { Env } from './Env';
import { Opt } from './Opt';

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

  public abstract act(stateList: any): number;
  public abstract learn(r1: number): void;
  public abstract reset(): void;
  public abstract toJSON(): object;
  public abstract fromJSON(json: {}): void;
}
