import { Opt } from './../.';

export class TDOpt extends Opt {
  protected alpha: number = 0.01;
  protected epsilon: number = 0.1;
  protected gamma: number = 0.75;

  protected beta: number = 0.01;
  protected lambda: number = 0;
  protected qInitValue: number = 0;
  protected replacingTraces: boolean = true;
  protected smoothPolicyUpdate: boolean = false;
  protected numberOfPlanningSteps: number = 0;
  protected update: string = 'qlearn';


  /**
   * Sets the Value Function Learning Rate
   * @param alpha defaults to 0.01
   */
  public setAlpha(alpha: number): void {
    this.alpha = alpha;
  }

  /**
   * Sets the Epsilon Factor (Exploration Factor or Greedy Policy)
   * @param epsilon value from [0,1); defaults to 0.1
   */
  public setEpsilon(epsilon: number): void {
    this.epsilon = epsilon;
  }

  /**
   * Sets the Future Reward Discount Factor
   * @param gamma value from [0,1); defaults to 0.75
   */
  public setGamma(gamma: number): void {
    this.gamma = gamma;
  }

  /**
   * Sets the Function Learning Rate for Smooth Policy if activated Smooth Policy Updates
   * @param beta value from [0,1); defaults to 0.01
   */
  public setBeta(beta: number): void {
    this.beta = beta;
  }

  /**
   * Sets the eligibility trace decay. 0 = no eligibility trace
   * @param lambda defaults to 0
   */
  public setLambda(lambda: number): void {
    this.lambda = lambda;
  }

  /**
   * Sets an initial optimistic value for state action value function Q (Array(States*Actions)) 
   * @param qInitValue defaults to 0
   */
  public setQ(qInitValue: number): void {
    this.qInitValue = qInitValue;
  }

  /**
   * Set Trace Replacing
   * @param replacingTraces defaults to true
   */
  public setReplacingTraces(replacingTraces: boolean): void {
    this.replacingTraces = replacingTraces;
  }

  /**
   * Sets Smooth Policy Updates
   * @param smoothPolicyUpdate defaults to false
   */
  public setSmoothPolicyUpdate(smoothPolicyUpdate: boolean): void {
    this.smoothPolicyUpdate = smoothPolicyUpdate;
  }

  /**
   * Sets the number of Planning Steps
   * @param numberOfPlanningSteps value from [0,1); defaults to 0.01
   */
  public setNumberOfPlanningSteps(numberOfPlanningSteps: number): void {
    this.numberOfPlanningSteps = numberOfPlanningSteps;
  }

  /**
   * Sets the update target function.
   * @param update either 'qlearn' or 'sarsa'; defaults to 'qlearn'
   */
  public setUpdate(update: string): void {
    this.update = update;
  }

}
