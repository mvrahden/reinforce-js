import { Opt } from '../Opt';

export class DQNOpt extends Opt {
  protected trainingMode: boolean = true;
  protected numberOfHiddenUnits: Array<number> = [ 100 ];
  protected epsilonMax: number = 1.0;
  protected epsilonMin: number = 0.1;
  protected epsilonDecayPeriod: number = 1e6;
  protected epsilon: number = 0.05;
  
  protected gamma: number = 0.9;
  protected alpha: number = 0.01;
  protected experienceSize: number = 1e6;
  protected doLossClipping: boolean = true;
  protected lossClamp: number = 1.0;
  protected doRewardClipping: boolean = true;
  protected rewardClamp: number = 1.0;
  
  protected keepExperienceInterval: number = 25;
  protected replaySteps: number = 10;

  /**
   * Sets the number of neurons in hidden layers
   * @param numberOfHiddenUnits defaults to [ 100 ]
   */
  public setNumberOfHiddenUnits(numberOfHiddenUnits: Array<number>): void {
    this.numberOfHiddenUnits = numberOfHiddenUnits;
  }

  /**
   * Defines a linear annealing of Epsilon for an Epsilon Greedy Policy during 'training' = true
   * @param epsilonMax upper bound of epsilon; defaults to 1.0
   * @param epsilonMin lower bound of epsilon; defaults to 0.1
   * @param epsilonDecayPeriod number of timesteps; defaults to 1e6
   */
  public setEpsilonDecay(epsilonMax: number, epsilonMin: number, epsilonDecayPeriod: number): void {
    this.epsilonMax = epsilonMax;
    this.epsilonMin = epsilonMin;
    this.epsilonDecayPeriod = epsilonDecayPeriod;
  }

  /**
   * Sets the Epsilon Factor (Exploration Factor or Greedy Policy) during 'training' = false
   * @param epsilon value from [0,1); defaults to 0.05
   */
  public setEpsilon(epsilon: number): void {
    this.epsilon = epsilon;
  }

  /**
   * Sets the Future Reward Discount Factor
   * @param gamma value from [0,1); defaults to 0.9
   */
  public setGamma(gamma: number): void {
    this.gamma = gamma;
  }

  /**
   * Sets the Value Function Learning Rate
   * @param alpha defaults to 0.01
   */
  public setAlpha(alpha: number): void {
    this.alpha = alpha;
  }

  /**
   * Activates or deactivates the Reward clipping to -1 or +1.
   * @param doLossClipping defaults to true (active)
   */
  public setLossClipping(doLossClipping: boolean): void {
    this.doLossClipping = doLossClipping;
  }

  /**
   * Sets the loss clamp for robustness
   * @param lossClamp defaults to 1.0
   */
  public setLossClamp(lossClamp: number): void {
    this.lossClamp = lossClamp;
  }

  /**
   * Activates or deactivates the Reward clipping to -1 or +1.
   * @param doRewardClipping defaults to true (active)
   */
  public setRewardClipping(doRewardClipping: boolean): void {
    this.doRewardClipping = doRewardClipping;
  }

  /**
   * Activates or deactivates the Reward clipping to -1 or +1.
   * @param rewardClamp defaults to 1.0
   */
  public setRewardClamp(rewardClamp: number): void {
    this.rewardClamp = rewardClamp;
  }

  /**
   * Activates or deactivated the Training Mode of the Solver.
   * @param trainingMode defaults to true (active)
   */
  public setTrainingMode(trainingMode: boolean): void {
    this.trainingMode = trainingMode;
  }

  /**
   * Sets Replay Memory Size
   * @param experienceSize defaults to 1e6
   */
  public setExperienceSize(experienceSize: number): void {
    this.experienceSize = experienceSize;
  }

  /**
   * Sets the amount of time steps before another experience is added to replay memory
   * @param keepExperienceInterval defaults to 25
   */
  public setReplayInterval(keepExperienceInterval: number): void {
    this.keepExperienceInterval = keepExperienceInterval;
  }

  /**
   * Sets the amount of memory replays per iteration
   * @param replaySteps defaults to 10
   */
  public setReplaySteps(replaySteps: number): void {
    this.replaySteps = replaySteps;
  }
}
