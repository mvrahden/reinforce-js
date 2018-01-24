import { Opt } from '../Opt';

export class DQNOpt extends Opt {
  protected trainingMode: boolean = true;
  protected numberOfHiddenUnits: number = 100;
  protected epsilon: number = 0.05;
  protected epsilonMax: number = 1.0;
  protected epsilonMin: number = 0.1;
  protected epsilonDecayPeriod: number = 1e6;
  
  protected gamma: number = 0.9;
  protected alpha: number = 0.01;
  protected experienceSize: number = 1e6;
  protected doLossClipping: boolean = true;
  protected delta: number = 1.0;
  protected doRewardClipping: boolean = true;
  protected rewardClamp: number = 1.0;
  
  protected replayInterval: number = 25;
  protected replaySteps: number = 10;

  /**
   * Sets the Value Function Learning Rate
   * @param alpha defaults to 0.005
   */
  public setAlpha(alpha: number): void {
    this.alpha = alpha;
  }

  /**
   * Sets the Epsilon Factor (Exploration Factor or Greedy Policy) during 'training' = false
   * @param epsilon value from [0,1); defaults to 0.2
   */
  public setTrainingMode(trainingMode: boolean): void {
    this.trainingMode = trainingMode;
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
   * @param epsilon value from [0,1); defaults to 0.2
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
   * Sets the Number of Time Steps before another Experience is added to Replay Memory
   * @param replayInterval defaults to 25
   */
  public setExperienceAddEvery(replayInterval: number): void {
    this.replayInterval = replayInterval;
  }

  /**
   * Sets Replay Memory Size
   * @param experienceSize defaults to 1000000
   */
  public setExperienceSize(experienceSize: number): void {
    this.experienceSize = experienceSize;
  }

  /**
   * Sets the learning steps per iteration
   * @param replaySteps defaults to 10
   */
  public setLearningStepsPerIteration(replaySteps: number): void {
    this.replaySteps = replaySteps;
  }

  /**
   * Sets the delta (loss clamp) of the huber loss function for robustness
   * @param delta defaults to 1.0
   */
  public setDelta(delta: number): void {
    this.delta = delta;
  }

  /**
   * Sets the number of neurons in hidden layer
   * @param numberOfHiddenUnits defaults to 100
   */
  public setNumberOfHiddenUnits(numberOfHiddenUnits: number): void {
    this.numberOfHiddenUnits = numberOfHiddenUnits;
  }

  /**
   * Activates or deactivates the Reward clipping to -1 or +1.
   * @param doRewardClipping defaults to true
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
}
