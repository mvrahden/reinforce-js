import { Opt } from '../Opt';

export class DQNOpt extends Opt {
  protected numberOfHiddenUnits: number = 100;
  protected epsilon: number = 0.2;

  protected alpha: number = 0.01;
  protected gamma: number = 0.75;
  protected experienceSize: number = 5000;

  protected learningStepsPerIteration: number = 10;
  protected experienceAddEvery: number = 25;
  protected delta: number = 1.0;

  /**
   * Sets the Value Function Learning Rate
   * @param alpha defaults to 0.005
   */
  public setAlpha(alpha: number): void {
    this.alpha = alpha;
  }

  /**
   * Sets the Epsilon Factor (Exploration Factor or Greedy Policy)
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
   * @param timesteps defaults to 5
   */
  public setExperienceAddEvery(experienceAddEvery: number): void {
    this.experienceAddEvery = experienceAddEvery;
  }

  /**
   * Sets Replay Memory Size
   * @param experienceReplay defaults to 10000
   */
  public setExperienceSize(experienceSize: number): void {
    this.experienceSize = experienceSize;
  }

  /**
   * Sets the learning steps per iteration
   * @param steps defaults to 5
   */
  public setLearningStepsPerIteration(learningStepsPerIteration: number): void {
    this.learningStepsPerIteration = learningStepsPerIteration;
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
}
