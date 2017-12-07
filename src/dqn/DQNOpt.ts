import { Opt } from '../Opt';

export class DQNOpt extends Opt {
  private alpha: number = 0.01;
  private epsilon: number = 0.2;
  private gamma: number = 0.75;
  private experienceAddEvery: number = 25;
  private experienceSize: number = 5000;
  private learningStepsPerIteration: number = 10;
  private tdErrorClamp: number = 1.0;
  private numHiddenUnits: number = 100;

  /**
   * Sets the Value Function Learning Rate
   * @param alpha defaults to 0.005
   */
  public setAlpha(alpha: number): void {
    this.alpha = alpha;
  }

  /**
   * Sets the initial Epsilon Factor (Exploration Factor or Greedy Policy)
   * @param epsilon value from [0,1); defaults to 0.2
   */
  public setEpsilon (epsilon:number): void {
    this.epsilon = epsilon;
  }

  /**
   * Sets the initial Future Reward Discount Factor
   * @param gamma value from [0,1); defaults to 0.9
   */
  public setGamma(gamma: number): void {
    this.gamma = gamma;
  }

  /**
   * Sets the Number of Time Steps before another Experience is added to Replay Memory
   * @param timesteps defaults to 5
   */
  public setExperienceAddEvery(experienceAddEvery:number): void {
    this.experienceAddEvery = experienceAddEvery;
  }
  
  /**
   * Sets Replay Memory Size
   * @param experienceReplay defaults to 10000
   */
  public setExperienceSize(experienceSize:number): void {
    this.experienceSize = experienceSize;
  }
  
  /**
   * Sets the learning steps per iteration
   * @param steps defaults to 5
   */
  public setLearningStepsPerIteration(learningStepsPerIteration:number): void {
    this.learningStepsPerIteration = learningStepsPerIteration;
  }
  
  /**
   * Sets the TD-Error Clamp for Robustness
   * @param tdErrorClamp defaults to 1.0
   */
  public setTDErrorClamp(tdErrorClamp:number): void {
    this.tdErrorClamp = tdErrorClamp;
  }
  
  /**
   * Sets the number of neurons in hidden layer
   * @param numHiddenUnits defaults to 100
   */
  public setNumHiddenUnits(numHiddenUnits:number): void {
    this.numHiddenUnits = numHiddenUnits;
  }
}
