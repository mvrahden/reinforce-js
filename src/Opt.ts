export class Opt {
  private epsilon: number;
  private alpha: number;
  private experienceAddEvery: number;
  private experienceSize: number;
  private learningStepsPerIteration: number;
  private tdErrorClamp: number;
  private numHiddenUnits: number;
  private gamma: number;

  get(fieldname:string):any {
    return this[fieldname];
  }
  setGamma (gamma:number) {
    this.gamma = gamma;
  }
  setEpsilon (epsilon:number) {
    this.epsilon = epsilon;
  }
  setAlpha (alpha:number) {
    this.alpha = alpha;
  }
  setExperienceAddEvery (timesteps:number) {
    this.experienceAddEvery = timesteps;
  }
  setExperienceSize (experienceReplay:number) {
    this.experienceSize = experienceReplay;
  }
  setLearningStepsPerIteration (steps:number) {
    this.learningStepsPerIteration = steps;
  }
  setTDErrorClamp(tdErrorClamp:number) {
    this.tdErrorClamp = tdErrorClamp;
  }
  setNumHiddenUnits (numHiddenUnits:number) {
    this.numHiddenUnits = numHiddenUnits;
  }
}
