export class Env {
  constructor(
    private numStates: number,
    private maxNumActions: number
  ) {
  }

  getNumStates():number {
    return this.numStates;
  }
  getMaxNumActions():number {
    return this.maxNumActions;
  }
}
