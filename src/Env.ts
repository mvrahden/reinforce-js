export class Env {

  protected numStates: number;
  protected maxNumActions: number;

  constructor(numStates: number, maxNumActions: number) {
    this.numStates = numStates;
    this.maxNumActions = maxNumActions;
  }

  /**
   * Get property value of Env by fieldname
   * @param fieldname name of the property as `string`
   * @returns value or `undefined` of no value exists
   */
  public get(fieldname: string): number | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
}
