export class Env {

  protected readonly width: number;
  protected readonly height: number;
  protected readonly numberOfStates: number;
  protected readonly numberOfActions: number;

  constructor(width: number, height: number, numberOfStates: number, maxNumberOfActions: number) {
    this.width = width;
    this.height = height;
    this.numberOfStates = numberOfStates;
    this.numberOfActions = maxNumberOfActions;
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
