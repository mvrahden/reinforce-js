export class Opt {

  /**
   * Get property value of Opt by fieldname
   * @param fieldname name of the property as `string`
   * @returns value or `undefined` (string) if no value exists
   */
  public get(fieldname:string): any | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
}
