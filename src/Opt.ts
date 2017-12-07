export class Opt {

  /**
   * Get property value of Opt by fieldname
   * @param fieldname name of the property as `string`
   * @returns value or `undefined` of no value exists
   */
  public get(fieldname:string):number | undefined {
    return this[fieldname] ? this[fieldname] : undefined;
  }
}
