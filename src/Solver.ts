import { Opt } from "./Opt";

export abstract class Solver {
  public abstract act(stateList:Array<any>):number;
  public abstract learn(r1:number):void;
  public abstract reset():void;
  public abstract toJSON():object;
  public abstract fromJSON(json:{}):void;
  
  protected getopt (opt:Opt, fieldName:string, defaultValue:any):any {
    if(typeof opt === 'undefined') { return defaultValue; }
    return(typeof opt.get(fieldName) !== 'undefined') ? opt.get(fieldName) : defaultValue;
  }
}
