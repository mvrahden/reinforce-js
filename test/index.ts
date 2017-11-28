import { Env } from "../src/Env";
import { Opt } from "../src/Opt";
import { DQNSolver } from "../src/dqn/DQNSolver";

const env = new Env(9, 9);
const opt = new Opt();
opt.setAlpha(0.005); // value function learning rate
opt.setEpsilon(0.2); // initial epsilon for epsilon-greedy policy, [0, 1)
opt.setNumHiddenUnits(2);
opt.setGamma(0.9); // discount factor, [0, 1)
opt.setExperienceAddEvery(5); // number of time steps before we add another experience to replay memory
opt.setExperienceSize(10000); // size of experience
opt.setLearningStepsPerIteration(5);
opt.setTDErrorClamp(1.0); // for robustness
opt.setNumHiddenUnits(100); // number of neurons in hidden layer

const brain = new DQNSolver(env, opt);

console.log(JSON.stringify(brain));