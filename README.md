# Neural Hierarchical Reinforcement Learning

This project present an implementation of a **Hierarchical Reinforcement Learning** (**HRL**) algoritim, where state-action value functions are learned by a **Long-Short Term Memory** (**LSTM**) **Artificial Neural Network** (**ANN**).

## Python

This project was built on **python 3.5.2**. In order to run the following packages are needed:

* numpy **1.14.15**
* tensorflow **1.10.0**
* matplotlib **2.2.2** (*optional*, for GUI)


## Running the project

### Agent

To run the agent (manually), go to `<repository>/simulation/simModel/agent`, then run the script `load_agent.py`. Add the option `-i` to keep the script running after execution, otherwise it will close as soon as the agent is loaded.

```
python3 -i load_agent.py
```

You can also add a series of parameters to specify how the agent should be loaded. Each parameter is given to the script as a couple `key value`

```
python3 -i load_agent.py {key value}
```

The following options are available:

* *key:* `loop`, *values:* `True`,`False`, *default:* `False` <br /> if this option is set to `True`, then the script will require you to input the number of time-steps you want the agent to act. Once this is done, the program will keep asking you again a new number of time-steps (submitting nothing will result in the previous value being used). To stop the loop input `0`. <br/> This option is most useful if you just want the agent to act and observe its behavior
* *key:* `GUI`, *values:* `True`, `False`, *default:* `False` <br /> if this option is set to `True`, then the GUI will also be loaded. With the GUI you can observe the maze and the agent's current position and past visited locations.
* *key:* `noPrint`, *values:* `True`, `False`, *default:* `False` <br /> if this option is set to `True`, then the agent object will print no statement (no output)
* *key:* `path`, *values:* `path/to/file`, *default:* `None` <br /> if this option is left to value `None`, then the default environment will be loaded. If a path to a file is specified, then the latter will be used as maze (environment) for the agent. The file must be a CSV file, removing the header tags (see `<repository>/simulation/files/maze.txt` for reference)

These options can pe used in any order and number (eg. `python3 -i load_agent.py loop False GUI True`).

### Running the experiment

#### Average R per time-step
To run an experiment, go to `<repository>/testing/`, and run the python script `testing_avgR.py`.

In this experiment a number *n* of time-steps is specified. Then the agent will perform *n* actions, and the average reward at each time-step is recorded.

This script will require some parameters to be specified:

* *key:* `name`, *values:* `file/name` <br />  the name of the folder in which to store the experiment's results. The results will be found in `<repository>/tasting/tests/<name>`.
* *key:* `n`, *values:* `<integer>` <br />  the **number of time-steps**, that is the number of actions the agent is left to do
* *key:* `e`, *values:* `<integer>` <br /> the number of times the experiment has to be repeated.
* *key:* `maze`, *values:* `path/to/file` <br />   the path to the file defining the environment (maze map). Set to `def` to leave default maze.
* *key:* `pars`, *values:* `path/to/file` <br />  path to *json file* containing the parameters the agent should use (such as *learning rate* for instance)
* *key:* `origin`, *values:* `path/to/file` <br /> path to *json file* containing an experiment's parameter setting (*n*, *e*, *name*, ecc...) 

If any of these parameters is not given when lunching the script, they will be asked (as input) during the script's run.

After giving to the experiment a name of your liking, say *experiment_1*, you will find the results in the folder `<repository>/testing/tests/experiment_1_n`.
Here *n* is the lowest free integer such that the name *experiment_1_n* is not in the folder `<repository>/testing/tests`.


#### Restart experiment
To run an experiment, go to `<repository>/testing/`, and run the python script `testing_restart.py`.

The experiment run is very simple. Given a maze, the agent will start from the middle, and have to reach one of the two exits.
An **iteration** consists of the agent reaching an exit from the starting point. After an iteration is complete, the agent has a **visa**, that is a given number of time-steps, before it is pulled back to the starting point.
The number of time-steps required to complete an iteration is what is recorded.

This script will require some parameters to be specified:

* *key:* `name`, *values:* `file/name` <br />  the name of the folder in which to store the experiment's results. The results will be found in `<repository>/tasting/tests/<name>`.
* *key:* `v`, *values:* `<integer>` <br />  the **visa** size, that is, how long to wait (in time-steps) before the agent is pulled back to the starting point (after the exit is found).
* *key:* `n`, *values:* `<integer>` <br />  the **number of iterations**, that is the number of times the agent has to find the exit, and thus the number of times it is pulled back to the starting point.
* *key:* `e`, *values:* `<integer>` <br /> the number of times the experiment has to be repeated.
* *key:* `maze`, *values:* `path/to/file` <br />   the path to the file defining the environment (maze map). Set to `def` to leave default maze.
* *key:* `pars`, *values:* `path/to/file` <br />  path to *json file* containing the parameters the agent should use (such as *learning rate* for instance)
* *key:* `origin`, *values:* `path/to/file` <br /> path to *json file* containing an experiment's parameter setting (*n*, *e*, *name*, ecc...) 

If any of these parameters is not given when lunching the script, they will be asked (as input) during the script's run.

After giving to the experiment a name of your liking, say *experiment_1*, you will find the results in the folder `<repository>/testing/tests/experiment_1_n`.
Here *n* is the lowest free integer such that the name *experiment_1_n* is not in the folder `<repository>/testing/tests`.
