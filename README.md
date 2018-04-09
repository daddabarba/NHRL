# Neural Hierarchical Reinforcement Learning

This project present an implementation of a **Hierarchical Reinforcement Learning** (**HRL**) algoritim, where state-action value functions are learned by a **Long-Short Term Memory** (**LSTM**) **Artificial Neural Network** (**ANN**).

## Python

This project was built on **python 3.5.2**. In order to run the following packages are needed:

* numpy
* tensorflow
* scipy (*optional*, for testing)
* matplotlib (*optional*, for GUI)


## Running the project

### Agent

To run the agent (manually), go to *repository/simulation/simModel/agent*, then run the script *load_agent.py*, with option *-i* (to keep the script running after execution).

```
python3 -i load_agent.py
```

The GUI and agent will be loaded in the script, at which point it will ask you to specify how many steps the agent should performs. After this is done the script will keep asking you this same thing.To re-submit the previous number of steps, simply press *submit*.

To stop the loop submit 0. To not start the loop at all, add the parameters *loop* (the parameter) and then *False* (the setting).

```
python3 -i load_agent.py loop False
```

To specify a different path of the maze source file (from the default one), add the parameter *path* and then specify the relative path to the file.

```
python3 -i load_agent.py path path/to/file
```

You can combine these two settings, in any order.

### Experiment

To run an experiment, go to *repository/testing*, and run the python script *testing.py*.

The experiment run is very simple. Given a maze, the agent will start from the middle, and have to reach one of the two exits.
An *iteration* consists of the agent reaching an exit from the starting point. After an iteration is complete, the agent has a *visa*, that is a given number of time-steps, before it is pulled back to the starting point.
The number of time-steps required to complete an iteration is what is recorded.

This script, besides some formalities such as the results' folder name and the environment to use, will require you to choose among 2 parameters:

* the **number of iteration**, that is the number of times the agent has to find the exit, and thus the number of times it is pulled back to the starting point
* the *visa* size, that is, how long to wait (in time-steps) before the agent is pulled back to the starting point (after the exit is found)

After giving to the experiment a name of your liking, say *experiment_1*, you will find the results in the folder *repository/testing/tests/experiment_1_n*.
Here *n* is the lowest free integer such that the name *experiment_1_n* is not in the folder *repository/testing/tests*.