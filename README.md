#Neural Hierarchical Reinforcement Learning

This project present an implementation of a **Hierarchical Reinforcement Learning** (**HRL**) algoritim, where state-action value functions are learned by a **Long-Short Term Memory** (**LSTM**) **Artificial Neural Network** (**ANN**).

##Python

This project was built on **python 3.5.2**. In order to run the following packages are needed:

-numpy
-tensorflow
-scipy (*optional*, for testing)
-matplotlib (*optional*, for GUI)


##Running the project

###Agent

To run the agent (manually), go to repository/simulation/simModel/agent, then run the script *load_agent.py*, with option *-i* (to keep the script running afer execution).

'''
python3 -i load_agent.py
'''

The GUI and agent will be loaded in the script, at which point it will ask you to specify how many steps the agent should performs. After this is done the script will keep asking you this same thing.To re-submit the previous number of steps, simply press *submit*.

To stop the loop submit 0. To not start the loop at all, add the parameters *loop* (the parameter) and then *False* (the setting).

'''
python3 -i load_agent.py loop False
'''

To specify a different path of the maze source file (from the deafault one), add the parameter *path* and then specify the relative path to the file.

'''
python3 -i load_agent.py path path/to/file
'''

You can combine these two settings, in any order.