import sys
import agent

if len(sys.argv) != 0:
    a = agent.agent(environment=(sys.argv)[1])
else:
    a = agent.agent()