class tempTransition:

    def __init__(self, qAgent, newState, newAction, newRs = None):

        self.agent = qAgent

        self.oldState = qAgent.previous_state
        self.oldAction = qAgent.last_action

        self.oldRs = qAgent.last_policy

        self.newState = newState
        self.newAction = newAction
        self.newRs = newRs

    def __enter__(self):
        self.agent.previous_state = self.newState
        self.agent.last_action = self.newAction

        if self.newRs:
            self.agent.last_policy = self.newRs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.previous_state = self.oldState
        self.agent.last_action = self.oldAction
        self.agent.last_policy = self.oldRs