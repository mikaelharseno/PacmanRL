# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        "Each iteration"
        iterCount = 0
        while iterCount < self.iterations:
            stateVector = self.mdp.getStates()
            nextStateValues = [0 for i in range(len(stateVector))]
            for stateNum in range(len(stateVector)):
                state = stateVector[stateNum]
                if not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    actionValues = [0 for i in range(len(actions))]
                    for actionNum in range(len(actions)):
                        curAction = actions[actionNum]
                        nextstatesandprobs = self.mdp.getTransitionStatesAndProbs(state, curAction)
                        val = 0
                        for nextstate, prob in nextstatesandprobs:
                            """print("")
                            print("Start:")
                            print("State:")
                            print(state)
                            print("Action:")
                            print(curAction)
                            print("NextState:")
                            print(nextstate)
                            print(prob)
                            print(self.mdp.getReward(state, curAction, nextstate))
                            print(self.getValue(nextstate))"""
                            val += prob*(self.mdp.getReward(state, curAction, nextstate) + self.discount*self.getValue(nextstate))
                        actionValues[actionNum] = val
                    nextStateValues[stateNum] = max(actionValues)

            for stateNum in range(len(stateVector)):
                self.values[stateVector[stateNum]] = nextStateValues[stateNum]

            iterCount += 1
            """print("AAAAAAAAAAAAAAAAAAAAAAAAAA")
            print("Itercount:")
            print(iterCount)
            print(stateVector)
            print(nextStateValues)"""

        return None
    

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextstatesandprobs = self.mdp.getTransitionStatesAndProbs(state, action)
        val = 0
        for nextstate, prob in nextstatesandprobs:
            val += prob*(self.mdp.getReward(state, action, nextstate) + self.discount*self.getValue(nextstate))

        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        actionValues = [0 for i in range(len(actions))]
        for i in range(len(actions)):
            actionValues[i] = self.computeQValueFromValues(state, actions[i])
        maxActionIndex = actionValues.index(max(actionValues))
        maxAction = actions[maxActionIndex]

        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterCount = 0
        curStateNum = -1
        stateVector = self.mdp.getStates()
        while iterCount < self.iterations:
            curStateNum = (curStateNum + 1) % len(stateVector)
            state = stateVector[curStateNum]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                actionValues = [0 for i in range(len(actions))]
                for actionNum in range(len(actions)):
                    curAction = actions[actionNum]
                    nextstatesandprobs = self.mdp.getTransitionStatesAndProbs(state, curAction)
                    val = 0
                    for nextstate, prob in nextstatesandprobs:
                        val += prob*(self.mdp.getReward(state, curAction, nextstate) + self.discount*self.getValue(nextstate))
                    actionValues[actionNum] = val
                self.values[state] = max(actionValues)

            iterCount += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateVector = self.mdp.getStates()
        statePossVals = {state:0 for state in stateVector}
        predecessors = {state:util.Counter() for state in stateVector}
        for i in range(len(stateVector)):
            curState = stateVector[i]
            possActions = self.mdp.getPossibleActions(curState)
            for curAction in possActions:
                nextstatesandprobs = self.mdp.getTransitionStatesAndProbs(curState, curAction)
                for nextstate, prob in nextstatesandprobs:
                    predecessors[nextstate][curState] += 1
        """for state in stateVector:
            print("sfdsfd")
            print(state)
            print(predecessors[state].sortedKeys())"""
        pq = util.PriorityQueue()
        for i in range(len(stateVector)):
            state = stateVector[i]
            if not self.mdp.isTerminal(state):
                curVal = self.values[state]
            
                actions = self.mdp.getPossibleActions(state)
                actionValues = [0 for i in range(len(actions))]
                for i in range(len(actions)):
                    actionValues[i] = self.computeQValueFromValues(state, actions[i])
                possVal = max(actionValues)
                statePossVals[state] = possVal
            
                diff = abs(possVal - curVal)
                pq.push(state,-diff)
        iterCount = 0
        while iterCount < self.iterations:
            if pq.isEmpty():
                return
            else:
                nextstate = pq.pop()
                if not self.mdp.isTerminal(nextstate):
                    """print("State popped")
                    print(nextstate)"""
                    self.values[nextstate] = statePossVals[nextstate]
                    preds = predecessors[nextstate].sortedKeys()
                    for pred in preds:
                        state = pred
                        """print(pred)"""
                        curVal = self.values[pred]

                        actions = self.mdp.getPossibleActions(state)
                        actionValues = [0 for i in range(len(actions))]
                        for i in range(len(actions)):
                            actionValues[i] = self.computeQValueFromValues(state, actions[i])
                        possVal = max(actionValues)
                        statePossVals[state] = possVal

                        diff = abs(possVal - curVal)
                        """print(diff)"""
                        if diff > self.theta:
                            pq.update(state,-diff)
            iterCount += 1

        return
        

