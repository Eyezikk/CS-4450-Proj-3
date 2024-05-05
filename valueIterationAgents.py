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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # Run value iteration for self.iteration num of times, updating each state with its new value each time
        # Reminder: V0 gives every state a value of 0.
        "*** YOUR CODE HERE ***"

        beginningStates = self.mdp.getStates()

        for i in range(self.iterations):

            nextIterationVals = self.values.copy()
            for state in beginningStates:

                possibleActions = self.mdp.getPossibleActions(state)
                allActionBestVals = []

                for action in possibleActions:

                    # This will give the max later
                    possibleValsThisAction = []

                    # Multiply the probability by expected reward + discount*nextVal
                    transProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for (nextState, prob) in transProbs:

                        reward = self.mdp.getReward(state, action, nextState)
                        nextVal = self.discount * self.getValue(nextState)

                        possibleValsThisAction.append(prob * (reward + nextVal))

                    # Sum up the values * probabilities from all possible resulting states after taking the action to
                    # determine value of this action
                    allActionBestVals.append(sum(possibleValsThisAction))

                # If no valid actions, don't change value
                if (allActionBestVals == []):
                    nextIterationVals[state] = self.values[state]
                else:
                    nextIterationVals[state] = max(allActionBestVals)

            # We've now run value iteration on all states. Update the values counter in this batch.
            self.values = nextIterationVals


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

        possibleValsThisAction = []

        transProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for (nextState, prob) in transProbs:

            reward = self.mdp.getReward(state, action, nextState)
            nextVal = self.discount * self.getValue(nextState)

            possibleValsThisAction.append(prob * (reward + nextVal))

        return sum(possibleValsThisAction)

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)

        actionVals = util.Counter()

        for action in possibleActions:
            actionVals[action] = self.computeQValueFromValues(state, action)

        if (possibleActions == []):
            return None

        return actionVals.argMax()

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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

        beginningStates = self.mdp.getStates()

        i = 0
        while (i < self.iterations):

            #nextIterationVals = self.values.copy()

            for state in beginningStates:
                if (i > self.iterations - 1):
                    break

                possibleActions = self.mdp.getPossibleActions(state)
                allActionBestVals = []

                for action in possibleActions:

                    # This will give the max later
                    possibleValsThisAction = []

                    # Multiply the probability by expected reward + discount*nextVal
                    transProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for (nextState, prob) in transProbs:

                        reward = self.mdp.getReward(state, action, nextState)
                        nextVal = self.discount * self.getValue(nextState)

                        possibleValsThisAction.append(prob * (reward + nextVal))

                    # Sum up the values * probabilities from all possible resulting states after taking the action to
                    # determine value of this action
                    allActionBestVals.append(sum(possibleValsThisAction))

                # If no valid actions, don't change value
                if (allActionBestVals == []):
                    self.values[state] = 0
                else:
                    self.values[state] = max(allActionBestVals)
                i += 1


            # We've now run value iteration on all states. Update the values counter in this batch.
            #self.values = nextIterationVals


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.calculatePredecessors()
        states = self.mdp.getStates()
        priorityQueue = util.PriorityQueue()

        #nonTerminalStates = [state for state in states if not self.mdp.isTerminal(state)]
        for s in states:
            # Skip this state if this is a terminal state
            # I'd prefer just to iterate through the commented-out nonTerminalStates list above, but the project
            # details make it sound like we have to do it this way because it requires us to iterate through
            # self.mdp.getStates() or the autograder gets mad.
            if (self.mdp.isTerminal(s)):
                continue

            diff = abs(self.values[s] - self.computeBestQValue(s))
            priorityQueue.push(s, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            s = priorityQueue.pop()
            if (not self.mdp.isTerminal(s)):
                self.values[s] = self.computeBestQValue(s)

            for p in predecessors[s]:
                diff = abs(self.values[p] - self.computeBestQValue(p))
                if (diff > self.theta):
                    priorityQueue.update(p, -diff)


    def calculatePredecessors(self):
        predecessors = util.Counter()
        states = self.mdp.getStates()

        for currentState in states:

            predecessors[currentState] = set()

            # Find every state that could end up in currentState
            for otherState in states:

                otherStateActions = self.mdp.getPossibleActions(otherState)
                for action in otherStateActions:
                    # If that otherState has an action that could end up in currentState (prob > 0), it's a predecessor
                    # of currentState
                    for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(otherState, action):
                        if (nextState == currentState and prob > 0):
                            predecessors[currentState].add(otherState)

        return predecessors

    def computeBestQValue(self, state):

        possibleActions = self.mdp.getPossibleActions(state)

        actionVals = []

        for action in possibleActions:
            actionVals.append(self.computeQValueFromValues(state, action))

        if (possibleActions == []):
            return None

        return max(actionVals)

