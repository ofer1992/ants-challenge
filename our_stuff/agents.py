import util, random

class ValueEstimationAgent():
  """
    Abstract agent which assigns values to (state,action)
    Q-Values for an environment. As well as a value to a
    state and a policy given respectively by,

    V(s) = max_{a in actions} Q(s,a)
    policy(s) = arg_max_{a in actions} Q(s,a)

    Both ValueIterationAgent and QLearningAgent inherit
    from this agent. While a ValueIterationAgent has
    a model of the environment via a MarkovDecisionProcess
    (see mdp.py) that is used to estimate Q-Values before
    ever actually acting, the QLearningAgent estimates
    Q-Values while acting in the environment.
  """

  def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.discount = float(gamma)
    self.numTraining = int(numTraining)

  ####################################
  #    Override These Functions      #
  ####################################
  def getQValue(self, state, action):
    """
    Should return Q(state,action)
    """
    util.raiseNotDefined()

  def getValue(self, state):
    """
    What is the value of this state under the best action?
    Concretely, this is given by

    V(s) = max_{a in actions} Q(s,a)
    """
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
    What is the best action to take in the state. Note that because
    we might want to explore, this might not coincide with getAction
    Concretely, this is given by

    policy(s) = arg_max_{a in actions} Q(s,a)

    If many actions achieve the maximal Q-value,
    it doesn't matter which is selected.
    """
    util.raiseNotDefined()

  def getAction(self, state):
    """
    state: can call state.getLegalActions()
    Choose an action and return it.
    """
    util.raiseNotDefined()

class ReinforcementAgent(ValueEstimationAgent):
  """
    Abstract Reinforcemnt Agent: A ValueEstimationAgent
	  which estimates Q-Values (as well as policies) from experience
	  rather than a model

      What you need to know:
		  - The environment will call
		    observeTransition(state,action,nextState,deltaReward),
		    which will call update(state, action, nextState, deltaReward)
		    which you should override.
      - Use self.getLegalActions(state) to know which actions
		    are available in a state
  """
  ####################################
  #    Override These Functions      #
  ####################################

  def update(self, state, action, nextState, reward):
    """
	    This class will call this function, which you write, after
	    observing a transition and reward
    """
    util.raiseNotDefined()

  ####################################
  #    Read These Functions          #
  ####################################

  def getLegalActions(self,state):
    """
      Get the actions available for a given
      state. This is what you should use to
      obtain legal actions for a state
    """
    return self.actionFn(state)

  # def observeTransition(self, state,action,nextState,deltaReward):
  #   """
  #   	Called by environment to inform agent that a transition has
  #   	been observed. This will result in a call to self.update
  #   	on the same arguments
  #
  #   	NOTE: Do *not* override or call this function
  #   """
  #   self.episodeRewards += deltaReward
  #   self.update(state,action,nextState,deltaReward)
  #
  # def startEpisode(self):
  #   """
  #     Called by environment when new episode is starting
  #   """
  #   self.lastState = None
  #   self.lastAction = None
  #   self.episodeRewards = 0.0
  #
  # def stopEpisode(self):
  #   """
  #     Called by environment when episode is done
  #   """
  #   if self.episodesSoFar < self.numTraining:
  #     self.accumTrainRewards += self.episodeRewards
  #   else:
  #     self.accumTestRewards += self.episodeRewards
  #   self.episodesSoFar += 1
  #   if self.episodesSoFar >= self.numTraining:
  #     # Take off the training wheels
  #     self.epsilon = 0.0    # no exploration
  #     self.alpha = 0.0      # no learning
  #
  # def isInTraining(self):
  #     return self.episodesSoFar < self.numTraining
  #
  # def isInTesting(self):
  #     return not self.isInTraining()

  def __init__(self, actionFn=None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
    """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    ValueEstimationAgent.__init__(self, alpha, epsilon, gamma, numTraining)
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions()
    self.actionFn = actionFn
    # self.episodesSoFar = 0
    # self.accumTrainRewards = 0.0
    # self.accumTestRewards = 0.0
    # self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)


class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, actionFn, numTraining, epsilon, alpha, gamma)
    self.Q = util.Counter()


  def setQ(self, Q):
    self.Q = Q
  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    return self.Q[state, action]

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = self.getLegalActions(state)
    if not actions:
      return 0.0

    return max(self.getQValue(state, a) for a in actions)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    actions = self.getLegalActions(state)
    if not actions:
      return

    max_val = max(self.getQValue(state, a) for a in actions)
    best_actions = [a for a in actions if self.getQValue(state,a) == max_val]
    return random.choice(best_actions)

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    if not legalActions:
      return None

    if util.flipCoin(self.epsilon):
      return random.choice(legalActions)
    else:
      return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    maxQ = self.getValue(nextState)
    self.Q[state,action] = self.Q[state,action] + \
                        self.alpha*(reward+self.discount*maxQ - self.Q[state,action])



class ApproximateQAgent(QLearningAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor, actionFn=None, **args):
    self.featExtractor = extractor()
    QLearningAgent.__init__(self,actionFn=actionFn, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    return self.weights * self.featExtractor.getFeatures(state,action)

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    correction = self.alpha*(reward + self.discount * self.getValue(nextState) - self.getQValue(state,action))
    features = self.featExtractor.getFeatures(state,action)
    for f in features:
      features[f] *= correction
    self.weights = self.weights + features
