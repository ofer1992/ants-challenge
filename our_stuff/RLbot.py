#!/usr/bin/env python
from ants import *
import util
from util import Counter

import collections

class frozendict(collections.Mapping):
    """Don't forget the docstrings!!"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            self._hash = 0
            for pair in self.iteritems():
                self._hash ^= hash(pair)
        return self._hash

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

  def observeTransition(self, state,action,nextState,deltaReward):
    """
    	Called by environment to inform agent that a transition has
    	been observed. This will result in a call to self.update
    	on the same arguments

    	NOTE: Do *not* override or call this function
    """
    self.episodeRewards += deltaReward
    self.update(state,action,nextState,deltaReward)

  def startEpisode(self):
    """
      Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self):
      return self.episodesSoFar < self.numTraining

  def isInTesting(self):
      return not self.isInTraining()

  def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
    """
    actionFn: Function which takes a state and returns the list of legal actions

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    if actionFn == None:
        actionFn = lambda state: state.getLegalActions()
    self.actionFn = actionFn
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.numTraining = int(numTraining)
    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)

import pickle

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
    # self.Q = util.Counter()
    with open("Q.txt", 'rb') as f:
        self.Q = pickle.load(f)

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

def passable(map, loc):
    'true if not water'
    row, col = loc
    return map[row][col] > WATER

def destination(rows, cols, loc, direction):
    'calculate a new location given the direction and wrap correctly'
    row, col = loc
    d_row, d_col = AIM[direction]
    return ((row + d_row) % rows, (col + d_col) % cols)


def action_fn(state):
    actions = []
    for d in AIM:
        if passable(state[0], destination(state[2], state[3], state[1], d)):
            actions.append(d)

def gen_state(ants):
    return frozendict(ants.map), ants.my_ants()[0], ants.rows, ants.cols

def calc_reward(state, action, next_state):
    return 0

# define a class with a do_turn method
# the Ants.run method will parse and update bot input
# it will also run the do_turn method for us
class MyBot:
    def __init__(self):
        # define class level variables, will be remembered between turns
        self.agent = QLearningAgent(action_fn, 1, 0.5, 0.5, 0.8)
        self.last_state, self.last_action = None, None

    # do_setup is run once at the start of the game
    # after the bot has received the game settings
    # the ants class is created and setup by the Ants.run method
    def do_setup(self, ants):
        pass
        # initialize data structures after learning the game settings
        # self.size_of_map = ants.rows * ants.cols
        # self.hills = []
        # self.unseen = []
        # for row in range(ants.rows):
        #     for col in range(ants.cols):
        #         self.unseen.append((row, col))



    def features(self, ants):
        def min_distance(l1, l2):
            def min_d(loc, l):
                return min(ants.distance(loc, loc1) for loc1 in l)

            return min(min_d(loc, l2) for loc in l1)
        features = []
        features.append(len(ants.my_ants()))  # num of ants
        features.append(min_distance(ants.my_ants(), ants.food()))  # closest food
        features.append(len(ants.enemy_hills()))  # num of visible enemy hills
        features.append(len(ants.enemy_ants()))  # num of visible enemy ants
        features.append(0)  # num of ants to die
        features.append(0)  # num of ants we will kill
        features.append(min_distance(ants.enemy_ants(), ants.my_hills()))  # min dist of enemy ant to our hills
        # features.append(float(len(ants.visible())))  # % of map visible
        features.append(min_distance(ants.my_ants(), ants.my_hills()))  # min dist of our ants from our hills

        return

    # do turn is run once per turn
    # the ants class has the game state and is updated by the Ants.run method
    # it also has several helper methods to use
    def do_turn(self, ants):
        # loop through all my ants and try to give them orders
        # the ant_loc is an ant location tuple in (row, col) form
        state = gen_state(ants)
        reward = calc_reward(self.last_state, self.last_action, state)
        if not self.last_state:
            self.agent.update(self.last_state, self.last_action, state, reward)
        self.last_state = state
        self.last_action = self.agent.getAction(state)
        ants.issue_order((state[1], self.last_action))
        
    def end(self):
        with open("Q.txt", 'wb') as f:
            pickle.dump(self.agent.Q, f)
        
if __name__ == '__main__':
    # psyco will speed up python a little, but is not needed
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass
    
    try:
        # if run is passed a class with a do_turn method, it will do the work
        # this is not needed, in which case you will need to write your own
        # parsing function and your own game state class
        Ants.run(MyBot())
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
