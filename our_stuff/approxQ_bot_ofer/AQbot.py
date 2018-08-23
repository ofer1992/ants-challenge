import sys
sys.path.append("../../")  # so imports work
from our_stuff.approxQ_bot_ofer.ants import *
from our_stuff.agents import ApproximateQAgent
import pickle
import our_stuff.util
import copy
import math

EATING_FOOD_REWARD = 5
ENEMY_HILL_REWARD = 2000
OWN_HILL_REWARD = -5
VISION_REWARD = 5
NEWLY_FOUND_FOOD_REWARD = 1
ANT_DEAD_REWARD = -10
ENEMY_ANT_DEAD_REWARD = 50

class FeatureExtractor:
  def getFeatures(self, state, action):
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.
    """
    our_stuff.util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = our_stuff.util.Counter()
    feats[(state,action)] = 1.0
    return feats


class BasicExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = our_stuff.util.Counter()
        ants = state[0]
        ant_loc = state[1]

        # Possible solution to dead state
        if ant_loc is None:
            feats['dead'] = 1
            return feats

        new_ant_loc = ants.destination_with_obstacles(ant_loc, action)
        map_size = float(ants.cols * ants.rows)

        feats["bias"] = 1.0

        # Food
        if ants.food():
            food_distances = [ants.distance_manhattan(new_ant_loc, f) for f in ants.food()]
            food_d = min(food_distances)
            food_loc = ants.food()[food_distances.index(food_d)]
            feats["food"] = food_d / map_size
            # how_many_will_be_eaten = [d for d in food_distances if d == 1]
            feats["will-eat-food"] = 1 if food_d == 1 else 0
            # ants_distance_from_food = [ants.distance_manhattan(ant, food_loc) for ant in ants.my_ants()]
            # feats["there-is-a-closer-ant"] = 1 if min(ants_distance_from_food) < ants.distance_manhattan(new_ant_loc, food_loc) else 0

        # # Ants in attack range
        # friendly, unfriendly, dead = ants.attack_range_of_loc(new_ant_loc, state)
        # ants_in_range = friendly + unfriendly
        # feats["#-of-enemy-ants-in-range"] = len(unfriendly) / float(len(ants_in_range) + 1)
        # feats["#-of-friendly-ants-in-range"] = len(friendly) / float(len(ants_in_range) + 1)
        #

        # Enemy hills
        if ants.enemy_hills():
            enemy_hill_d = min(ants.distance_manhattan(new_ant_loc, h[0]) for h in ants.enemy_hills()) / map_size
            feats["enemy-hill"] = enemy_hill_d
            feats["will-step-on-enemy-hill"] = 1 if enemy_hill_d == 0 else 0

        feats.divideAll(10.0)
        return feats


def get_reward(prev_ants, prev_ant_loc, ants, ant_loc):
    reward = 0
    if ant_loc is None:
        return ANT_DEAD_REWARD

    # eating food
    if prev_ants.food():
        food_d = min(ants.distance_manhattan(ant_loc, f) for f in prev_ants.food())
        if food_d == 1:
            reward += EATING_FOOD_REWARD

    # stepping on enemy hill
    if ant_loc in prev_ants.enemy_hills():
        reward += ENEMY_HILL_REWARD

    # stepping on our hill
    if ant_loc in ants.my_hills():
        reward += OWN_HILL_REWARD

    # number of newly discovered foods
    new_food = sum(not prev_ants.visible(food) for food in ants.food())
    reward += new_food * NEWLY_FOUND_FOOD_REWARD

    # killing an enemy
    dead_enemies = prev_ants.attack_range_of_loc(prev_ant_loc, prev_ants)[2]
    reward += ENEMY_ANT_DEAD_REWARD * len(dead_enemies)


    reward -= 1

    return reward

def legal_actions(state):
    actions = [NOOP]
    ants = state[0]
    ant_loc = state[1]

    # This happens when ant has died. The next state is (ants, None)
    if ant_loc is None:
        return actions
    for d in AIM:
        new_loc = ants.destination(ant_loc, d)
        if ants.unoccupied_including_orders(new_loc):
            actions.append(d)
    return actions

class ApproxQBot:
    def __init__(self, train, load_from_file):
        self.agent = ApproximateQAgent(actionFn=legal_actions, extractor=BasicExtractor, epsilon=0.1, alpha=0.2, gamma=0.8)
        if load_from_file:
            sys.stderr.write("Loaded weights from file\n")
            with open("Weights.txt",   'rb') as f:
                Weights = pickle.load(f)
            self.agent.setW(Weights)
        self.prev_state = None
        self.train = train
        self.accumulated_rewards = 0

    def do_setup(self, ants):
        self.prev_state = None
        self.accumulated_rewards = 0

    def do_turn(self, state):
        if self.train and self.prev_state:
            for ant in self.prev_state.my_ants():
                action = self.prev_state.orders2[ant]
                curr_loc = state.destination_with_obstacles(ant, action)
                if curr_loc not in state.ant_list or state.ant_list[curr_loc] !=MY_ANT:
                    curr_loc = None
                reward = get_reward(self.prev_state, ant, state, curr_loc)
                self.accumulated_rewards += reward
                self.agent.update((self.prev_state, ant), action, (state, curr_loc), reward)

        action_function = self.agent.getAction if self.train else\
            self.agent.getPolicy
        for ant in state.my_ants():
            action = action_function((state, ant))
            state.remember_order(ant, action)
            state.issue_order((ant, action))

        # saving state and actions for next turn
        self.prev_state = copy.deepcopy(state)

    def do_endgame(self, state):
        """
        Called once in end of game to update Q-weights.
        """
        sys.stderr.write("Total reward: "+str(self.accumulated_rewards)+"\n")
        sys.stderr.write("Turns: "+str(state.turns_so_far)+"\n")
        if self.train and self.prev_state:
            for ant in self.prev_state.my_ants():
                action = self.prev_state.orders2[ant]
                curr_loc = state.destination_with_obstacles(ant, action)
                if curr_loc not in state.ant_list or state.ant_list[curr_loc] !=MY_ANT:
                    curr_loc = None
                reward = get_reward(self.prev_state, ant, state, curr_loc)
                self.agent.update((self.prev_state, ant), action, (state, curr_loc), reward)


    def save_weights(self):
        sys.stderr.write("Weights:"+str(self.agent.weights)+"\n")
        if self.train:
            with open("Weights.txt", 'wb') as f:
                pickle.dump(self.agent.weights, f)

def run(bot, training_rounds):
    'parse input, update game state and call the bot classes do_turn method'
    ants = Ants()
    map_data = ''
    rounds = 0
    win, draw, loss = 0, 0, 0
    while (True):
        try:
            current_line = sys.stdin.readline()  # string new line char
            current_line = current_line.rstrip('\r\n')

            if current_line.lower() == 'ready':
                ants.setup(map_data)
                bot.do_setup(ants)
                ants.finish_turn()
                map_data = ''
            elif current_line.lower() == 'go':
                ants.update(map_data)
                # call the do_turn method of the class passed in
                bot.do_turn(ants)
                ants.finish_turn()
                map_data = ''

            # support for multi-game input
            elif current_line.lower() == 'end':
                sys.stderr.write("finished game\n")
                rounds += 1
                if rounds == training_rounds:
                    sys.stderr.write("training over. dumping weights\n")
                    bot.save_weights()
                    sys.stderr.write("finished dump!\n")
                    bot.train = False
                current_line = sys.stdin.readline().rstrip('\r\n')
                while not current_line.startswith('playerturns'):
                    current_line = sys.stdin.readline().rstrip('\r\n')
                current_line = sys.stdin.readline().rstrip('\r\n')
                while current_line != 'go':
                    map_data += current_line + '\n'
                    current_line = sys.stdin.readline().rstrip('\r\n')
                ants.update(map_data)
                bot.do_endgame(ants)
                map_data = ''
            else:
                map_data += current_line + '\n'
        except EOFError:
            sys.stderr.write("GOT EOF\n")
            break
        except KeyboardInterrupt:
            raise
        except:
            # don't raise error or return so that bot attempts to stay alive
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()


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
        training_rounds = int(sys.argv[1])
        b = ApproxQBot(training_rounds > 0, True)
        run(b, training_rounds)

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
