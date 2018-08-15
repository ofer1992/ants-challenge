from ants import *
from our_stuff.agents import ApproximateQAgent
import sys
import pickle
import our_stuff.util
import copy
import math

EATING_FOOD_REWARD = 5
ENEMY_HILL_REWARD = 20
OWN_HILL_REWARD = -5
VISION_REWARD = 5
NEWLY_FOUND_FOOD_REWARD = 1

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
        ant_id = state[1]
        new_ant_loc = ants.destination(ants.my_ants()[ant_id], action)
        map_size = float(ants.cols * ants.rows)

        feats["bias"] = 1.0

        feats["obstacle"] = 0 if ants.passable(new_ant_loc) else 1
        # sys.stderr.write(str(feats["obstacle"]) + '\n') # TODO there's an error here, we thinks

        # Food
        if ants.food():
            food_distances = [ants.distance(new_ant_loc, f) for f in ants.food()]
            food_d = min(food_distances)
            food_loc = ants.food()[food_distances.index(food_d)]
            feats["food"] = 1 - (1 / (1 + math.exp(-0.6 * food_d + 3)))
            feats["will-eat-food"] = 1 if food_d == 1 else 0
            ants_distance_from_food = [ants.distance(ant, food_loc) for ant in ants.my_ants()]
            # feats["there-is-a-closer-ant"] = 1 if min(ants_distance_from_food) < ants.distance(new_ant_loc, food_loc) else 0

        # Percentage of the map we see
        ants.visible([0, 0])  # Triggers updating ants.vision
        vision = sum(row.count(True) for row in ants.vision)
        sys.stderr.write("vision: " + str(vision) + '\n')
        feats["map-vision"] = vision
        # sys.stderr.write("ratio map feature value is: " + str(feats["ratio-of-map"]) + '\n')

        # # Ants in attack range
        # friendly, unfriendly, dead = ants.attack_range_of_loc(new_ant_loc)
        # feats["#-of-enemy-ants-in-range"] = len(unfriendly)
        # feats["#-of-friendly-ants-in-range"] = len(friendly)
        #
        # # Enemy hills
        # if ants.enemy_hills():
        #     enemy_hill_d = min(ants.distance(new_ant_loc, h[0]) for h in ants.enemy_hills()) / map_size
        #     feats["enemy-hill"] = enemy_hill_d
        #     feats["will-step-on-enemy-hill"] = 1 if enemy_hill_d == 0 else 0

        # Stepping on other ants' new location
        # if new_ant_loc in ants.orders.values():
        #     feats["will-collide-with-ant"] = 1

        feats.divideAll(10.0)
        return feats


def get_reward(prev_ants, prev_actions, ants, ant_id):
    reward = 0
    prev_ant_loc = prev_ants.my_ants()[ant_id]
    new_ant_loc = prev_ants.destination_with_obstacles(prev_ants.my_ants()[ant_id], prev_actions[ant_id])
    # dead or tried to walk through barrier TODO: how to identify collision between two ants.
    # if new_ant_loc not in ants.my_ants():
    #     reward -= 1

    # eating food
    if prev_ants.food():
        food_d = min(ants.distance(new_ant_loc, f) for f in prev_ants.food())
        if food_d == 1:
            reward += EATING_FOOD_REWARD
    #
    # # stepping on enemy hill
    # if new_ant_loc in prev_ants.enemy_hills():
    #     reward += ENEMY_HILL_REWARD
    #
    # # stepping on our hill
    # if new_ant_loc in ants.my_hills():
    #     reward += OWN_HILL_REWARD

    # # number of newly discovered foods
    # new_food = sum(not prev_ants.visible(food) for food in ants.food())
    # # sys.stderr.write(str(new_food) + '\n')
    # reward += new_food * NEWLY_FOUND_FOOD_REWARD
    # # if new_food > 0:
    # #     sys.stderr.write("found new food reward: " + str(reward) + '\n')

    # # killing an enemy
    # dead_enemies = prev_ants.attack_range_of_loc(prev_ant_loc)[2]
    # reward += 5 * len(dead_enemies)

    # # dying
    # if 0 in ants.dead_list[prev_ant_loc] or 0 in ants.dead_list[new_ant_loc]: # TODO: find out which is the right condition
    #     # sys.stderr.write("ant dead" + "\n")
    #     # sys.stderr.write(str(prev_ant_loc) + " " + str(new_ant_loc) + "\n")
    #     reward -= 5

    reward -= 0.2

    return reward

def legal_actions(state):
    actions = []
    ants = state[0]
    ant_id = state[1]

    # This happens when ant has died. The next state is (ants, None)
    if ant_id is None:
        # sys.stderr.write("action_fn got a state with None ant_id\n")
        return actions
    for d in AIM:
        new_loc = ants.destination(ants.my_ants()[ant_id], d)
        # sys.stderr.write(str(ants.orders.values())+"\n")
        if ants.unoccupied_including_orders(new_loc):
            actions.append(d)
    if actions is None:
        sys.stderr.write("action_fn returned a none\n")
    return actions

class ApproxQBot:
    def __init__(self, train, load_from_file):
        self.agent = ApproximateQAgent(actionFn=legal_actions, extractor=BasicExtractor, epsilon=1, alpha=0.5, gamma=0.9)
        if load_from_file:
            sys.stderr.write("Loaded weights from file\n")
            with open("Weights.txt",   'rb') as f:
                Weights = pickle.load(f)
            self.agent.setW(Weights)
        self.prev_state, self.prev_actions = None, None
        self.train = train

    def do_setup(self, ants):
        self.prev_state, self.prev_actions = None, None

    def do_turn(self, state):

        # Calculate rewards and update Q-Function weights per ant.
        # sys.stderr.write('before reward\n')
        if self.train and self.prev_state:
            ants_list = self.prev_state.my_ants()
            for prev_ant_id in range(len(self.prev_state.my_ants())):
                # sys.stderr.write('before reward loc\n')
                curr_loc = state.destination(self.prev_state.my_ants()[prev_ant_id], self.prev_actions[prev_ant_id])
                # sys.stderr.write('before reward calculation\n')
                reward = get_reward(self.prev_state, self.prev_actions, state, prev_ant_id)
                curr_id = state.my_ants().index(curr_loc) if curr_loc in state.my_ants() else None
                # sys.stderr.write('before reward update\n')
                self.agent.update((self.prev_state, prev_ant_id), self.prev_actions[prev_ant_id], (state, curr_id), reward)

        # sys.stderr.write('before actions\n')
        actions = []
        if self.train:
            for ant_id in range(len(state.my_ants())):
                actions.append(self.agent.getAction((state, ant_id)))
                state.remember_order(state.my_ants()[ant_id], actions[-1])
        else:
            for ant_id in range(len(state.my_ants())):
                actions.append(self.agent.getPolicy((state, ant_id)))
                state.remember_order(state.my_ants()[ant_id], actions[-1])

        # sys.stderr.write('before issuing\n')
        # Issuing new orders
        for ant_id in range(len(state.my_ants())):

            state.issue_order((state.my_ants()[ant_id], actions[ant_id]))

        # saving state and actions for next turn
        self.prev_state = copy.deepcopy(state)
        self.prev_actions = actions


    def do_endgame(self, state):
        """
        Called once in end of game to update Q-weights.
        """
        # sys.stderr.write("Weights:"+str(self.agent.weights)+"\n")
        if self.train and self.prev_state:
            for ant_id in range(len(self.prev_state.my_ants())):
                next_loc = state.destination(self.prev_state.my_ants()[ant_id], self.prev_actions[ant_id])
                reward = get_reward(self.prev_state, self.prev_actions, state, ant_id)
                next_id = state.my_ants().index(next_loc) if next_loc in state.my_ants() else None
                self.agent.update((self.prev_state, ant_id), self.prev_actions[ant_id], (state, next_id), reward)


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
    while (True):
        try:
            current_line = sys.stdin.readline()  # string new line char
            # sys.stderr.write(current_line)
            current_line = current_line.rstrip('\r\n')
            # sys.stderr.write(current_line + "   outside\n")

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
                    sys.stderr.write("finished (steamy) dump!\n")
                    bot.train = False
                current_line = sys.stdin.readline().rstrip('\r\n')
                while not current_line.startswith('playerturns'):
                    # sys.stderr.write(current_line+" inside loop 1\n")
                    current_line = sys.stdin.readline().rstrip('\r\n')
                # sys.stderr.write(current_line + " after loop 1\n")
                current_line = sys.stdin.readline().rstrip('\r\n')
                while current_line != 'go':
                    # sys.stderr.write(current_line+" inside loop 2\n")
                    map_data += current_line + '\n'
                    current_line = sys.stdin.readline().rstrip('\r\n')
                # sys.stderr.write(current_line + " after loop 2\n")
                ants.update(map_data)
                bot.do_endgame(ants)
                bot.agent.epsilon *= 0.99
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
        b = ApproxQBot(training_rounds > 0, training_rounds == 0)
        run(b, training_rounds)

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
