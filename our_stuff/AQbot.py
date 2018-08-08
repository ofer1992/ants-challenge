from ants import *
from agents import ApproximateQAgent
import sys
import pickle
import util
import copy



class FeatureExtractor:
  def getFeatures(self, state, action):
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.
    """
    util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats


class BasicExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        ants = state[0]
        ant_id = state[1]
        new_ant_loc = ants.destination(ants.my_ants()[ant_id], action)
        map_size = float(ants.cols * ants.rows)
        feats["bias"] = 1.0

        feats["obstacle"] = 0 if ants.passable(new_ant_loc) else 1

        if ants.food():
            food_d = min(ants.distance(new_ant_loc, f) for f in ants.food()) / map_size
            feats["food"] = food_d
            feats["will-eat-food"] = 1 if food_d == 0 else 0

        if ants.enemy_hills():
            enemy_hill_d = min(ants.distance(new_ant_loc, h[0]) for h in ants.enemy_hills()) / map_size
            feats["enemy-hill"] = enemy_hill_d
            feats["will-step-on-enemy-hill"] = 1 if enemy_hill_d == 0 else 0

        feats.divideAll(10.0)

        return feats

def get_reward(prev_ants, prev_actions, ants, ant_id):
    reward = 0
    new_ant_loc = prev_ants.destination(prev_ants.my_ants()[ant_id], prev_actions[ant_id])
    # dead or tried to walk through barrier TODO: how to identify collision between two ants.
    # if new_ant_loc not in ants.my_ants():
    #     reward -= 1

    # eating food
    if new_ant_loc in prev_ants.food():
        reward += 100

    # stepping on enemy hill
    if new_ant_loc in prev_ants.enemy_hills():
        reward += 5

    # stepping on our hill
    if new_ant_loc in ants.my_hills():
        reward += -1

    # util.printErr(reward)
    return reward

def action_fn(state):
    actions = []
    ants = state[0]
    ant_id = state[1]
    if ant_id is None:
        return actions
    for d in AIM:
        if ants.unoccupied(ants.destination(ants.my_ants()[ant_id], d)):
            actions.append(d)
    return actions

class ApproxQBot:
    def __init__(self, train, load_from_file):
        self.agent = ApproximateQAgent(actionFn=action_fn, extractor=BasicExtractor, epsilon=0.5, alpha=0.5, gamma=1)
        if load_from_file:
            sys.stderr.write("Loaded Q from file\n")
            with open("Q.txt",   'rb') as f:
                Q = pickle.load(f)
            self.agent.setQ(Q)
        self.last_state, self.last_action = None, None
        self.train = train

    def do_setup(self, ants):
        self.last_state, self.last_action = None, None



    def do_turn(self, ants):

        state = ants  # for approxQ, states are not stored in the dictionary, only weights of features.

        if self.train and self.last_state:
            for ant_id in range(len(self.last_state.my_ants())):
                next_loc = state.destination(self.last_state.my_ants()[ant_id], self.last_action[ant_id])
                reward = get_reward(self.last_state, self.last_action, state, ant_id)
                next_id = state.my_ants().index(next_loc) if next_loc in state.my_ants() else None
                self.agent.update((self.last_state, ant_id), self.last_action[ant_id], (state, next_id), reward)

        actions = []
        if self.train:
            for ant_id in range(len(state.my_ants())):
                actions.append(self.agent.getAction((state, ant_id)))
        else:
            for ant_id in range(len(state.my_ants())):
                actions.append(self.agent.getPolicy((state, ant_id)))


        # Issuing new orders
        for ant_id in range(len(state.my_ants())):
            ants.issue_order((state.my_ants()[ant_id], actions[ant_id]))

        # saving state and actions for next turn
        self.last_state = copy.deepcopy(state)
        self.last_action = actions


    def do_endgame(self, ants):
        state = ants  # for approxQ, states are not stored in the dictionary, only weights of features.

        if self.train and self.last_state:
            for ant_id in range(len(self.last_state.my_ants())):
                next_loc = state.destination(self.last_state.my_ants()[ant_id], self.last_action[ant_id])
                reward = get_reward(self.last_state, self.last_action, state, ant_id)
                next_id = state.my_ants().index(next_loc) if next_loc in state.my_ants() else None
                self.agent.update((self.last_state, ant_id), self.last_action[ant_id], (state, next_id), reward)


    def save_q(self):
        if self.train:
            with open("Q.txt", 'wb') as f:
                pickle.dump(self.agent.Q, f)

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
                # sys.stderr.write("finished game\n")
                rounds += 1
                if rounds == training_rounds:
                    sys.stderr.write("training over. dumping Q\n")
                    bot.save_q()
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
