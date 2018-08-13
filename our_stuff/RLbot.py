#!/usr/bin/env python
from ants import *
from agents import QLearningAgent
import sys
import pickle


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
    return actions


def gen_state(ants):
    map = []
    for i in range(ants.rows):
        row = []
        for j in range(ants.cols):
            row.append(ants.map[i][j])
        map.append(tuple(row))

    return (tuple(map), ants.my_ants()[0], ants.rows, ants.cols)


def calc_reward(state, action, next_state):
    if action == 'n':
        return 1
    else:
         return 0

def reward2(ants):
    hills = [a[0] for a in ants.enemy_hills()]
    if ants.my_ants()[0] == (11, 5):
        sys.stderr.write("WON!\n")
        return 500
    return 0

def features(ants):
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

    return features

class MyBot:
    def __init__(self, train, load_from_file):
        self.agent = QLearningAgent(action_fn, 1, 0.9, 0.5, 1)
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

        state = gen_state(ants)

        if self.train and self.last_state:
            reward = reward2(ants)
            self.agent.update(self.last_state, self.last_action, state, reward)

        self.last_state = state
        if self.train:
            self.last_action = self.agent.getAction(state)
        else:
            self.last_action = self.agent.getPolicy(state)
        ants.issue_order((state[1], self.last_action))

    def do_endgame(self, ants):
        sys.stderr.write(str(ants.my_ants())+"\n")
        state = gen_state(ants)

        if self.train and self.last_state:
            reward = reward2(ants)
            self.agent.update(self.last_state, self.last_action, state, reward)


    def save_q(self):
        if self.train:
            with open("Q.txt", 'wb') as f:
                pickle.dump(self.agent.Q, f)


def run(bot, training_rounds):
    'parse input, update game state and call the bot classes do_turn method'
    ants = Ants()
    map_data = ''
    rounds = 0
    while(True):
        try:
            current_line = sys.stdin.readline() # string new line char
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
        b = MyBot(training_rounds > 0, training_rounds == 0)
        run(b, training_rounds)

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
