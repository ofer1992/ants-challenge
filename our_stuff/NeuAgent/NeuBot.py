from our_stuff.NeuAgent.ants import *
import sys
import pickle
import copy
from NN import NN

id_to_action = {
    0: 'n',
    1: 'e',
    2: 's',
    3: 'w'
}

class NeuroBot:
    def __init__(self):
        self.agent = NN(OBSERVABILITY_SQUARE_SIZE**2*5)

    def do_setup(self, ants):
        pass

    def do_turn(self, state):
        self.agent.step(0., False) # reward for last turn. currently, reward is only victory or defeat/draw.

        for ant in state.my_ants():
            obs = state.ant_observation(ant).flatten()
            action = self.agent.get_action(obs)
            state.issue_order((ant, id_to_action[action]))


    def do_endgame(self, score):
        """
        Called once in end of game to update Q-weights.
        """
        self.agent.step(float(score), True) # reward for last turn. currently, reward is only victory or defeat/draw.

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
                score = 0
                rounds += 1
                current_line = sys.stdin.readline().rstrip('\r\n')
                while not current_line.startswith('playerturns'):
                    if current_line.startswith('score'):
                        line = current_line.split()
                        # sys.stderr.write(current_line+"\n")
                        score = int(line[1]) - 1.1
                    current_line = sys.stdin.readline().rstrip('\r\n')
                current_line = sys.stdin.readline().rstrip('\r\n')
                while current_line != 'go':
                    map_data += current_line + '\n'
                    current_line = sys.stdin.readline().rstrip('\r\n')
                ants.update(map_data)
                bot.do_endgame(score)
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
        b = NeuroBot()
        run(b, training_rounds)

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
