from our_stuff.naive_Q.ants import *
from our_stuff.agents import QLearningAgent
import sys
import pickle
import copy


id_to_action = {
    0: 'n',
    1: 'e',
    2: 's',
    3: 'w'
}


EATING_FOOD_REWARD = 5
ENEMY_HILL_REWARD = 2000
OWN_HILL_REWARD = -5
VISION_REWARD = 5
NEWLY_FOUND_FOOD_REWARD = 1
ANT_DEAD_REWARD = -10
ENEMY_ANT_DEAD_REWARD = 50


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
    # sys.stderr.write(str(new_food) + '\n')
    reward += new_food * NEWLY_FOUND_FOOD_REWARD

    # killing an enemy
    dead_enemies = prev_ants.attack_range_of_loc(prev_ant_loc, prev_ants)[2]
    reward += ENEMY_ANT_DEAD_REWARD * len(dead_enemies)


    reward -= 1

    return reward


class NaiveBot:
    def __init__(self, train, load_from_file):
        self.agent = QLearningAgent(Ants.action_func, 1, 0.9, 0.2, 0.99)
        if load_from_file:
            sys.stderr.write("Loaded Q from file\n")
            with open("Q.txt",   'rb') as f:
                Q = pickle.load(f)
            self.agent.setQ(Q)
        self.prev_state, self.last_action = None, None
        self.accumulated_rewards = 0
        self.train = train

    def do_setup(self, ants):
        self.prev_state, self.last_action = None, None
        self.accumulated_rewards = 0

    def do_turn(self, state):
        sys.stderr.write("Starting turn\n")
        # sys.stderr.write(str(state.my_ants())+"\n")
        if self.train and self.prev_state:
            for ant in self.prev_state.my_ants():
                prev_obs = self.prev_state.ant_observation(ant)
                action = self.prev_state.orders2[ant]
                curr_loc = state.destination_with_obstacles(ant, action)
                if curr_loc not in state.ant_list or state.ant_list[curr_loc] !=MY_ANT:
                    curr_obs = None
                else:
                    curr_obs = state.ant_observation(curr_loc)
                reward = get_reward(self.prev_state, ant, state, curr_loc)
                self.accumulated_rewards += reward
                self.agent.update(prev_obs, action, curr_obs, reward)

        action_function = self.agent.getAction if self.train else\
            self.agent.getPolicy
        for ant in state.my_ants():
            obs = state.ant_observation(ant)
            action = action_function(obs)
            # state.remember_order(ant, action)
            state.issue_order((ant, action))

        # saving state and actions for next turn
        self.prev_state = copy.deepcopy(state)
        # sys.stderr.write(str(state.orders2)+"\n")
        # sys.stderr.write(str(state.my_ants())+"\n")


    def do_endgame(self, state):
        """
        Called once in end of game to update Q-weights.
        """
        sys.stderr.write("Total reward: "+str(self.accumulated_rewards)+"\n")
        sys.stderr.write("Turns: "+str(state.turns_so_far)+"\n")
        if self.train and self.prev_state:
            for ant in self.prev_state.my_ants():
                prev_obs = self.prev_state.ant_observation(ant)
                action = self.prev_state.orders2[ant]
                curr_loc = state.destination_with_obstacles(ant, action)
                if curr_loc not in state.ant_list or state.ant_list[curr_loc] !=MY_ANT:
                    curr_obs = None
                else:
                    curr_obs = state.ant_observation(curr_loc)
                reward = get_reward(self.prev_state, ant, state, curr_loc)
                self.accumulated_rewards += reward
                self.agent.update(prev_obs, action, curr_obs, reward)


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
                score = 0
                rounds += 1
                if rounds == training_rounds:
                    sys.stderr.write("training over. dumping Q\n")
                    bot.save_q()
                    bot.train = False
                current_line = sys.stdin.readline().rstrip('\r\n')
                while not current_line.startswith('playerturns'):
                    # sys.stderr.write(current_line+" inside loop 1\n")
                    current_line = sys.stdin.readline().rstrip('\r\n')
                current_line = sys.stdin.readline().rstrip('\r\n')
                while current_line != 'go':
                    map_data += current_line + '\n'
                    current_line = sys.stdin.readline().rstrip('\r\n')
                ants.update(map_data)
                bot.do_endgame(ants)
                # bot.agent.epsilon *= 0.99
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
        b = NaiveBot(training_rounds > 0, training_rounds == 0)
        run(b, training_rounds)

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
