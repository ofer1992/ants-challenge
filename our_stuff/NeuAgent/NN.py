import numpy as np
import pickle
import sys

resume = False  # resume from previous checkpoint?

class NN:
    # hyperparameters
    H = 200  # number of hidden layer neurons
    A = 4  # size of action space
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2


    def __init__(self, D):
        """
        :param D: rows * cols, input dimensionality
        """
        # model initialization
        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = {'W1': np.random.randn(self.H, D) / np.sqrt(D),
                          'W2': np.random.randn(self.A, self.H) / np.sqrt(self.H)}

        self.grad_buffer = {k: np.zeros_like(v) for k, v in
                       self.model.iteritems()}  # update buffers that add up gradients over a batch
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # rmsprop memory

        self.xs, self.hs, self.dlogps, self.drs = [], [], [], []  # reset array memory
        self.actions_in_turn = 0 # basically number of ants in current turn
        self.actions_in_turn_array = [] # holds for each turn in game the number of ants. used for discounted reward calculation.
        self.reward_sum = 0
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

    @staticmethod
    def softmax(x):
        p = x - np.max(x)  # normalization trick to avoid numeric instability
        return np.exp(p) / np.sum(np.exp(p))

    def prepro(self, I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def discount_rewards(self,r):
        """ take 1D float array of rewards and compute discounted reward """
        assert r.dtype == 'float'
        discounted_r = np.zeros_like(r)
        running_add = 0
        i = len(self.actions_in_turn_array) - 1
        first_in_turn = True
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0 and first_in_turn: running_add = 0  # reset the sum, since this was a game boundary (pong specific!) TODO: modify
            if first_in_turn:
                running_add = running_add * self.gamma + r[t]
                first_in_turn = False
            if self.actions_in_turn_array[i] <= 1:
                first_in_turn = True
                i -= 1
            else:
                self.actions_in_turn_array[i] -= 1
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)  # OFER: why logp?
        p = self.softmax(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epdlogp, epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).T
        # dh = np.outer(epdlogp, self.model['W2'])
        dh = np.dot(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}

    def get_action(self, observation):
        self.actions_in_turn +=1
        x = observation # TODO: add preprocessing?

        # forward the policy network and sample an action from the returned probability
        prob_vec, h = self.policy_forward(x)
        action = np.random.choice(range(4), p=prob_vec)

        # record various intermediates (needed later for backprop)
        self.xs.append(x)  # observation
        self.hs.append(h)  # hidden state
        y = np.zeros_like(prob_vec)  # a "fake label"
        y[action] = 1
        self.dlogps.append( # TODO: consider how the change from scalar in original code to vector affects computation
            y - prob_vec)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        return action

    def step(self, reward, done):
        # step the environment and get new measurements
        self.reward_sum += reward
        self.actions_in_turn_array.append(self.actions_in_turn)
        self.drs.extend([reward] * self.actions_in_turn)  # record reward (has to be done after we call step() to get reward for previous action)
        self.actions_in_turn = 0
        if done:  # an episode finished
            self.episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(self.xs)
            eph = np.vstack(self.hs)
            epdlogp = np.vstack(self.dlogps)
            epr = np.vstack(self.drs)
            self.xs, self.hs, self.dlogps, self.drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = self.discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.) # TODO: I suspect this will not work now that dlogp is a vector
            grad = self.policy_backward(eph, epdlogp, epx)
            for k in self.model: self.grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if self.episode_number % self.batch_size == 0:
                for k, v in self.model.iteritems():
                    g = self.grad_buffer[k]  # gradient
                    self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2
                    self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                    self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            # boring book-keeping
            running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, running_reward)
            if self.episode_number % 100 == 0:
                sys.stderr.write("Dumping weights\n")
                pickle.dump(self.model, open('save.p', 'wb'))
            self.reward_sum = 0

        if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')