from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from collections import deque
import numpy as np

from fizzbuzz import FizzBuzz

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=15, action_size=4, hidden_size=16):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(action_size, activation='softmax'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.summary()

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs     = np.zeros((batch_size, 15))
        targets    = np.zeros((batch_size, 4))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all():
                retmainQs = self.model.predict(next_state_b.reshape(1, 15))[0].argmax()
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b.reshape(1, 15))[0][next_action]

            targets[i] = self.model.predict(state_b.reshape(1, 15))
            targets[i][action_b.argmax()] = target

        self.model.fit(inputs, targets, epochs=1, verbose=0)

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

class Agent:
    def get_action(self, state, epoch, main_qn):
        epsilon = 0.001 + 0.9 / (1.0 + epoch)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = main_qn.model.predict(state.reshape(1, 15))[0]
            action = np.argmax(retTargetQs)
        else:
            action = np.random.choice([0, 1, 2, 3])

        return to_categorical(action, 4)

    def save(self):
        return

if __name__ == '__main__':
    N_EPOCHS = 10000
    S_BATCH  = 4
    GAMMA    = 0.99

    env = FizzBuzz(1, 100)

    main_qn   = QNetwork(learning_rate=0.0001)
    target_qn = QNetwork(learning_rate=0.0001)

    memory = Memory(max_size=10000)

    agent = Agent()

    learned_flag = False

    for epoch in range(N_EPOCHS):
        if learned_flag:
            break

        print('Epoch: {}'.format(epoch + 1))

        env.reset()
        state, reward, terminal = env.random_step()

        target_qn.model.set_weights(main_qn.model.get_weights())

        while not terminal:
            action = agent.get_action(state, epoch, main_qn)
            learned_flag = env.is_learned()
            next_state, reward, terminal = env.step(action.argmax())

            memory.add((state, action, reward, next_state))

            state = next_state

            if memory.len() > S_BATCH:
                main_qn.replay(memory, S_BATCH, GAMMA, target_qn)

            target_qn.model.set_weights(main_qn.model.get_weights())

    env.reset()
    state, _, terminal = env.random_step()
    while not terminal:
        action = agent.get_action(state, 10000, main_qn)
        next_state, _, terminal = env.step(action.argmax(), verbose=True)
        state = next_state

    agent.save()