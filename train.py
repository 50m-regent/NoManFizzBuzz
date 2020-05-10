from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from collections import deque
import numpy as np

from fizzbuzz import FizzBuzz

class Model:
    def __init__(self):
        learning_rate = 0.01
        state_size    = 15
        action_size   = 4
        hidden_size   = 16

        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(action_size, activation='softmax'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.summary()

    def replay(self, memory, batch_size, gamma, target_model):
        inputs     = np.zeros((batch_size, 15))
        outputs    = np.zeros((batch_size, 4))
        mini_batch = memory.sample(batch_size)

        for i, (state, action, reward, next_state) in enumerate(mini_batch):
            inputs[i:i + 1] = state
            target          = reward

            if not (next_state == np.zeros(state.shape)).all():
                retmainQs = self.model.predict(next_state.reshape(1, 15))[0].argmax()
                next_action = np.argmax(retmainQs)
                target = reward + gamma * target_model.model.predict(next_state.reshape(1, 15))[0][next_action]

            outputs[i] = self.model.predict(state.reshape(1, 15))
            outputs[i][action.argmax()] = target

        self.model.fit(inputs, outputs, epochs=1, verbose=0)

class Memory:
    def __init__(self):
        self.buffer = deque()

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        indice = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indice]

class Agent:
    def get_action(self, state, epoch, main_model):
        epsilon = 0.001 + 0.9 / (1.0 + epoch)

        if epsilon < np.random.uniform(0, 1):
            action = main_model.model.predict(state.reshape(1, 15))[0].argmax()
        else:
            action = np.random.choice([0, 1, 2, 3])

        return to_categorical(action, 4)

def evaluate(env):
    env.reset()
    state, _, finished = env.random_step()
    while not finished:
        action = agent.get_action(state, N_EPOCHS, main_model)
        next_state, _, finished = env.step(action.argmax(), verbose=True)
        state = next_state

if __name__ == '__main__':
    N_EPOCHS = 5000
    S_BATCH  = 4
    GAMMA    = 0.99

    env = FizzBuzz(1, 1000)

    main_model   = Model()
    target_model = Model()

    memory = Memory()
    agent  = Agent()

    learned_flag = False

    for epoch in range(N_EPOCHS):
        if learned_flag:
            break

        print('Epoch: {}'.format(epoch + 1))

        env.reset()
        state, reward, finished = env.random_step()

        target_model.model.set_weights(main_model.model.get_weights())

        while not finished:
            action = agent.get_action(state, epoch, main_model)
            learned_flag = env.is_learned()
            next_state, reward, finished = env.step(action.argmax())

            memory.add((state, action, reward, next_state))

            state = next_state

            if len(memory.buffer) > S_BATCH:
                main_model.replay(memory, S_BATCH, GAMMA, target_model)

            target_model.model.set_weights(main_model.model.get_weights())

    env = FizzBuzz(500, 1000)
    evaluate(env)