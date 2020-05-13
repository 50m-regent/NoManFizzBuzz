from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
import numpy as np

class Model:
    def __init__(self):
        learning_rate = 0.01 # 学習率
        state_size    = 15   # 入力サイズ(今の状態)
        action_size   = 4    # 出力サイズ(0、1、2、3のいずれか)
        hidden_size   = 16   # 隠れ層の大きさ

        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(action_size, activation='softmax'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.summary()

    # 実際に学習させる関数
    def replay(self, memory, batch_size, gamma, target_model):
        inputs     = np.zeros((batch_size, 15))
        outputs    = np.zeros((batch_size, 4))
        mini_batch = memory.sample(batch_size)

        for i, (state, action, reward, next_state) in enumerate(mini_batch):
            inputs[i:i + 1] = state
            target          = reward

            if not (next_state == np.zeros(state.shape)).all():
                q = self.model.predict(next_state.reshape(1, 15))[0].argmax()
                next_action = np.argmax(q) # 最もQ値が高い行動を次の行動として選択
                target = reward + gamma * target_model.model.predict(
                    next_state.reshape(1, 15)
                )[0][next_action] # 実際の報酬

            # 現状の予想値を修正して学習させる
            outputs[i] = self.model.predict(state.reshape(1, 15))
            outputs[i][action.argmax()] = target

        self.model.fit(inputs, outputs, epochs=1, verbose=0)