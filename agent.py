import numpy as np
from keras.utils.np_utils import to_categorical

class Agent:
    # 行動を選択
    def get_action(self, state, epoch, main_model):
        epsilon = 0.001 + 0.9 / (1.0 + epoch)

        if epsilon < np.random.uniform(0, 1):
            action = main_model.model.predict(state.reshape(1, 15))[0].argmax()
        else: # ある程度の確率でランダム動作をする
            action = np.random.choice([0, 1, 2, 3])

        return to_categorical(action, 4)
