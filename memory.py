from collections import deque
import numpy as np

class Memory:
    def __init__(self):
        self.buffer = deque()

    # 現在の状況、どう動いたか、その結果どうなったか、その行動の報酬を格納
    def add(self, exp):
        self.buffer.append(exp)

    # ランダムに格納されているデータを取り出す
    def sample(self, batch_size):
        indice = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indice]
