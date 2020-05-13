from random import randint
from numpy import array

class FizzBuzz:
    def __init__(self, start, end):
        self.start = start # 始まり
        self.end   = end   # 終わり

    def reset(self): # 初期化
        self.turn = self.start # 今の数字
        self.state = []        # 今の状況

        # 初期状況を設定(過去15ターン)
        for i in range(self.turn - 15, self.turn):
            if i % 15 == 0:
                self.state.append(3)
            elif i % 3 == 0:
                self.state.append(1)
            elif i % 5 == 0:
                self.state.append(2)
            else:
                self.state.append(0)

    # 十分学習できたかどうか
    def is_learned(self):
        return self.turn == self.end

    # ターン遷移
    def step(self, action, verbose=False):
        if verbose:
            print(self.turn, [self.turn, 'Fizz', 'Buzz', 'FizzBuzz'][action], end=', ')

        reward   = 0     # 得点
        terminal = False # ゲームが終了したかどうか

        self.state = self.state[1:] + [action]

        if action == 1:   # Fizz
            if self.turn % 3 == 0 and self.turn % 5 != 0:
                reward   = 1
                terminal = False
            else:
                reward   = -1
                terminal = True
        elif action == 2: # Buzz
            if self.turn % 5 == 0 and self.turn % 3 != 0:
                reward   = 1
                terminal = False
            else:
                reward   = -1
                terminal = True
        elif action == 3: # FizzBuzz
            if self.turn % 15 == 0:
                reward   = 1
                terminal = False
            else:
                reward   = -1
                terminal = True
        else:             # Number
            if self.turn % 3 != 0 and self.turn % 5 != 0:
                reward   = 1
                terminal = False
            else:
                reward   = -1
                terminal = True

        if self.turn == self.end:
            terminal = True

        self.turn += 1

        return array(self.state), reward, terminal

    def random_step(self): # 初期化用
        return array(self.state), 0, False