from random import randint
from numpy import array

class FizzBuzz:
    def __init__(self, start, end):
        self.start = start
        self.end   = end

    def reset(self):
        self.turn = self.start
        self.state = []

        for i in range(self.turn - 15, self.turn):
            if i % 15 == 0:
                self.state.append(3)
            elif i % 3 == 0:
                self.state.append(1)
            elif i % 5 == 0:
                self.state.append(2)
            else:
                self.state.append(0)

    def is_learned(self):
        return self.turn == self.end

    def step(self, action, verbose=False):
        if verbose:
            print(self.turn, [self.turn, 'Fizz', 'Buzz', 'FizzBuzz'][action])

        reward   = 0
        terminal = False

        self.state = self.state[1:] + [action]

        if action == 1:   # Fizz
            if self.turn % 3 == 0:
                reward   = 1
                terminal = False
            else:
                reward   = -1
                terminal = True
        elif action == 2: # Buzz
            if self.turn % 5 == 0:
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

    def random_step(self):
        return array(self.state), 0, False