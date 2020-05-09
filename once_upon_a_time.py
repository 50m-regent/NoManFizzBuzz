class FizzBuzz:
    def __init__(self, start, end):
        self.LIST = [
            'FizzBuzz',
            0,
            0,
            'Fizz',
            0,
            'Buzz',
            'Fizz',
            0,
            0,
            'Fizz',
            'Buzz',
            0,
            'Fizz',
            0,
            0
        ]

        self.start = start
        self.end = end

    def run(self):
        fizzbuzz = ''
        for i in range(self.start, self.end + 1):
            fizzbuzz += '{} {}\n'.format(i, self.LIST[i % 15])

        print(fizzbuzz, end='')

if __name__ == '__main__':
    start = int(input('Start: '))
    end   = int(input('End: '))
    
    fb = FizzBuzz(start, end)
    fb.run()