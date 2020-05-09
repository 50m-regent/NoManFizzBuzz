#include <stdio.h>
#include <string>

class FizzBuzz {
private:
    int start, end;
    const std::string LIST[15] = {
        "FizzBuzz",
        "0",
        "0",
        "Fizz",
        "0",
        "Buzz",
        "Fizz",
        "0",
        "0",
        "Fizz",
        "Buzz",
        "0",
        "Fizz",
        "0",
        "0"
    };
public:
    FizzBuzz(int start, int end) {
        this->start = start;
        this->end   = end;
    }

    void run() {
        for (int i = start; i <= end; i++) {
            printf("%d %s\n", i, LIST[i % 15]);
        }
    }
};

int main() {
    int start, end;
    scanf("%d", &start);
    scanf("%d", &end);

    FizzBuzz fb(start, end);
    fb.run();
}