from fizzbuzz import FizzBuzz
from model import Model
from memory import Memory
from agent import Agent

def evaluate(env):
    env.reset()
    state, _, finished = env.random_step()
    while not finished:
        action = agent.get_action(state, N_EPOCHS, main_model)
        next_state, _, finished = env.step(action.argmax(), verbose=True)
        state = next_state

if __name__ == '__main__':
    N_EPOCHS = 5000 # 訓練回数
    S_BATCH  = 4    # バッチサイズ
    GAMMA    = 0.99 # 時間経過による報酬減少率

    env = FizzBuzz(1, 100) # 学習環境

    main_model   = Model()
    target_model = Model()

    memory = Memory() # メモリ
    agent  = Agent()  # エージェント

    learned_flag = False # 学習が終了したか否か

    for epoch in range(N_EPOCHS):
        if learned_flag:
            break

        print('Epoch: {}'.format(epoch + 1))

        # 初期状態設定
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

    env = FizzBuzz(4000, 5000)
    evaluate(env)