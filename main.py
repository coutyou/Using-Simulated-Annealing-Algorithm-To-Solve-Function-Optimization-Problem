import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from FunctionAnnealer import *


ITER_TIMES = 20


def get_init_state():
    state = []
    for i in range(STATE_LEN):
        state.append(np.random.uniform(STATE_MIN[i], STATE_MAX[i]))
    return np.array(state)


def state_valid(state):
    valid = True
    for i in range(STATE_LEN):
        if state[i] < STATE_MIN[i] or state[i] > STATE_MAX[i]:
            valid = False
            break
    return valid


def ackley_func(state):
    a = 20
    b = 0.2
    c = 2 * np.pi

    def exp1(state):
        sum_ = 0
        for num in state:
            sum_ += num ** 2
        return np.exp(- b * np.sqrt(sum_ / STATE_LEN))

    def exp2(state):
        sum_ = 0
        for num in state:
            sum_ += np.cos(c * num)
        return np.exp(sum_ / STATE_LEN)

    return - a * exp1(state) - exp2(state) + a + np.exp(1)


if __name__ == '__main__':
    best_energy = 1e10
    best_state = None
    best_record = None
    es = []
    for _ in range(ITER_TIMES):
        init_state = get_init_state()

        fa = FunctionAnnealer(init_state, ackley_func, state_valid, LR)
        state, e, record = fa.anneal()

        es.append(e)
        if e < best_energy:
            best_energy = e
            best_state = state
            best_record = record
        
        print()

    print(f"\nBest Result: {min(es)}    Worst Result: {max(es)}   Avg Of Results: {np.mean(es)}    Var Of Results: {np.var(es)}")
    print(f"Best State: {best_state}")

    plt.figure()
    plt.plot(range(len(best_record)), best_record)
    plt.xlabel("Times Of Update")
    plt.ylabel("Energy")
    plt.title(f"Energy Curve")
    
    plt.show() 