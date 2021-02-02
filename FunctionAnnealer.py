import math
import random
import sys
import time
import numpy as np

STATE_MAX = [40, 40]
STATE_MIN = [-40, -40]
STATE_LEN = 2
LR = 1

def time_string(seconds):
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return '%4i:%02i:%02i' % (h, m, s)


class FunctionAnnealer(object):
    def __init__(self, initial_state, func, state_valid, lr):
        self.Tmax = 1000
        self.Tmin = 0.2
        self.steps = 1500000
        self.updates = 1000

        self.best_state = None
        self.best_energy = None
        self.start = None
        self.energy_record = []

        self.state = initial_state.copy()
        self.func = func
        self.state_valid = state_valid
        self.lr = lr

    def move(self):
        initial_energy = self.energy()

        while True:
            rand_data = np.random.normal(0, 1, STATE_LEN)
            if self.state_valid(self.state + self.lr * rand_data):
                self.state += self.lr * rand_data
                break

        return self.energy() - initial_energy

    def energy(self):
        return self.func(self.state)

    def update(self, step, T, E, acceptance, improvement):
        elapsed = time.time() - self.start
        if step == 0:
            print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
                  file=sys.stderr)
            print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
                  .format(Temp=T,
                          Energy=E,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def anneal(self):
        step = 0
        self.start = time.time()

        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        T = self.Tmax
        E = self.energy()
        prevState = self.state.copy()
        prevEnergy = E
        self.best_state = self.state.copy()
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        while step < self.steps:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.state = prevState.copy()
                E = prevEnergy
            else:
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.state.copy()
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.state.copy()
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(
                        step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0
                    self.energy_record.append(E)

        return self.best_state, self.best_energy, self.energy_record