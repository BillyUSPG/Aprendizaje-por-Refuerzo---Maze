import logging  
import random  
from datetime import datetime  

import numpy as np  

from environment import Status  
from models import AbstractModel  


class QTableTraceModel(AbstractModel):

    default_check_convergence_every = 5 

    def __init__(self, game, **kwargs):

        super().__init__(game, name="QTableTraceModel", **kwargs)  
        self.Q = dict()  

    def train(self, stop_at_convergence=False, **kwargs):
        
        
        discount = kwargs.get("discount", 0.90)  
        exploration_rate = kwargs.get("exploration_rate", 0.10)  
        exploration_decay = kwargs.get("exploration_decay", 0.995)  
        learning_rate = kwargs.get("learning_rate", 0.10)  
        eligibility_decay = kwargs.get("eligibility_decay", 0.80)  
        episodes = max(kwargs.get("episodes", 1000), 1)  
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()  
        start_time = datetime.now()  

        for episode in range(1, episodes + 1):
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)  
            state = tuple(state.flatten())  

            etrace = dict()  

            while True:
               
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)  
                else:
                    action = self.predict(state)  

                try:
                    etrace[(state, action)] += 1
                except KeyError:
                    etrace[(state, action)] = 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward  

                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                delta = reward + discount * max_next_Q - self.Q[(state, action)]

                for key in etrace.keys():
                    self.Q[key] += learning_rate * delta * etrace[key]

                for key in etrace.keys():
                    etrace[key] *= (discount * eligibility_decay)

                if status in (Status.WIN, Status.LOSE):
                    break

                state = next_state  

                self.environment.render_q(self)  

            cumulative_reward_history.append(cumulative_reward)  


            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))


            if episode % check_convergence_every == 0:
                w_all, win_rate = self.environment.check_win_all(self)  
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        q = self.q(state)  

        logging.debug("q[] = {}".format(q))  

        actions = np.nonzero(q == np.max(q))[0]  
        return random.choice(actions) 
