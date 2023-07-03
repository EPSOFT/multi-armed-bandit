import time

import matplotlib.pyplot as plt
import numpy as np

K = 5
TOTAL_PLAYS = 500
TOTAL_STEPS = 2000


class Bandit:
    def __init__(self, arms: int, sample_count: int, total_steps: int, total_plays: int, mean_range, std):
        self.arms = arms
        self.mean_range = mean_range
        self.std = std
        self.total_steps = total_steps
        self.total_plays = total_plays
        self.sample_count = sample_count

        self.samples = None
        self.reward_history = None
        self.chose_optimal_history = None
        self.optimal_action = None

        self.ave_reward = self.arm_pulled = None
        self.reset()
        self.reset_history()
        self.rng = np.random.default_rng(seed=0)
        self.init_random_samples()

    # =========================================   LifeCycle  =========================================

    def init_random_samples(self):
        means = self.rng.uniform(self.mean_range[0], self.mean_range[1], size=(1, self.arms))
        self.optimal_action = np.argmax(means, axis=1)
        self.samples = means + self.rng.normal(0, self.std, size=(self.sample_count, self.arms))

    def reset(self):
        self.ave_reward = np.zeros((1, self.arms))
        self.arm_pulled = np.zeros((1, self.arms))

    def reset_history(self):
        self.reward_history = np.zeros((self.total_plays, self.total_steps))
        self.chose_optimal_history = np.zeros((self.total_plays, self.total_steps), dtype=np.bool8)
    # =========================================   Plot  =========================================

    def plot_distributions(self):
        plt.violinplot(dataset=self.samples)
        plt.xlabel('Action')
        plt.ylabel('Reward distribution')
        plt.title(f'Rewards distribution for {self.arms} armed bandit')
        plt.show()

    # =========================================   Action  =========================================

    def select_eps_greedy_action(self, eps):
        random_guess = self.rng.random()
        if random_guess > eps:  # Greedy action
            first_argmax = self.ave_reward.max(axis=1)
            max_indices = np.argwhere(self.ave_reward == first_argmax)[:, 1]
            if (len(max_indices)) > 1:
                return self.rng.choice(max_indices)
            else:
                return max_indices[0]
        else:
            return self.rng.choice(self.arms)  # Random action

    def get_max_ave_reward(self):
        return self.ave_reward.max(axis=1)

    # =========================================   PLay  =========================================

    def step(self, eps, step, play):
        action = self.select_eps_greedy_action(eps)
        reward = self.samples[self.rng.choice(self.sample_count), action]
        alpha = 1 / (self.arm_pulled[0, action] + 1)
        self.ave_reward[0, action] += alpha * (reward - self.ave_reward[0, action])
        self.arm_pulled[0, action] += 1
        self.reward_history[play, step] = reward
        self.chose_optimal_history[play, step] = action == self.optimal_action

    def play_n_step(self, steps, eps, play):
        for step in range(steps):
            self.step(eps, step, play)


if __name__ == '__main__':

    def plot_ave_reward(ave_rewards, epsilons):
        fig, ax = plt.subplots()
        for i in range(len(ave_rewards)):
            ax.plot(np.mean(ave_rewards[i], axis=0), label=f'eps={epsilons[i]}')
            plt.xlabel('steps')
            plt.ylabel('Reward')
        plt.title(f'Average rewards for different epsilons')
        plt.legend()
        fig.show()


    def plot_optimal_chosen(chosen_optimal):
        fig, ax = plt.subplots()
        for i in range(len(chosen_optimal)):
            ax.plot(np.sum(chosen_optimal[i], axis=0)/TOTAL_PLAYS, label=f'eps={epsilons[i]}')
            plt.xlabel('steps')
            plt.ylabel('Chose optimal (%)')
        plt.title(f'Percentage of optimal action choice')
        plt.legend()
        fig.show()


    start_time = time.time()

    bandit = Bandit(K, 2000, TOTAL_STEPS, TOTAL_PLAYS, (10, 50), 10)
    bandit.plot_distributions()

    epsilons = [0.1, 0.2, 0.5]
    ave_reward_hist = []
    chosen_optimal_hist=[]
    for epsilon in epsilons:
        for play in range(TOTAL_PLAYS):
            bandit.play_n_step(TOTAL_STEPS, epsilon, play)
            bandit.reset()

        ave_reward_hist.append(bandit.reward_history)
        chosen_optimal_hist.append(bandit.chose_optimal_history)
        bandit.reset_history()

    print(f'Execution took {time.time() - start_time:0.3f} seconds')

    plot_ave_reward(ave_reward_hist, epsilons)
    plot_optimal_chosen(chosen_optimal_hist)
