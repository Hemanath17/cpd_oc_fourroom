# agents/option_critic.py
import numpy as np

def softmax(x, temp=1.0):
    z = x / max(temp, 1e-12)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class TabularOptionCritic:
    """
    Tabular OC with:
     - QU(s, ω, a) table (critic)
     - θ(s, ω, a) for intra-option Boltzmann policy
     - ϑ(s, ω) for termination sigmoid
     - Greedy policy over options (πΩ)
    """
    def __init__(self, n_states, n_actions, n_options, gamma,
                 alpha_critic=0.5, alpha_theta=0.25, alpha_beta=0.25,
                 temperature=0.001, epsilon_option=0.0, seed=0):
        self.nS = n_states
        self.nA = n_actions
        self.nO = n_options
        self.gamma = gamma
        self.alpha_c = alpha_critic
        self.alpha_th = alpha_theta
        self.alpha_b = alpha_beta
        self.temperature = temperature
        self.eps_opt = epsilon_option

        self.rng = np.random.RandomState(seed)

        # critic: QU(s, ω, a)
        self.QU = np.zeros((self.nS, self.nO, self.nA), dtype=np.float32)
        # actor: θ(s, ω, a)
        self.theta = np.zeros((self.nS, self.nO, self.nA), dtype=np.float32)
        # termination: ϑ(s, ω)
        self.vartheta = np.zeros((self.nS, self.nO), dtype=np.float32)

        self.current_option = None

    def beta(self, s):
        # βω(s) for all ω
        return sigmoid(self.vartheta[s])

    def pi_omega(self, s, omega):
        # intra-option policy: Boltzmann over theta[s, omega, :]
        return softmax(self.theta[s, omega], temp=self.temperature)

    def q_option(self, s, omega):
        # QΩ(s, ω) = Σ_a πω(a|s) QU(s, ω, a)
        pi = self.pi_omega(s, omega)
        return np.dot(pi, self.QU[s, omega])

    def q_option_all(self, s):
        # vector of QΩ(s, ω) for all ω
        return np.array([self.q_option(s, w) for w in range(self.nO)], dtype=np.float32)

    def v_option(self, s):
        # VΩ(s) = max_ω QΩ(s, ω) since greedy πΩ
        return np.max(self.q_option_all(s))

    def select_option(self, s):
        q_vals = self.q_option_all(s)
        if self.rng.rand() < self.eps_opt:
            return self.rng.randint(self.nO)
        return int(np.argmax(q_vals))

    def select_action(self, s, omega):
        pi = self.pi_omega(s, omega)
        return int(self.rng.choice(self.nA, p=pi))

    def should_terminate(self, s, omega):
        b = self.beta(s)[omega]
        return self.rng.rand() < b

    def start_episode(self, s):
        # pick an initial option greedily
        self.current_option = self.select_option(s)

    def step(self, s, a, r, s_next, done):
        """
        Perform one tabular OC update (Algorithm 1 in appendix of the paper)
        Returns the active option after update (might switch).
        """
        omega = self.current_option

        # 1) Critic (intra-option Q-learning style)
        beta_next = self.beta(s_next)[omega] if not done else 0.0

        q_omega_next = 0.0
        q_max_next = 0.0
        if not done:
            q_omega_next = self.q_option(s_next, omega)  # stay
            q_max_next = np.max(self.q_option_all(s_next))  # switch
        target = r + self.gamma * ((1 - beta_next) * q_omega_next + beta_next * q_max_next) * (0.0 if done else 1.0)

        delta = target - self.QU[s, omega, a]
        self.QU[s, omega, a] += self.alpha_c * delta

        # 2) Policy improvement (intra-option gradient)
        # grad log πω(a|s) for tabular softmax over theta[s, omega, :]
        pi = self.pi_omega(s, omega)
        grad_log_pi = -pi
        grad_log_pi[a] += 1.0
        # chain rule: dθ <- αθ * QU(s, ω, a) * grad_log_pi
        # baseline: QΩ(s, ω)
        adv = self.QU[s, omega, a] - self.q_option(s, omega)
        self.theta[s, omega] += self.alpha_th * adv * grad_log_pi


        # 3) Termination gradient
        # AΩ(s_next, ω) = QΩ(s_next, ω) - VΩ(s_next)
        if not done:
            xi = 0.01  # paper: small regularizer
            A = self.q_option(s_next, omega) - self.v_option(s_next) + xi

            b = self.beta(s_next)[omega]
            db_dvartheta = b * (1 - b)
            self.vartheta[s_next, omega] -= self.alpha_b * db_dvartheta * A

        # 4) Option termination and switch
        if not done:
            if self.should_terminate(s_next, omega):
                self.current_option = self.select_option(s_next)
        return self.current_option
