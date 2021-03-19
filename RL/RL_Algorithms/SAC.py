import numpy as np
import torch
from SAC_model import ActorNetwork, CriticNetwork
from algo import Algorithm, ReplayBuffer


class SAC(Algorithm):
    def __init__(self, state_shape, action_shape, seed=0,
                 batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 buffer_size=10 ** 4, start_steps=10 ** 4, tau=5e-3, min_alpha=0.05, reward_scale=1.0):
        super().__init__()

        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
        )
        self.actor = ActorNetwork(state_shape, action_shape).to(self.dev)
        self.critic = CriticNetwork(state_shape, action_shape).to(self.dev)

        # Target Network を用いて学習を安定化させる.
        self.critic_target = CriticNetwork(state_shape, action_shape).to(self.dev).eval()

        # adjust entropy(\alpha)
        self.min_alpha = torch.tensor(min_alpha)
        self.alpha = torch.tensor(min_alpha * 3.0, requires_grad=True)

        # init target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = torch.optim.Adam([self.alpha], lr=lr_alpha)


        # param
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.learning_steps = 0

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps):
        t += 1
        if steps <= self.start_steps:  # 最初はランダム.
            action = env.action_space.sample()
        else:
            action, _ = self.explore(state)
        n_state, rew, done, _ = env.step(action)

        # done_masked = False if t == env._max_episode_steps else done  # 最大ステップ数に到達してdone=Trueになった場合を補正する.
        # self.buffer.append(state, action, rew, done_masked, n_state)  # add data to buffer
        self.buffer.append(state, action, rew, done, n_state)  # add data to buffer
        if done:  # エピソードが終了した場合には，環境をリセットする．
            t = 0
            n_state = env.reset()
        return n_state, t

    def update_critic(self, states, actions, rews, dones, n_states):
        # (r(s,a) + \gamma V(s') - Q(s,a))^2 = (r(s,a) + \gamma {min[Q(s',a')] - \alpha \log \pi (a|s)} - Q(s,a))^2
        now_q1, now_q2 = self.critic(states, actions)
        with torch.no_grad():
            n_actions, log_pis = self.actor.sample(n_states, False)
            q1, q2 = self.critic_target(n_states, n_actions)
            target_vs = torch.min(q1, q2) - self.alpha * log_pis

        target_qs = self.reward_scale * rews + self.gamma * target_vs * (1.0 - dones)  # r(s,a) + \gamma V(s')
        # loss funcs
        loss_c1 = (now_q1 - target_qs).pow_(2).mean()
        loss_c2 = (now_q2 - target_qs).pow_(2).mean()
        # update
        self.optim_critic.zero_grad()
        (loss_c1 + loss_c2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        acts, log_pis = self.actor.sample(states)
        q1, q2 = self.critic(states, acts)
        loss_actor = (self.alpha * log_pis - torch.min(q1, q2)).mean()
        # update
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()
        self.entropy_adjust_func(log_pis)

    def update_target(self):
        for target, trained in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.mul_(1.0 - self.tau)
            target.data.add_(self.tau * trained.data)

    def update(self):
        self.learning_steps += 1
        states, actions, rews, dones, n_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rews, dones, n_states)
        self.update_actor(states)
        self.update_target()

    def entropy_adjust_func(self, log_pis):
        with torch.no_grad():
            loss = log_pis + self.min_alpha
        loss = -(self.alpha * loss).mean()
        self.optim_alpha.zero_grad()
        loss.backward()
        self.optim_alpha.step()