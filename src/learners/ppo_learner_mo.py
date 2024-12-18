import copy
from collections import deque

import numpy as np
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry

class PPOLearnerMo:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_objectives = args.n_objectives
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        # Lexicographic parameters
        self.mu = np.zeros((self.n_agents, self.n_objectives-1))
        self.j = np.zeros((self.n_agents, self.n_objectives-1))

        self.recent_losses = [[deque(maxlen=50) for _ in range(self.n_objectives)] for _ in range(self.n_agents)]
        # TODO: assert args.beta_values and args.eta_values shapes
        self.beta = np.array([self.args.beta_values for _ in range(self.n_agents)], dtype=np.float32)
        self.eta = np.array([[self.args.eta_values for _ in range(self.n_objectives-1)] for _ in range(self.n_agents)], dtype=np.float32)
        print("Beta: ", self.beta)
        print("Eta: ", self.eta)
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        assert self.n_objectives > 1 and self.args.common_reward == False
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,self.n_objectives), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(self.n_agents,self.n_objectives), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                    rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        mo_mask = copy.copy(mask)
        mo_mask = mo_mask.unsqueeze(-1)
        mo_mask = mo_mask.expand(-1, -1, -1, self.n_objectives)
        critic_mask = mo_mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # Lexico update
        first_order_weights = th.zeros((self.n_agents, self.n_objectives)).to(old_pi.device)
        for ag in range(self.n_agents):
            for obj in range(self.n_objectives - 1):
                w = self.beta[ag, obj] + self.mu[ag, obj] * sum(
                    [self.beta[ag, j] for j in range(obj + 1, self.n_objectives)])
                first_order_weights[ag, obj] = w
            first_order_weights[ag, -1] = th.tensor(self.beta[ag, self.n_objectives - 1],
                                                    dtype=first_order_weights.dtype)

        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()

            # Calculate policy grad with mask

            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            # Expand first_order_weights to align with the last dimensions of advantages
            # Perform the dot product along the last dimension
            first_order_weighted_advantages = th.einsum('abcd,cd->abc', advantages, first_order_weights)

            surr1 = ratios * first_order_weighted_advantages # actor loss
            surr2 = (
                    th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                    * first_order_weighted_advantages
            ) # actor clipped loss

            pg_loss = (-((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum()/ mask.sum()) # final actor loss

            # Keep track of the losses independently for each agent and objective
            for obj in range(self.n_objectives):
                _surr1 = ratios * advantages[:, :, :, obj]
                _surr2 = (
                    th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages[:, :, :, obj]
                )
                _surr = th.min(_surr1, _surr2)
                _pg_loss = -((_surr1 + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()
                for ag in range(self.n_agents):
                    self.recent_losses[ag][obj].append(_pg_loss.detach())

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
                self.args.target_update_interval_or_tau > 1
                and (self.critic_training_steps - self.last_target_update_step)
                / self.args.target_update_interval_or_tau
                >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Update Lagrange multipliers
        for i in range(self.n_objectives-1):
            for ag in range(self.n_agents):
                self.j[ag, i] = (-th.tensor(self.recent_losses[ag][i])).mean()
                self.mu[ag,i] += self.eta[ag, i] * (self.j[ag, i] - (-self.recent_losses[ag][i][-1]))
                self.mu[ag,i] = max(0, self.mu[ag,i])


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )
            self.logger.log_stat("entropy", entropy.mean().item(), t_env)
            # Log lexico parameters for each agent
            for ag in range(self.n_agents):
                for obj in range(self.n_objectives-1):
                    self.logger.log_stat(f"Agent_{ag}/lagr_multiplier_{obj}", self.mu[ag, obj], t_env)
                    self.logger.log_stat(f"Agent_{ag}/mean_recent_loss_{obj}", self.j[ag, obj], t_env)

            self.logger.log_stat(
                "advantage_mean",
                (advantages * mo_mask).sum().item() / mo_mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])

        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                            self.args.gamma ** (step) * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                            self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                            self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
                self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
