from mb_agg import *
from agent_utils import eval_actions
from agent_utils import select_action, select_action2
from models.PPO_Actor1 import Job_Actor, Mch_Actor
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from epsGreedyForMch import PredictMch
import os
device = torch.device(configs.device)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR



class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.mask_mch = []
        self.first_task = []
        self.pre_task = []
        self.action = []
        self.mch = []
        self.dur = []
        self.mch_time = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.mask_mch[:]
        del self.first_task[:]
        del self.pre_task[:]
        del self.action[:]
        del self.mch[:]
        del self.dur[:]
        del self.mch_time[:]
def initWeights(net, scheme='orthogonal'):

   for e in net.parameters():
      if scheme == 'orthogonal':
         if len(e.size()) >= 2:
            nn.init.orthogonal_(e)

      elif scheme == 'normal':
         nn.init.normal(e, std=1e-2)
      elif scheme == 'xavier':
         nn.init.xavier_normal(e)
def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs
class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_job = Job_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=configs.neighbor_pooling_type,
                                    input_dim=configs.input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)
        self.policy_mch = Mch_Actor(n_j=configs.n_j,
                                    n_m=configs.n_m,
                                    num_layers=configs.num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=configs.neighbor_pooling_type,
                                    input_dim=configs.input_dim,
                                    hidden_dim=configs.hidden_dim,
                                    num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                                    device=device)

        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)

        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())

        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)

        self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)


        self.MSE = nn.MSELoss()

    def update(self,  memories, epoch):
        '''self.policy_job.train()
        self.policy_mch.train()'''
        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []

        for i in range(configs.batch_size):
            rewards = []

            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.r_mb[0][i]).tolist()),
                                           reversed(memories.done_mb[0][i].tolist())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0)
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)

            job_log_prob = []
            mch_log_prob = []
            val = []
            mch_a =None
            last_hh = None
            entropies = []
            job_entropy = []
            mch_entropies = []
            job_scheduler = LambdaLR(self.job_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            mch_scheduler = LambdaLR(self.mch_optimizer, lr_lambda=lambda f: 0.96 ** epoch)
            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            env_mask_mch = memories.mask_mch[0]
            env_dur = memories.dur[0]
            first_task = memories.first_task[0]
            pool=None
            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i]
                env_adj = memories.adj_mb[i]
                env_candidate = memories.candidate_mb[i]
                env_mask = memories.mask_mb[i]


                a_index = memories.a_mb[i]
                env_mch_time = memories.mch_time[i]

                old_action = memories.action[i]
                old_mch = memories.mch[i]

                a_entropy, v, log_a, action_node, _, mask_mch_action, hx = self.policy_job(x=env_fea,
                                                                                           graph_pool=g_pool_step,
                                                                                           padded_nei=None,
                                                                                           adj=env_adj,
                                                                                           candidate=env_candidate
                                                                                           , mask=env_mask
                                                                                           , mask_mch=env_mask_mch
                                                                                           , dur=env_dur
                                                                                           , a_index=a_index
                                                                                           , old_action=old_action
                                                                                           ,mch_pool=pool
                                                                                           , old_policy=False
                                                                                           )
                pi_mch,pool = self.policy_mch(action_node, hx, mask_mch_action, env_mch_time,mch_a,last_hh,policy=True)
                val.append(v)
                dist = Categorical(pi_mch)
                log_mch = dist.log_prob(old_mch)
                mch_entropy = dist.entropy()

                job_entropy.append(a_entropy)
                mch_entropies.append(mch_entropy)
                # entropies.append((mch_entropy+a_entropy))

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)

            job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
                                                                                                     0).permute(1, 0)
            mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
                                                                                                     0).permute(1, 0)
            val = torch.stack(val, 0).squeeze(-1).permute(1, 0)
            job_entropy = torch.stack(job_entropy, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            job_v_loss_sum = 0
            mch_loss_sum = 0
            mch_v_loss_sum = 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                advantages = rewards_all_env[j] - val[j].detach()
                advantages = adv_normalize(advantages)
                job_surr1 = job_ratios * advantages
                job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                job_v_loss = self.MSE(val[j], rewards_all_env[j])
                job_loss = -1*torch.min(job_surr1, job_surr2) + 0.5*job_v_loss - 0.01 * job_entropy[j]
                job_loss_sum += job_loss
                job_v_loss_sum += job_v_loss

                mch_surr1 = mch_ratios * advantages
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                #mch_v_loss = self.MSE(val[j], rewards_all_env[j])
                mch_loss = -1*torch.min(mch_surr1, mch_surr2) - 0.01 * mch_entropies[j] + 0.5*job_v_loss
                mch_loss_sum += mch_loss
                mch_v_loss_sum += job_v_loss

            # take gradient step
            # loss_sum = torch.stack(loss_sum,0)
            # v_loss_sum = torch.stack(v_loss_sum,0)
            self.job_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)
            self.job_optimizer.step()
            # scheduler.step()
            # Copy new weights into old policy:
            self.policy_old_job.load_state_dict(self.policy_job.state_dict())
            if configs.decayflag:
                self.job_scheduler.step()

            self.mch_optimizer.zero_grad()
            mch_loss_sum.mean().backward(retain_graph=True)
            self.mch_optimizer.step()
            # scheduler.step()
            # Copy new weights into old policy:
            self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
            if configs.decayflag:
                self.mch_scheduler.step()

            return job_loss_sum.mean().item(), mch_loss_sum.mean().item()