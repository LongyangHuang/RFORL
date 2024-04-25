# This is used to run SCORE on D4RL-MuJoCo tasks

import os
import argparse
import numpy as np
import torch
import gym, d4rl
import pandas as pd
import RFORL
import utils as utils


from utils import *



# Evaluation
def eval_policy(policy, eval_env, mean, std, max_path_length=1000, eval_episodes=5):
    total_reward=0
    for _ in range(eval_episodes):
        state  = eval_env.reset()
        path_length = 0
        

        while path_length < max_path_length:
            s = (np.array(state).reshape(1,-1) - mean)/std
            a = policy.select_action(s)
            next_o, r, d, _ = eval_env.step(a)
            total_reward+=r
            path_length += 1
            if d:
                break
            state = next_o
             
        
    
    total_reward/=eval_episodes

    d4rl_score = eval_env.get_normalized_score(total_reward) * 100

    print("---------------------------------------")
    print(f"Score: {d4rl_score}")
    print("---------------------------------------")

    return d4rl_score


# Training
def train_RFORL(device, args):
    env = gym.make(args.env_name)
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": device,
        "discount": args.discount,
        "tau": args.tau,
        "lr": args.lr,
        # TD3
        "expl_noise": args.expl_noise,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "tao": args.tao,
        # RFORL
        "num_ensemble": args.num_ensemble,
        "spectral_norm": args.spectral_norm,
    }

    

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, args.num_ensemble, ber_mean=args.ber_mean)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    replay_buffer.set_mask()
    mean,std = replay_buffer.normalize_states() 
    replay_buffer.normalize_rewards()
    print('Loaded buffer')
 
	
    
    # Initialize policy
    policy = RFORL.RFORL(**kwargs)
       
    
    evaluations = []
    for epoch in range(int(args.epochs)):

        print(epoch)   
        policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        paths = eval_policy(policy, env, mean, std, max_path_length=args.max_path_length)
        evaluations.append(paths)

        log = pd.DataFrame(evaluations)
        log.to_csv(f"./results/{file_name}.csv")

    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="halfcheetah-expert-v2")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)                        # How often (time steps) we evaluate
    parser.add_argument("--epochs", default=2e2, type=int)                           # Maximum epoch
    parser.add_argument("--batch_size", default=180, type=int)                       # Mini batch size for networks
    parser.add_argument("--discount", default=0.99, type=float)                      # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                          # Target network update rate
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--max_path_length", default=1000, type=int)
    
    # TD3
    parser.add_argument("--expl_noise", default=1.0, type=float)                     # Std of Gaussian exploration noise
    parser.add_argument("--policy_noise", default=0.2, type=float)                   # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                     # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)                        # Frequency of delayed policy updates
                       
    # RFORL
    parser.add_argument("--tao", default=0.001, type=float)  
    parser.add_argument('--num_ensemble', default=10, type=int)            # Number of ensemble networks
    parser.add_argument('--ber_mean', default=1.0, type=float)                       # Mask ratio for bootstrapped sampling
    parser.add_argument("--spectral_norm", action='store_true', default=False)
    args = parser.parse_args()    

    file_name = f"RFORL_{args.env_name}_{args.seed}"
    print("---------------------------------------")
    print(f"Setting: Training SCORE, Env: {args.env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists(f"./results"):
        os.makedirs(f"./results")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_RFORL(device, args)
