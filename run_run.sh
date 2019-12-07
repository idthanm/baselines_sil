#!/bin/bash
mpirun -np 4 python -m baselines.run --env CrossroadEnd2end-v0 --env_type user_defined --alg ppo2 --alg_submodule ppo2 --num_timesteps 1e6 --network cnn_plus_vector --log_path '/home/yang/Documents/Work_Data/Project/Toyota_RL/Toyata_2019/201912/logs'

python -m baselines.run --env CrossroadEnd2end-v0 --env_type user_defined --alg ppo2 --alg_submodule ppo2 --num_timesteps 1 --network cnn_plus_vector --load_path '/home/yang/00100' --play

