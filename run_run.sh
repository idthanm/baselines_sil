#!/bin/bash
mpirun -np 4 --oversubscribe python -m baselines.run --env CrossroadEnd2end-v0 --env_type user_defined --alg ppo2 --alg_submodule ppo2 --num_timesteps 5e5 --network mlp --save_interval 1 --training_task left --log_path '/home/yang/Documents/Work_Data/Project/Toyota_RL/Toyata_2019/202003/logs_test' --load_path '~' --render

python -m baselines.run --env CrossroadEnd2end-v0 --env_type user_defined --alg ppo2 --alg_submodule ppo2 --num_timesteps 1 --network mlp --load_path '/home/yang/Documents/Work_Data/Project/Toyota_RL/Toyata_2019/202003/logs_test/checkpoints/00001' --play

