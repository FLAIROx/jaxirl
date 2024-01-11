import os

import wandb
from jaxirl.irl.main import main

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

sweep_configuration = {
    "method": "grid",
    "name": "airl_sweep",
    "metric": {"goal": "maximize", "name": "avg_return"},
    "parameters": {
        "log_gen_every": {"value": 2},
        "reward_net_hsize": {"value": [128, 128]},
        "reward_net_sigmoid": {"value": False},
        "wandb_log": {"value": True},
        "plot": {"value": True},
        "train_rng": {"value": "SAME"},
        "save_to_file": {"value": False},
        "env": {"values": ["halfcheetah"]},
        "reward_type": {"value": "REWARD_STATE"},
        "reward_normalize": {"value": True},
        "real_reward_type": {"value": "IRL_STATE"},
        "num_eval_steps": {"value": 1000},
        "loss": {"value": "IRL"},
        "discr_batch_size": {"value": 4096},
        "seed": {"values": [1]},
        "seeds": {"value": 5},
        "discr_l1_loss": {"value": 0.0},
        "dual": {"value": False},
        "run_test": {"value": False},
        "discr_loss": {"value": "bce"},
        "irl_lrate_init": {"values": [1e-4]},
        "num_eval_envs": {"value": 100},
        "inner_lr_linear": {"values": [True]},
        "inner_lr": {"values": [4e-4]},
        "inner_steps": {"values": [15]},
        "discr_schedule_type": {"values": ["linear"]},
        "discr_trans_decay": {"values": [200]},
        "discr_final_lr": {"values": [1e-5]},
        "discr_updates_every": {"value": 1},
        "discr_updates": {"values": [4]},
        "num_updates_inner_loop": {"value": 1},
        "buffer_size": {"value": 100000},
    },
}


def pagent():
    main()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Faster_ES_Shaping")
    print("Sweep ID: ", sweep_id)
    wandb.agent(sweep_id, function=pagent, count=400)
