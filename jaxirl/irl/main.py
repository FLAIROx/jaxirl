import os
import pickle
from jaxirl.utils.parser import get_parser
from jaxirl.utils.plot import plot
from matplotlib import pyplot as plt

import wandb
from jaxirl.utils.utils import get_irl_config, get_plot_filename
from jaxirl.irl.irl import IRL
from jaxirl.irl.bc import BC
from jaxirl.irl.rl import RL
from jaxirl.utils.env_utils import get_env, get_test_params

from jaxirl.utils.utils import LossType, RewardWrapper, generate_config

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax

print("Visible devices", jax.devices())

import argparse
from jaxirl.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)
from jaxirl.training.ppo_v2_irl import make_train, eval


def main(es_config=None):
    run = None
    if es_config is None:
        run = wandb.init(project="IRL")
        es_config = wandb.config
        print("wandb run initialized: ", run.name)
    env, env_params, original_training_config = get_env(es_config["env"])
    es_config, es_training_config = get_irl_config(es_config, original_training_config)
    if es_config["wandb_log"]:
        total_config = {**original_training_config, **es_config}
        if run is None:
            run = wandb.init(project="IRL", config=total_config)
            print("wandb run initialized: ", run.name)
        else:
            run.config.update(total_config)
        # create dir for saving rewards
        rewards_dir = f"{os.getcwd()}/rewards"
        save_dir = None
        if es_config["save_to_file"]:
            if not os.path.exists(rewards_dir):
                os.mkdir(rewards_dir)
            save_dir = (
                f"{os.getcwd()}/rewards/{str(run.name).replace(' ', '_')}_recover_NN"
            )
            print("save dir:", save_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        es_config["save_dir"] = save_dir
    # train base agent
    if original_training_config["DISCRETE"]:
        make_train_fn, eval_fn = make_train, eval
    else:
        make_train_fn, eval_fn = make_train_cont, eval_cont

    # TRAIN EXPERT AGENT
    true_env = RewardWrapper(
        env=env,
        env_params=env_params,
        reward_network=None,
        rew_network_params=None,
        training_config=original_training_config,
    )
    trained_expert_path = f"{os.getcwd()}/experts/{es_config['env']}.pkl"
    trained_test_expert_path = f"{os.getcwd()}/experts_test/{es_config['env']}.pkl"
    rng = jax.random.PRNGKey(es_config["seed"])
    if not os.path.exists(trained_expert_path):
        print("Trained expert not found, retraining")
        train_fn = make_train_fn(
            config=original_training_config,
            env=true_env,
            env_params=env_params,
            runner_state_start=None,
            log_timestep_returns=True,
        )
        expert_train_out = jax.jit(train_fn)(rng)
        expert_obsv, expert_actions = eval_fn(
            num_envs=es_config["num_eval_envs"],
            num_steps=es_config["num_eval_steps"],
            env=true_env,
            env_params=env_params,
            agent_params=expert_train_out["runner_state"][0].params,
            rng=jax.random.PRNGKey(0),
            network=true_env.agent_net,
        )
        with open(trained_expert_path, "wb") as f:
            test_original_returns = (
                expert_train_out["metrics"]["timestep_returned_episode_returns"]
                .mean(-1)
                .reshape(-1)
            )
            pickle.dump(
                {
                    "returns": test_original_returns,
                    "final_return": test_original_returns[-1],
                    "expert_obsv": expert_obsv,
                    "expert_actions": expert_actions,
                },
                f,
            )
    else:
        print("Trained expert FOUND")
    if not os.path.exists(trained_test_expert_path):
        print("Trained TEST expert not found, retraining")
        test_env_params = get_test_params(es_config)
        test_train_fn = make_train_fn(
            config=original_training_config,
            env=true_env,
            env_params=test_env_params,
            runner_state_start=None,
            log_timestep_returns=True,
        )
        test_train_out = jax.jit(test_train_fn)(rng)
        test_expert_obsv, test_expert_actions = eval_fn(
            num_envs=es_config["num_eval_envs"],
            num_steps=es_config["num_eval_steps"],
            env=true_env,
            env_params=test_env_params,
            agent_params=test_train_out["runner_state"][0].params,
            rng=jax.random.PRNGKey(0),
            network=true_env.agent_net,
        )
        with open(trained_test_expert_path, "wb") as f:
            original_returns = (
                test_train_out["metrics"]["timestep_returned_episode_returns"]
                .mean(-1)
                .reshape(-1)
            )
            pickle.dump(
                {
                    "returns": original_returns,
                    "final_return": original_returns[-1],
                    "expert_obsv": test_expert_obsv,
                    "expert_actions": test_expert_actions,
                },
                f,
            )
    else:
        print("Trained TEST expert FOUND")

    with open(trained_expert_path, "rb") as f:
        expert_train_out = pickle.load(f)
        original_returns = expert_train_out["returns"]
        last_return = expert_train_out["final_return"]
        expert_obsv = expert_train_out["expert_obsv"]
        expert_actions = expert_train_out["expert_actions"]
    with open(trained_test_expert_path, "rb") as f:
        test_expert_train_out = pickle.load(f)
        test_last_return = test_expert_train_out["final_return"]
        test_env_params = get_test_params(es_config)
        test_expert_obsv = test_expert_train_out["expert_obsv"]
        test_expert_actions = test_expert_train_out["expert_actions"]
    plt.plot(original_returns)
    if es_config["wandb_log"]:
        wandb.log(
            step=0,
            data={
                "base_last_return": last_return,
                "original_train_plt": plt,
            },
        )
    print("Expert last return", last_return)
    print("Expert test last return", test_last_return)

    if LossType[es_config["loss"]] == LossType.IRL:
        training_class = IRL(
            env=env,
            env_params=env_params,
            training_config=es_training_config,
            es_config=es_config,
            logging_run=run,
            expert_data=(expert_obsv, expert_actions),
            test_expert_data=(test_expert_obsv, test_expert_actions),
        )
        best_discr, irl_train_metrics = training_class.train(rng=rng)
        rew_net_params = best_discr.params
        reward_net = training_class.discr
        if es_config["plot"]:
            sz = (
                es_config["num_updates_inner_loop"]
                * es_config["inner_steps"]
                * es_training_config["NUM_ENVS"],
            )
            steps = es_config["irl_generations"]
            filename_plot = get_plot_filename(es_config)
            plot(
                es_config["env"],
                irl_train_metrics,
                last_return,
                steps,
                sz,
                filename_plot,
            )
        wandb.finish()
        return
    # standard RL
    elif LossType[es_config["loss"]] == LossType.NONE:
        training_class = RL(
            env=env,
            env_params=env_params,
            training_config=es_training_config,
            es_config=es_config,
            logging_run=run,
        )
        training_class.train(rng=rng)
        wandb.finish()
        return
    elif LossType[es_config["loss"]] == LossType.BC:
        training_class = BC(
            env=env,
            env_params=env_params,
            training_config=es_training_config,
            es_config=es_config,
            logging_run=run,
            expert_data=(expert_obsv, expert_actions),
        )
        training_class.train(rng=rng)
        wandb.finish()
        return
    else:
        raise NotImplementedError("Loss not supported")

    best_reward_env = RewardWrapper(
        env=env,
        env_params=env_params,
        reward_network=reward_net,
        rew_network_params=rew_net_params,
        include_action=training_class._include_action,
        training_config=original_training_config,
    )
    train_fn_best = make_train_fn(
        config=original_training_config,
        env=best_reward_env,
        env_params=env_params,
        runner_state_start=None,
        log_timestep_returns=True,
    )
    train_out_best = jax.jit(train_fn_best)(rng)
    best_returns = (
        train_out_best["metrics"]["timestep_returned_episode_returns"]
        .mean(-1)
        .reshape(-1)
    )
    if es_config["wandb_log"]:
        for t in range(0, best_returns.shape[0], 100):
            wandb.log(
                data={
                    "best_training": best_returns[t],
                    "original_training": original_returns[t],
                    "improvement": best_returns[t] - original_returns[t],
                }
            )
    wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    for x in args.seed:
        es_config = generate_config(args, int(x))
        main(es_config)
