import gym
import click
from gym.envs.registration import register
import os
import torch
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from actor import DiagGaussianActor

DESC = """
TUTORIAL: Arm+Gripper tele-op using input devices (keyboard / spacenav) \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

from robohive.utils.quat_math import euler2quat, mulQuat
from robohive.logger.roboset_logger import RoboSet_Trace
from robohive.logger.grouped_datasets import Trace as RoboHive_Trace
import numpy as np
import click
import gym


@click.command(help=DESC)
@click.option(
    "-e", "--env_name", type=str, help="environment to load", default="door-v1"
)
@click.option(
    "-ea",
    "--env_args",
    type=str,
    default=None,
    help=("env args. E.g. --env_args \"{'is_hardware':True}\""),
)
@click.option(
    "-rn",
    "--reset_noise",
    type=float,
    default=0.0,
    help=("Amplitude of noise during reset"),
)
@click.option(
    "-an",
    "--action_noise",
    type=float,
    default=0.0,
    help=("Amplitude of action noise during rollout"),
)
@click.option(
    "-o", "--output", type=str, default="teleOp_trace.h5", help=("Output name")
)
@click.option("-h", "--horizon", type=int, help="Rollout horizon", default=100)
@click.option(
    "-n",
    "--num_rollouts",
    type=int,
    help="number of repeats for the rollouts",
    default=1,
)
@click.option(
    "-f",
    "--output_format",
    type=click.Choice(["RoboHive", "RoboSet"]),
    help="Data format",
    default="RoboHive",
)
@click.option(
    "-c",
    "--camera",
    multiple=True,
    type=str,
    default=[],
    help=("list of camera topics for rendering"),
)
@click.option(
    "-r",
    "--render",
    type=click.Choice(["onscreen", "offscreen", "onscreen+offscreen", "none"]),
    help="visualize onscreen or offscreen",
    default="onscreen",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="seed for generating environment instances",
    default=123,
)
@click.option(
    "-m",
    "--model_path",
    type=str,
    help="path to model to run",
    default=None,
)
def main(
    env_name,
    env_args,
    reset_noise,
    action_noise,
    output,
    horizon,
    num_rollouts,
    output_format,
    camera,
    render,
    seed,
    model_path
):
    # seed and load environments
    np.random.seed(seed)
    env = (
        gym.make(env_name)
        if env_args == None
        else gym.make(env_name, **(eval(env_args)))
    )
    env.seed(seed)
    env.env.mujoco_render_frames = True if "onscreen" in render else False

    # prep the logger
    if output_format == "RoboHive":
        trace = RoboHive_Trace("TeleOp Trajectories")
    elif output_format == "RoboSet":
        trace = RoboSet_Trace("TeleOp Trajectories")

    model = None
    if model_path != None:
        model = DiagGaussianActor(38, 28, 1024, 2, [-5, 2])
        model.load_state_dict(torch.load(model_path))
    
    # Collect rollouts
    total_rwds = 0
    for i_rollout in range(num_rollouts):
        # start a new rollout
        print("rollout {} start".format(i_rollout))
        group_key = "Trial" + str(i_rollout)
        trace.create_group(group_key)
        reset_noise = reset_noise * np.random.uniform(
            low=-1, high=1, size=env.init_qpos.shape
        )
        env.reset(reset_qpos=env.init_qpos + reset_noise, blocking=True)

        # recover init state
        obs, rwd, done, env_info = env.forward()
        act = np.zeros(env.action_space.shape)
        rwds = 0

        # start rolling out
        for i_step in range(horizon + 1):
            # nan actions for last log entry
            policy_act = np.random.uniform(env.action_space.shape)
            if model != None:
                policy_act = model(torch.from_numpy(obs.astype(np.float32))).mean.detach().numpy()

            # log values at time=t ----------------------------------
            datum_dict = dict(
                time=env.time,
                observations=obs,
                actions=policy_act,
                rewards=rwd,
                env_infos=env_info,
                done=done,
            )
            trace.append_datums(group_key=group_key, dataset_key_val=datum_dict)
            # print(f't={env.time:2.2}, a={act}, o={obs[:3]}')

            # step env using action from t=>t+1 ----------------------
            if i_step < horizon:  # incase last actions (nans) can cause issues in step
                obs, rwd, done, env_info = env.step(policy_act)
                rwds += rwd
        print("rollout {} end".format(i_rollout))
        print("Reward:", rwds)
        total_rwds += rwds
    print("Average Reward:", total_rwds / num_rollouts)
    # save and close
    env.close()
    trace.save(output, verify_length=True)

    # render video outputs
    if len(camera) > 0:
        if camera[0] != "default":
            trace.render(
                output_dir=".",
                output_format="mp4",
                groups=":",
                datasets=camera,
                input_fps=1 / env.dt,
            )
        elif output_format == "RoboHive":
            trace.render(
                output_dir=".",
                output_format="mp4",
                groups=":",
                datasets=[
                    "env_infos/obs_dict/rgb:left_cam:240x424:2d",
                    "env_infos/obs_dict/rgb:right_cam:240x424:2d",
                    "env_infos/obs_dict/rgb:top_cam:240x424:2d",
                    "env_infos/obs_dict/rgb:Franka_wrist_cam:240x424:2d",
                ],
                input_fps=1 / env.dt,
            )
        elif output_format == "RoboSet":
            trace.render(
                output_dir=".",
                output_format="mp4",
                groups=":",
                datasets=[
                    "data/rgb_left",
                    "data/rgb_right",
                    "data/rgb_top",
                    "data/rgb_wrist",
                ],
                input_fps=1 / env.dt,
            )


if __name__ == "__main__":
    main()
