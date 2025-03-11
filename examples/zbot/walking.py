"""Run reinforcement learning unit test for zbot.

Runs a simple walking policy on the zbot.
"""

import argparse
import asyncio
import logging
import math
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
import pickle
import torch
from models_and_configs.dummy_vec_env import DummyVecEnv



import colorlogging
import numpy as np
# import onnxruntime as ort
from rsl_rl.runners import OnPolicyRunner
from pykos import KOS
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=31, nn_id=0, kp=17.8, kd=0.0, max_torque=1.62, joint_name="left_hip_yaw"),
    Actuator(actuator_id=32, nn_id=1, kp=17.8, kd=0.0, max_torque=1.62, joint_name="left_hip_roll"),
    Actuator(actuator_id=33, nn_id=2, kp=17.8, kd=0.0, max_torque=1.62, joint_name="left_hip_pitch"),
    Actuator(actuator_id=34, nn_id=3, kp=17.8, kd=0.0, max_torque=1.62, joint_name="left_knee"),
    Actuator(actuator_id=35, nn_id=4, kp=17.8, kd=0.0, max_torque=1.62, joint_name="left_ankle"),
    Actuator(actuator_id=41, nn_id=5, kp=17.8, kd=0.0, max_torque=1.62, joint_name="right_hip_yaw"),
    Actuator(actuator_id=42, nn_id=6, kp=17.8, kd=0.0, max_torque=1.62, joint_name="right_hip_roll"),
    Actuator(actuator_id=43, nn_id=7, kp=17.8, kd=0.0, max_torque=1.62, joint_name="right_hip_pitch"),
    Actuator(actuator_id=44, nn_id=8, kp=17.8, kd=0.0, max_torque=1.62, joint_name="right_knee"),
    Actuator(actuator_id=45, nn_id=9, kp=17.8, kd=0.0, max_torque=1.62, joint_name="right_ankle"),
]

ACTUATOR_ID_TO_POLICY_IDX = {actuator.actuator_id: actuator.nn_id for actuator in ACTUATOR_LIST}
print(ACTUATOR_ID_TO_POLICY_IDX)

ACTUATOR_IDS = [actuator.actuator_id for actuator in ACTUATOR_LIST]

def get_remap_indices(current_list, target_list):
    """
    Generates indices to remap an array from the current actuator list order to the target dof_names order.
    
    Args:
        current_list (list[str]): List of joint names in the current order.
        target_list (list[str]): List of joint names in the target order.
    
    Returns:
        list[int]: Indices mapping current_list order to target_list order.
    """
    return [current_list.index(name) for name in target_list]


# Define the target order of degrees of freedom (DOFs)
DOF_NAMES_TRAINED_POLICY = [
    "right_hip_pitch",
    "left_hip_pitch",
    "right_hip_yaw",
    "left_hip_yaw",
    "right_hip_roll",
    "left_hip_roll",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
]

# Extract current joint names from ACTUATOR_LIST
CURRENT_DOF_NAMES = [actuator.joint_name for actuator in ACTUATOR_LIST]

# Compute remap indices
TO_TRAINED_POLICY_INDICES = get_remap_indices(CURRENT_DOF_NAMES, DOF_NAMES_TRAINED_POLICY)
TO_CURRENT_INDICES = get_remap_indices(DOF_NAMES_TRAINED_POLICY, CURRENT_DOF_NAMES)


def remap_to_trained_policy_order(array):
    """Reorders an array to match the target dof_names order."""
    return array[TO_TRAINED_POLICY_INDICES]


def remap_to_current_order(array):
    """Reorders an array back to the original ACTUATOR_LIST order."""
    return array[TO_CURRENT_INDICES]



def transform_by_quat(v, q):
    """Transforms a vector v by quaternion q (applies rotation)."""
    q_vec = np.array([q[1], q[2], q[3]])  # Extract vector part
    uv = np.cross(q_vec, v) * 2.0
    return v + q[0] * uv + np.cross(q_vec, uv)


def get_gravity_orientation(qw, qx, qy, qz):
    """
    Args:
        quaternion: np.ndarray[float, float, float, float]

    Returns:
        gravity_orientation: np.ndarray[float, float, float]
    """

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


async def simple_walking(
    model_path: str | Path,
    cfgs_path: str | Path,
    default_position: list[float],
    host: str,
    port: int,
    num_seconds: float | None = 10.0,
) -> None:
    """Runs a simple walking policy.

    Args:
        model_path: The path to the ONNX model.
        default_position: The default joint positions for the legs.
        host: The host to connect to.
        port: The port to connect to.
        num_seconds: The number of seconds to run the policy for.
    """
    assert len(default_position) == len(ACTUATOR_LIST)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfgs_path, "rb"))

    # Load the model

    # isn't really used, just to create the runner!
    env = DummyVecEnv(num_envs=1, obs_dim=36, action_dim=10)

    runner = OnPolicyRunner(env, train_cfg)
    print("Loading model...")
    runner.load(model_path)
    print("Model loaded!")
    policy = runner.get_inference_policy()

    # model_path = Path(model_path)
    # if not model_path.exists():
    #     raise FileNotFoundError(f"Model file not found: {model_path}")

    # session = ort.InferenceSession(model_path)

    # Get input and output details
    # output_details = [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()]

    # def policy(input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    #     results = session.run(None, input_data)
    #     return {output_details[i]["name"]: results[i] for i in range(len(output_details))}




    async with KOS(ip=host, port=port) as sim_kos:
        for actuator in ACTUATOR_LIST:
            await sim_kos.actuator.configure_actuator(
                actuator_id=actuator.actuator_id,
                kp=actuator.kp,
                kd=actuator.kd,
                max_torque=actuator.max_torque,
            )

        await sim_kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 0.4},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            joints=[
                {
                    "name": actuator.joint_name,
                    "pos": pos,
                }
                for actuator, pos in zip(ACTUATOR_LIST, default_position)
            ],
        )
        start_time = time.time()
        end_time = None if num_seconds is None else start_time + num_seconds

        default = np.array(default_position)
        target_q = np.zeros(10, dtype=np.double)
        prev_actions = np.zeros(10, dtype=np.double)
        # hist_obs = np.zeros(570, dtype=np.double)

    #     input_data = {
    #         "x_vel.1": np.zeros(1).astype(np.float32),
    #         "y_vel.1": np.zeros(1).astype(np.float32),
    #         "rot.1": np.zeros(1).astype(np.float32),
    #         "t.1": np.zeros(1).astype(np.float32),
    #         "dof_pos.1": np.zeros(10).astype(np.float32),
    #         "dof_vel.1": np.zeros(10).astype(np.float32),
    #         "prev_actions.1": np.zeros(10).astype(np.float32),
    #         "projected_gravity.1": np.zeros(3).astype(np.float32),
    #         "buffer.1": np.zeros(570).astype(np.float32),
    #     }

        x_vel_cmd = 0.15
        y_vel_cmd = 0.0
        yaw_vel_cmd = 0.0
        # yaw_vel_cmd = 0.2

        target_commands = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd])
        target_commands_scale = np.array([obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["lin_vel"], obs_cfg["obs_scales"]["ang_vel"]])


        frequency = 50

        start_time = time.time()
        next_time = start_time + 1 / frequency

        while end_time is None or time.time() < end_time:


            response, raw_quat = await asyncio.gather(
                sim_kos.actuator.get_actuators_state(ACTUATOR_IDS),
                sim_kos.imu.get_quaternion()
            )
        
            positions = np.array([math.radians(state.position) for state in response.states])
            # remap to trained policy order
            positions = remap_to_trained_policy_order(positions)

            velocities = np.array([math.radians(state.velocity) for state in response.states])
            # remap to trained policy order
            velocities = remap_to_trained_policy_order(velocities)
            r = R.from_quat([raw_quat.x, raw_quat.y, raw_quat.z, raw_quat.w])
            
            gvec = get_gravity_orientation(qw=raw_quat.w, qx=raw_quat.x, qy=raw_quat.y, qz=raw_quat.z)
            print(f"Gravity vector before: [{gvec[0]:.3f}, {gvec[1]:.3f}, {gvec[2]:.3f}]\n")
            gvec = np.array([-gvec[2], gvec[0],  gvec[1]])  # APPLY CORRECTION TO MATCH SIM2SIM
            print(f"Gravity vector: [{gvec[0]:.3f}, {gvec[1]:.3f}, {gvec[2]:.3f}]\n")

            obs = np.concatenate([
                gvec,  # 3
                target_commands * target_commands_scale,  # 3
                # (dof_pos - env_cfg["default_joint_angles"]) * obs_cfg["obs_scales"]["dof_pos"], 
                (positions - default) * obs_cfg["obs_scales"]["dof_pos"], # 10 # NOTE: diff from env_cfg values of 0
                velocities * obs_cfg["obs_scales"]["dof_vel"],  # 10
                prev_actions,  # 10
            ]).astype(np.float32)

            # Convert to a PyTorch tensor and ensure shape (1, 39)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            curr_actions = policy(obs_tensor).detach().numpy().squeeze() 

            # multiply by -1 
            # curr_actions = curr_actions * -1
            
            # doing clipping and changes from the zbot.env.step function
            # def seems to make it stabilize for a little bit longer
            curr_actions = np.clip(curr_actions, - env_cfg["clip_actions"], env_cfg["clip_actions"])
            curr_actions = curr_actions * env_cfg["action_scale"]

            # set prev_actions to current actions, while still in trained policy order
            prev_actions = curr_actions.copy()

            # remap to current actuator order
            curr_actions = remap_to_current_order(curr_actions)

            print(f"Actions: {curr_actions}")

            positions = curr_actions

            target_q = positions + default

            commands = []
            for actuator_id in ACTUATOR_IDS:
                policy_idx = ACTUATOR_ID_TO_POLICY_IDX[actuator_id]
                raw_value = target_q[policy_idx]
                command_deg = raw_value

                # I believe policy output is already in degrees, so this may not be needed.
                # command_deg = math.degrees(raw_value)

                commands.append({"actuator_id": actuator_id, "position": command_deg})
            print("commands: ", commands)

            await sim_kos.actuator.command_actuators(commands)
            await asyncio.sleep(max(0, next_time - time.time()))
            next_time += 1 / frequency


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--num-seconds", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, default="model.onnx")
    parser.add_argument("--config", type=str, default="config.pkl")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    # Start kos-sim
    logger.info("Starting simulator server...")
    sim_process = subprocess.Popen(["kos-sim", "zbot-v2-fixed"])
    time.sleep(2)

    # Defines the default joint positions for the legs.
    # default_position = [
    #     0.0,  # left hip yaw
    #     0.0,  # left hip roll
    #     -0.37,  # left hip pitch -0.53770,
    #     0.7960,  # left knee
    #     0.42,  # left ankle
    #     0.0,  # right hip yaw
    #     0.0,  # right hip roll
    #     0.37,  # right hip pitch 0.53770,
    #     -0.7960,  # right knee
    #     -0.42,  # right ankle
    # ]

    default_position = [
        0.0,  # left hip yaw
        0.0,  # left hip roll
        0.0, # left hip pitch -0.53770,
        0.0,  # left knee
        0.0,  # left ankle
        0.0,  # right hip yaw
        0.0,  # right hip roll
        0.0,  # right hip pitch 0.53770,
        0.0,  # right knee
        0.0
    ]

    await simple_walking(args.model, args.config, default_position, args.host, args.port, args.num_seconds)


if __name__ == "__main__":
    asyncio.run(main())