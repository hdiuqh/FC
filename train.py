import numpy as np
import matplotlib.pyplot as plt

from RL-buffer import ReplayBuffer
from RL import TD3
from mpc import SparseMPC
from feature import TerrainFeatureExtractor

class VehicleEnv:

    def __init__(self):

        self.dt = 0.05

        self.vx = 8.0

        self.ey = 0.0

        self.epsi = 0.0

    def reset(self):

        self.ey = np.random.uniform(-0.5, 0.5)

        self.epsi = np.random.uniform(-0.1, 0.1)

        return np.array([self.ey, self.epsi])

    def step(self, delta):

        # simple bicycle model

        self.ey += self.vx * np.sin(self.epsi) * self.dt

        self.epsi += (self.vx / 2.7) * delta * self.dt

        state = np.array([self.ey, self.epsi])

        return state

# Reward Function

def compute_reward(state, action):

    ey, epsi = state

    reward = (

        -4.0 * ey**2

        -2.0 * epsi**2

        -0.3 * action**2

    )

    return reward

# Feature Wrapper

def get_terrain_feature(az_signal):

    extractor = TerrainFeatureExtractor(az_signal)

    feat, _, _, _ = extractor.extract()

    return np.array([

        feat["RMS"],

        feat["PSD_low"],

        feat["PSD_mid"],

        feat["PSD_high"],

        feat["Dominant_frequency"]

    ])

# Training

def train(mode="mpc_td3"):

    env = VehicleEnv()

    mpc = SparseMPC()

    state_dim = 11

    td3 = TD3(state_dim=state_dim)

    buffer = ReplayBuffer()

    episodes = 50

    episode_length = 200

    reward_history = []

    for ep in range(episodes):

        state = env.reset()

        ep_reward = 0

        for t in range(episode_length):

            # -------------------------------------------------
            # terrain signal (IMU vertical accel)
            # -------------------------------------------------

            az = np.sin(0.1 * t) + 0.1 * np.random.randn()

            terrain_feat = get_terrain_feature(

                np.ones(50) * az

            )

            # -------------------------------------------------
            # MPC nominal control
            # -------------------------------------------------

            x0 = state

            u_sparse, u_mpc = mpc.control(x0)

            delta_mpc = u_mpc[0]

            # -------------------------------------------------
            # TD3 state
            # -------------------------------------------------

            td3_state = np.concatenate([

                state,

                terrain_feat,

                [delta_mpc]

            ])

            # -------------------------------------------------
            # TD3 action
            # -------------------------------------------------

            if mode == "mpc_td3":

                delta_rl = td3.select_action(td3_state)

                action = delta_mpc + delta_rl

            else:

                action = delta_mpc

            # safety clip

            action = np.clip(action, -0.6, 0.6)

            # -------------------------------------------------
            # step environment
            # -------------------------------------------------

            next_state = env.step(action)

            reward = compute_reward(next_state, action)

            done = False

            # -------------------------------------------------
            # store replay
            # -------------------------------------------------

            if mode == "mpc_td3":

                buffer.push(

                    td3_state,

                    delta_rl,

                    reward,

                    td3_state,

                    done

                )

            state = next_state

            ep_reward += reward

            # -------------------------------------------------
            # training
            # -------------------------------------------------

            if mode == "mpc_td3" and len(buffer) > 1000:

                td3.train(buffer, batch_size=64)

        reward_history.append(ep_reward)

        print(

            f"Episode {ep}, Reward: {ep_reward:.2f}"

        )

    return reward_history

# Evaluation

def evaluate():

    print("Running evaluation...")

    mpc_reward = train(mode="mpc")

    hybrid_reward = train(mode="mpc_td3")

    plt.plot(mpc_reward, label="Pure MPC")

    plt.plot(hybrid_reward, label="MPC + TD3")

    plt.legend()

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("Performance Comparison")

    plt.show()


if __name__ == "__main__":

    evaluate()