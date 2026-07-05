import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================
# Simulation Parameters


DT = 0.05                # sampling time
TIME = 60                # simulation duration
N = int(TIME / DT)

VX = 8.0                 # vehicle speed (m/s)

class TerrainGenerator:

    def __init__(self, terrain="low"):

        self.terrain = terrain

    def generate(self):

        t = np.arange(N) * DT

        if self.terrain == "low":

            z = (
                    0.03*np.sin(0.5*t)
                  + 0.01*np.sin(4*t)
                  + 0.005*np.random.randn(N)
            )

        elif self.terrain == "medium":

            z = (
                    0.08*np.sin(0.8*t)
                  + 0.03*np.sin(7*t)
                  + 0.015*np.random.randn(N)
            )

        elif self.terrain == "high":

            z = (
                    0.15*np.sin(1.2*t)
                  + 0.08*np.sin(10*t)
                  + 0.03*np.random.randn(N)
            )

        else:

            raise ValueError("Unknown terrain")

        return t, z



# Vehicle Response Generator


class VehicleSimulator:

    def __init__(self):

        pass

    def simulate(self, terrain_height):

        # vertical acceleration
        az = np.gradient(np.gradient(terrain_height, DT), DT)

        # GPS trajectory
        x = np.arange(N) * VX * DT

        # lateral error
        ey = 0.15*np.sin(0.12*x)

        ey += 0.02*np.random.randn(N)

        # heading error

        epsi = np.gradient(ey)

        # steering angle

        delta = 0.12*ey + 0.02*np.random.randn(N)

        # yaw rate

        r = np.gradient(delta)

        return {

            "x": x,

            "ey": ey,

            "epsi": epsi,

            "delta": delta,

            "yaw_rate": r,

            "az": az

        }


def save_dataset(name, data):

    df = pd.DataFrame(data)

    df.to_csv(name, index=False)

    print(f"Saved -> {name}")

# Visualization

def visualize(terrain, result):

    plt.figure(figsize=(12,8))

    plt.subplot(311)
    plt.plot(result["x"], terrain)
    plt.ylabel("Terrain Height (m)")
    plt.title("Terrain")

    plt.subplot(312)
    plt.plot(result["x"], result["az"])
    plt.ylabel("Vertical Acceleration")

    plt.subplot(313)
    plt.plot(result["x"], result["ey"])
    plt.ylabel("Lateral Error (m)")
    plt.xlabel("Distance (m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    terrains = [

        "low",

        "medium",

        "high"

    ]

    for terrain_name in terrains:

        print("--------------------------")
        print("Generating:", terrain_name)

        generator = TerrainGenerator(terrain_name)

        t, terrain = generator.generate()

        simulator = VehicleSimulator()

        result = simulator.simulate(terrain)

        save_dataset(

            terrain_name + "_terrain.csv",

            result

        )

        visualize(terrain, result)
