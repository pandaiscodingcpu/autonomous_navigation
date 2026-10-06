import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Kalman:
    def __init__(self,dT):
        # static process noise Q
        self.dT = 1 / dT
        self.Q = np.array([[0.01,0],
                           [0,0.01]])
        # static measurement noise H
        self.H = np.array([[1,0]])
        # state to transition matrix
        self.A = np.array([[1,self.dT],
                           [0,1]])
        # noise level
        self.R = np.array([[10000]]) # more value. less trust on sensor readings
        self.Xk = np.array([[0],
                            [0]]) # initialization
        # covariance matrix
        self.P = np.eye(2,2) * 500
        self.I = np.eye(2,2)
    def init_state(self, x0):
        self.Xk = np.array([[x0], [0.0]])
    def predict(self):
        self.Xk = self.A @ self.Xk
        self.P = (self.A @ self.P @ self.A.T) + self.Q

    def kalman(self,z):
        # how much predictions differs from actual readings
        y = z - (self.H @ self.Xk)
        # innovation covariance (update the covariance)
        S = (self.H @ self.P @ self.H.T) + self.R
        # kalman_results gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # update the state
        self.Xk = self.Xk + (K @ y)
        # update P
        self.P = (self.I - K @ self.H) @ self.P
        #print(self.Xk)
        return self.Xk  #[position,velocity]

class Demo:
    def __init__(self,rate):
        self.rate = rate
    def multiplot(self,filename,column):
        kf = Kalman(self.rate)  # rate
        filtered_alt = []
        filtered_vel = []
        df = pd.read_csv(f'../data/{filename}')
        # purposely adding noise
        df[column] = df[column] + np.random.normal(0, 10, len(df))
        raw_alt_values = list(df[column])
        for i in range(len(df)):
            z = np.array([[raw_alt_values[i]]])  # current measurement
            kf.predict()
            state = kf.kalman(z)
            filtered_alt.append(state[0, 0])
            filtered_vel.append(state[1, 0])
        # Calculate noisy velocity (dx/dt)
        dT = self.rate
        t = [i / dT for i in range(len(df))]
        # np.gradient calculates the derivative while maintaining the exact array length
        noisy_vel = np.gradient(raw_alt_values, t)
        # Set up the figure and 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        # Graph 1: Noisy Altitude and Filtered Altitude vs. i
        axes[0].plot(t,raw_alt_values, label='Noisy Altitude', color='red', alpha=0.5)
        axes[0].plot(t,filtered_alt, label='Filtered Altitude', color='blue', linewidth=2)
        axes[0].set_title(f'Graph 1: Noisy vs. Filtered Altitude at {dT} Hz')
        axes[0].set_ylabel('Altitude')
        axes[0].legend()
        # Graph 2: Noisy Velocity and Filtered Velocity vs. i
        axes[1].plot(t,noisy_vel, label='Noisy Velocity (dx/dt)', color='orange', alpha=0.5)
        axes[1].plot(t,filtered_vel, label='Filtered Velocity', color='green', linewidth=2)
        axes[1].set_title(f'Graph 2: Noisy vs. Filtered Velocity at {dT} Hz')
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        # Graph 3: Filtered Altitude and Filtered Velocity vs. i
        # Using twinx() because altitude and velocity will have very different numerical scales
        ax3_alt = axes[2]
        ax3_vel = axes[2].twinx()
        line1, = ax3_alt.plot(t,filtered_alt, label='Filtered Altitude', color='blue', linewidth=2)
        line2, = ax3_vel.plot(t,filtered_vel, label='Filtered Velocity', color='green', linewidth=2)
        ax3_alt.set_title(f'Graph 3: Filtered Altitude and Velocity at {dT} Hz')
        ax3_alt.set_ylabel('Altitude', color='blue')
        ax3_vel.set_ylabel('Velocity', color='green')
        # Combine legends for the dual-axis graph
        ax3_alt.legend(handles=[line1, line2], loc='upper right')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'kalman_{self.rate}_smoothed_out.png', dpi=150)
        plt.show()

        plt.figure(figsize=(9, 5))
        plt.scatter(t, filtered_vel, label='Filtered Velocity', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.legend()
        plt.savefig(f'kalman_vel_{self.rate}_smoothed_out.png', dpi=150)
        plt.show()

    # return the available datasets (Helper function) 
    def list_datasets(self,data_dir: Path = Path(__file__).resolve().parent.parent / "data") -> list[str]:
        return sorted(f.name for f in data_dir.iterdir() if f.is_file())


if __name__ == '__main__':
    d = Demo(rate=100) # eg. 100Hz
    filename = ""
    column = ""
    print("Select the dataset")
    datasets = list(d.list_datasets())
    for f in datasets:
        print(f)
    choice = int(input())
    if choice == 1:
        filename = datasets[0]
        df = pd.read_csv(f"../data/{filename}")
        cols = df.columns
        print("Enter the column name:")
        print(cols)
        column = str(input())
    else:
        filename = datasets[1]
        df = pd.read_csv(f"../data/{filename}")
        cols = df.columns
        print("Enter the column name:")
        print(cols)
        column = str(input())
    d.multiplot(filename,column)