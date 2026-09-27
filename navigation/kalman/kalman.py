import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Kalman:
    def __init__(self,dT):
        # static process noise Q
        self.dT = 1 / dT
        self.Q = np.array([[0.1,0],
                           [0,0.1]])
        # static measurement noise H
        self.H = np.array([[1,0]])
        # state to transition matrix
        self.A = np.array([[1,self.dT],
                           [0,1]])
        # noise level
        self.R = np.array([[100]]) # more value. less trust on sensor readings
        self.Xk = np.array([[0],
                            [0]]) # initialization
        # covariance matrix
        self.P = np.eye(2,2) * 500
        self.I = np.eye(2,2)

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
        return self.Xk


class Demo:
    def __init__(self,rate):
        self.rate = rate
    def multiplot(self):
        kf = Kalman(self.rate)  # rate
        filtered_alt = []
        filtered_vel = []

        df = pd.read_csv('../data/stress_csv.csv')
        # purposely adding noise
        df[' ALTITUDE'] = df[' ALTITUDE'] + np.random.normal(0, 10, len(df))
        raw_alt_values = list(df[' ALTITUDE'])
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

if __name__ == '__main__':
    d = Demo(rate=100) # eg. 10Hz
    d.multiplot()