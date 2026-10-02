# implementation of proportional controller
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from kalman.kalman import Kalman
class Proportional:
    def __init__(self,target_lat,target_lon,target_vel):
        # proportional constant
        self.Kp = 0.5
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_vel = target_vel

    def proportional_controller(self):
        df = pd.read_csv(ROOT / "data" / "stress_csv.csv")
        # true sensor readings (or in this case the csv ones)
        lat_true = df[' GNSS_LATITUDE'].to_numpy()
        lon_true = df[' GNSS_LONGITUDE'].to_numpy()
        RATE = 100 # in Hz
        dt = 1 / RATE
        kf_lat, kf_lon = Kalman(RATE), Kalman(RATE)
        # tuning the R for Lat and Lon
        kf_lat.R = np.array([[25.0]])
        kf_lon.R = np.array([[25.0]])
        # initially start with first line of the csv
        kf_lat.init_state(lat_true[0])
        kf_lon.init_state(lon_true[0])

        # starts at first GNSS point
        pos_lat, pos_lon = lat_true[0], lon_true[0] # here, we consider the current location
        est_lat_hist, est_lon_hist, meas_lat_hist, meas_lon_hist = [], [], [], []
        eX, eY = [], [] # to store the error
        true_lat_hist,true_lon_hist = [], []

        for i in range(len(df)):
            # sensor: noisy measurement of the true position
            # adding the noise here
            z_lat = pos_lat + np.random.normal(0, 5)
            z_lon = pos_lon + np.random.normal(0, 5)

            # Kalman filter
            kf_lat.predict()
            est_lat = kf_lat.kalman(np.array([[z_lat]]))[0, 0]
            kf_lon.predict()
            est_lon = kf_lon.kalman(np.array([[z_lon]]))[0, 0]

            # P controller acts on the FILTERED estimate
            error_X = self.target_lat - est_lat
            error_Y = self.target_lon - est_lon
            u_lat = self.Kp * error_X
            u_lon = self.Kp * error_Y

            # plant: velocity command integrates into position (+ wind disturbance if we want)
            pos_lat += u_lat * dt
            pos_lon += u_lon * dt

            meas_lat_hist.append(z_lat)
            meas_lon_hist.append(z_lon)
            est_lat_hist.append(est_lat)
            est_lon_hist.append(est_lon)
            true_lat_hist.append(pos_lat)
            true_lon_hist.append(pos_lon)
            eX.append(error_X); eY.append(error_Y)
        self.plot(RATE, meas_lat_hist, est_lat_hist, true_lat_hist,meas_lon_hist, est_lon_hist, true_lon_hist, eX, eY)

    def plot(self, rate, meas_lat, est_lat, true_lat, meas_lon, est_lon, true_lon, eX, eY):
        t = np.arange(len(est_lat)) / rate
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        # Latitude: noisy sensor vs Kalman estimate vs true position
        ax[0, 0].plot(t, meas_lat, color='red', alpha=0.3, label='Measured (noisy)')
        ax[0, 0].plot(t, est_lat, color='blue', linewidth=2, label='Kalman estimate')
        ax[0, 0].plot(t, true_lat, color='black', linestyle='--', label='True position')
        ax[0, 0].axhline(self.target_lat, color='green', linestyle=':', label='Target')
        ax[0, 0].set_title('Latitude'); ax[0, 0].set_xlabel('Time (s)'); ax[0, 0].legend()

        # Longitude
        ax[0, 1].plot(t, meas_lon, color='red', alpha=0.3, label='Measured (noisy)')
        ax[0, 1].plot(t, est_lon, color='blue', linewidth=2, label='Kalman estimate')
        ax[0, 1].plot(t, true_lon, color='black', linestyle='--', label='True position')
        ax[0, 1].axhline(self.target_lon, color='green', linestyle=':', label='Target')
        ax[0, 1].set_title('Longitude'); ax[0, 1].set_xlabel('Time (s)'); ax[0, 1].legend()

        # Error convergence (should decay toward 0)
        ax[1, 0].plot(t, eX, color='tab:green', label='Error Lat')
        ax[1, 0].plot(t, eY, color='tab:pink', label='Error Lon')
        ax[1, 0].axhline(0, color='black', linewidth=0.8)
        ax[1, 0].set_title(f'Controller error (Kp = {self.Kp})')
        ax[1, 0].set_xlabel('Time (s)'); ax[1, 0].set_ylabel('Error'); ax[1, 0].legend()

        # Trajectory in the lat/lon plane
        ax[1, 1].plot(true_lat, true_lon, color='black', label='True path')
        ax[1, 1].plot(est_lat, est_lon, color='blue', alpha=0.6, label='Estimated path')
        ax[1, 1].scatter(true_lat[0], true_lon[0], color='orange', s=80, zorder=5, label='Start')
        ax[1, 1].scatter(self.target_lat, self.target_lon, color='green', marker='*',
                        s=200, zorder=5, label='Target')
        ax[1, 1].set_title('Trajectory'); ax[1, 1].set_xlabel('Latitude')
        ax[1, 1].set_ylabel('Longitude'); ax[1, 1].legend()

        for a in ax.flat:
            a.grid(True, alpha=0.3)
        plt.tight_layout()
        #plt.savefig('proportional_closed_loop.png', dpi=150)
        plt.show()

if __name__ == '__main__':
    p = Proportional(50.0000,-45.0000,3.000)
    p.proportional_controller()
