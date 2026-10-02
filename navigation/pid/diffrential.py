# implementation of differential (derivative) controller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from kalman.kalman import Kalman


class Differential:
    def __init__(self, target_lat, target_lon, target_vel, wind_lat=0.2, wind_lon=0.2):
        # derivative constant
        self.Kd = 0.5
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.target_vel = target_vel
        # constant wind drift acting on the plant. A D-only controller has nothing to react
        # to unless the vehicle is moving, so a drift is needed to see it work.
        self.wind_lat = wind_lat
        self.wind_lon = wind_lon

    def differential_controller(self):
        df = pd.read_csv(ROOT / "data" / "stress_csv.csv")
        lat_true = df[' GNSS_LATITUDE'].to_numpy()
        lon_true = df[' GNSS_LONGITUDE'].to_numpy()
        RATE = 100  # in Hz
        dt = 1 / RATE
        kf_lat, kf_lon = Kalman(RATE), Kalman(RATE)
        kf_lat.R = np.array([[25.0]])
        kf_lon.R = np.array([[25.0]])
        kf_lat.init_state(lat_true[0])
        kf_lon.init_state(lon_true[0])

        # plant starts at the first GNSS point
        pos_lat, pos_lon = lat_true[0], lon_true[0]

        meas_lat_hist, meas_lon_hist = [], []
        est_lat_hist, est_lon_hist = [], []
        true_lat_hist, true_lon_hist = [], []
        eX, eY = [], []
        dX, dY = [], []   # derivative of error
        uX, uY = [], []   # controller output

        for i in range(len(df)):
            # sensor: noisy measurement of the true position
            z_lat = pos_lat + np.random.normal(0, 5)
            z_lon = pos_lon + np.random.normal(0, 5)

            # Kalman filter: returns [[position], [velocity]]
            kf_lat.predict()
            state_lat = kf_lat.kalman(np.array([[z_lat]]))
            kf_lon.predict()
            state_lon = kf_lon.kalman(np.array([[z_lon]]))
            est_lat, vel_lat = state_lat[0, 0], state_lat[1, 0]
            est_lon, vel_lon = state_lon[0, 0], state_lon[1, 0]

            # error on the filtered estimate
            error_X = self.target_lat - est_lat
            error_Y = self.target_lon - est_lon

            # D controller: de/dt. The target is constant, so de/dt = -(estimated velocity).
            # Using the Kalman velocity avoids differentiating noisy data directly.
            d_error_X = -vel_lat
            d_error_Y = -vel_lon
            u_lat = self.Kd * d_error_X
            u_lon = self.Kd * d_error_Y

            # plant: velocity command + wind drift integrates into position
            pos_lat += (u_lat + self.wind_lat) * dt
            pos_lon += (u_lon + self.wind_lon) * dt

            meas_lat_hist.append(z_lat)
            meas_lon_hist.append(z_lon)
            est_lat_hist.append(est_lat)
            est_lon_hist.append(est_lon)
            true_lat_hist.append(pos_lat)
            true_lon_hist.append(pos_lon)
            eX.append(error_X); eY.append(error_Y)
            dX.append(d_error_X); dY.append(d_error_Y)
            uX.append(u_lat); uY.append(u_lon)

        self.plot(RATE, meas_lat_hist, est_lat_hist, true_lat_hist,
                  meas_lon_hist, est_lon_hist, true_lon_hist, eX, eY, dX, dY, uX, uY)

    def plot(self, rate, meas_lat, est_lat, true_lat, meas_lon, est_lon, true_lon,
             eX, eY, dX, dY, uX, uY):
        t = np.arange(len(est_lat)) / rate
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        # Latitude
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

        # Derivative of error (left axis) and controller output u = Kd * de/dt (right axis)
        ax[1, 0].plot(t, dX, color='tab:green', label='de/dt Lat')
        ax[1, 0].plot(t, dY, color='tab:pink', label='de/dt Lon')
        ax[1, 0].axhline(0, color='black', linewidth=0.8)
        ax[1, 0].set_xlabel('Time (s)'); ax[1, 0].set_ylabel('de/dt')
        ax[1, 0].set_title(f'Error rate and derivative output (Kd = {self.Kd})')
        ax_u = ax[1, 0].twinx()
        ax_u.plot(t, uX, color='tab:orange', linestyle='--', label='u Lat')
        ax_u.plot(t, uY, color='tab:purple', linestyle='--', label='u Lon')
        ax_u.set_ylabel('Controller output u')
        h1, l1 = ax[1, 0].get_legend_handles_labels()
        h2, l2 = ax_u.get_legend_handles_labels()
        ax[1, 0].legend(h1 + h2, l1 + l2, loc='upper right')

        # Trajectory
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
        # plt.savefig('differential_closed_loop.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    p = Differential(50.0000, -45.0000, 3.000)
    p.differential_controller()