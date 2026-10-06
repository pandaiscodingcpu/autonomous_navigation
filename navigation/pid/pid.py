import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Assuming your kalman module is in the parent directory as before
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from kalman.kalman import Kalman


class PIDAxis:
    """PID controller for a single axis. One instance per axis (lat, lon)."""

    def __init__(self, Kp, Ki, Kd, u_max):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_max = u_max
        self.integral = 0.0

    def update(self, error, d_error, dt):
        # P term: proportional to the present error
        p = self.Kp * error

        # I term: accumulated past error, clamped (anti-windup)
        self.integral += error * dt
        if self.Ki > 0:
            limit = self.u_max / self.Ki
            self.integral = float(np.clip(self.integral, -limit, limit))
        i = self.Ki * self.integral

        # D term: rate of change of the error (predicts where the error is heading)
        d = self.Kd * d_error

        # total output, saturated to the actuator limit 
        u = float(np.clip(p + i + d, -self.u_max, self.u_max))
        return u, p, i, d


class PID:
    def __init__(self, current_lat, current_lon, current_alt, target_lat, target_lon, wind_lat=0.0, wind_lon=0.0):
        # PID gains
        self.Kp = 0.7
        self.Ki = 0.0
        self.Kd = 1.0
        # actuator saturation (maximum velocity command)
        self.u_max = 5.0
        
        # Navigation points
        self.current_lat = current_lat
        self.current_lon = current_lon
        self.current_alt = current_alt # Alt included for API completeness, but PID is acting 2D
        self.target_lat = target_lat
        self.target_lon = target_lon
        
        # constant wind drift acting on the plant, 0 = no wind.
        self.wind_lat = wind_lat
        self.wind_lon = wind_lon

    def pid_controller(self):
        RATE = 100  # in Hz
        dt = 1 / RATE
        
        total_steps = 3000 

        kf_lat, kf_lon = Kalman(RATE), Kalman(RATE)
        kf_lat.R = np.array([[1000.0]])
        kf_lon.R = np.array([[1000.0]])
        
        # Initialize Kalman filters with current position
        kf_lat.init_state(self.current_lat)
        kf_lon.init_state(self.current_lon)

        ctrl_lat = PIDAxis(self.Kp, self.Ki, self.Kd, self.u_max)
        ctrl_lon = PIDAxis(self.Kp, self.Ki, self.Kd, self.u_max)

        # plant starts at the provided current location
        pos_lat, pos_lon = self.current_lat, self.current_lon

        h = {k: [] for k in ['meas_lat', 'meas_lon', 'est_lat', 'est_lon',
                             'true_lat', 'true_lon', 'eX', 'eY',
                             'pX', 'iX', 'dX', 'pY', 'iY', 'dY', 'uX', 'uY']}

        print("Simulating PID controlled flight to target...")
        for _ in range(total_steps):
            # sensor: noisy measurement of the true position
            z_lat = pos_lat + np.random.normal(0, 10)
            z_lon = pos_lon + np.random.normal(0, 10)

            # Kalman filter
            kf_lat.predict()
            s_lat = kf_lat.kalman(np.array([[z_lat]]))
            kf_lon.predict()
            s_lon = kf_lon.kalman(np.array([[z_lon]]))
            est_lat, vel_lat = s_lat[0, 0], s_lat[1, 0]
            est_lon, vel_lon = s_lon[0, 0], s_lon[1, 0]

            # error on the filtered estimate
            error_X = self.target_lat - est_lat
            error_Y = self.target_lon - est_lon

            u_lat, p_lat, i_lat, d_lat = ctrl_lat.update(error_X, -vel_lat, dt)
            u_lon, p_lon, i_lon, d_lon = ctrl_lon.update(error_Y, -vel_lon, dt)

            # plant integrates into new true position
            pos_lat += (u_lat + self.wind_lat) * dt
            pos_lon += (u_lon + self.wind_lon) * dt

            for k, v in zip(h.keys(),
                            [z_lat, z_lon, est_lat, est_lon, pos_lat, pos_lon,
                             error_X, error_Y, p_lat, i_lat, d_lat,
                             p_lon, i_lon, d_lon, u_lat, u_lon]):
                h[k].append(v)

        self.plot(RATE, h)

    def plot(self, rate, h):
        t = np.arange(len(h['est_lat'])) / rate
        
        # Increased figure size to prevent any overlapping
        fig, ax = plt.subplots(3, 2, figsize=(15, 14))

        # Latitude
        ax[0, 0].plot(t, h['meas_lat'], color='red', alpha=0.3, label='Measured')
        ax[0, 0].plot(t, h['est_lat'], color='blue', linewidth=2, label='Estimated')
        ax[0, 0].plot(t, h['true_lat'], color='black', linestyle='--', label='True')
        ax[0, 0].axhline(self.target_lat, color='green', linestyle=':', label='Target')
        ax[0, 0].set_title('Latitude'); ax[0, 0].set_xlabel('Time (s)')
        ax[0, 0].legend(loc='best', fontsize='small')

        # Longitude
        ax[0, 1].plot(t, h['meas_lon'], color='red', alpha=0.3, label='Measured')
        ax[0, 1].plot(t, h['est_lon'], color='blue', linewidth=2, label='Estimated')
        ax[0, 1].plot(t, h['true_lon'], color='black', linestyle='--', label='True')
        ax[0, 1].axhline(self.target_lon, color='green', linestyle=':', label='Target')
        ax[0, 1].set_title('Longitude'); ax[0, 1].set_xlabel('Time (s)')
        ax[0, 1].legend(loc='best', fontsize='small')

        # Error convergence
        ax[1, 0].plot(t, h['eX'], color='tab:green', label='Error Lat')
        ax[1, 0].plot(t, h['eY'], color='tab:pink', label='Error Lon')
        ax[1, 0].axhline(0, color='black', linewidth=0.8)
        ax[1, 0].set_title(f'Controller error (Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd})')
        ax[1, 0].set_xlabel('Time (s)'); ax[1, 0].set_ylabel('Error (deg)')
        ax[1, 0].legend(loc='best', fontsize='small')

        # Calculate final error in METERS
        final_lat = h['true_lat'][-1]
        final_lon = h['true_lon'][-1]
        
        # Approx: 1 degree Lat = 111,320 meters. 1 degree Lon = 111,320 * cos(lat) meters
        lat_err_m = (self.target_lat - final_lat) * 111320.0
        lon_err_m = (self.target_lon - final_lon) * (111320.0 * np.cos(np.radians(self.target_lat)))
        final_dist_m = np.sqrt(lat_err_m**2 + lon_err_m**2)

        # Trajectory Plot
        ax[1, 1].plot(h['true_lat'], h['true_lon'], color='black', label='True path')
        ax[1, 1].plot(h['est_lat'], h['est_lon'], color='blue', alpha=0.6, label='Estimated path')
        ax[1, 1].scatter(h['true_lat'][0], h['true_lon'][0], color='orange', s=80, zorder=5, label='Start')
        ax[1, 1].scatter(self.target_lat, self.target_lon, color='green', marker='*', s=200, zorder=5, label='Target')
        
        # Placed the error directly in the title to guarantee no overlap with the lines
        ax[1, 1].set_title(f'Trajectory\n[ Final Landed Error: {final_dist_m:.2f} meters ]', fontweight='bold')
        ax[1, 1].set_xlabel('Latitude'); ax[1, 1].set_ylabel('Longitude')
        ax[1, 1].legend(loc='best', fontsize='small')

        # Contribution of each term to the control output
        for col, (axis, p, i, d, u) in enumerate([('Lat', 'pX', 'iX', 'dX', 'uX'),
                                                  ('Lon', 'pY', 'iY', 'dY', 'uY')]):
            a = ax[2, col]
            a.plot(t, h[p], label='P term', color='tab:blue')
            a.plot(t, h[i], label='I term', color='tab:orange')
            a.plot(t, h[d], label='D term', color='tab:green')
            a.plot(t, h[u], label='Total u', color='black', linestyle='--')
            a.axhline(0, color='black', linewidth=0.8)
            a.set_title(f'{axis} control output breakdown')
            a.set_xlabel('Time (s)'); a.set_ylabel('u')
            a.legend(loc='best', fontsize='small')

        for a in ax.flat:
            a.grid(True, alpha=0.3)
            
        # Increased padding specifically to stop titles/labels from bleeding into each other
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        plt.show()


if __name__ == '__main__':
    CURRENT_LAT = 10.0000
    CURRENT_LON = -20.0000
    CURRENT_ALT = 1000.0  
    
    TARGET_LAT = 50.0000
    TARGET_LON = -45.0000

    p = PID(
        current_lat=CURRENT_LAT, 
        current_lon=CURRENT_LON, 
        current_alt=CURRENT_ALT, 
        target_lat=TARGET_LAT, 
        target_lon=TARGET_LON,
        wind_lat=0.0, 
        wind_lon=0.0
    )
    
    p.pid_controller()