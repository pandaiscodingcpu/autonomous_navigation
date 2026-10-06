import numpy as np
import matplotlib.pyplot as plt
import math

class ExtendedKalmanFilter:
    def __init__(self, rate):
        self.dT = 1.0 / rate

        # State vector: [p_N, p_E, p_D, psi, V_a, v_D, W_N, W_E]^T
        self.Xk = np.zeros((8, 1))

        # Initial Covariance Matrix
        self.P = np.eye(8) * 500.0

        # Process Noise Covariance (Q)
        self.Q = np.eye(8) * 0.01
        self.Q[6, 6] = 1.2  # Wind North might change faster
        self.Q[7, 7] = 1.5  # Wind East might change faster

        # Measurement Noise Covariance (R)
        self.R = np.eye(3) * 10.0

        # Measurement Matrix (H)
        self.H = np.zeros((3, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.I = np.eye(8)

    def init_state(self, pN0, pE0, pD0, psi0=0.0):
        """Initialize the starting coordinates and heading."""
        self.Xk = np.array([[pN0], [pE0], [pD0], [psi0], [0.0], [0.0], [0.0], [0.0]])

    def predict(self, omega_z):
        """Non-linear prediction step."""
        dt = self.dT
        pN, pE, pD, psi, Va, vD, WN, WE = self.Xk.flatten()

        # 1. Non-linear state transition f(x, u)
        pN_next = pN + (Va * np.cos(psi) + WN) * dt
        pE_next = pE + (Va * np.sin(psi) + WE) * dt
        pD_next = pD + vD * dt
        psi_next = psi + omega_z * dt

        self.Xk[0, 0] = pN_next
        self.Xk[1, 0] = pE_next
        self.Xk[2, 0] = pD_next
        self.Xk[3, 0] = psi_next

        # 2. Compute Jacobian matrix (F) of f(x, u)
        F = np.eye(8)
        F[0, 3] = -Va * np.sin(psi) * dt
        F[0, 4] = np.cos(psi) * dt
        F[0, 6] = dt
        F[1, 3] = Va * np.cos(psi) * dt
        F[1, 4] = np.sin(psi) * dt
        F[1, 7] = dt
        F[2, 5] = dt

        # 3. Predict Covariance
        self.P = (F @ self.P @ F.T) + self.Q

    def update(self, z):
        """Measurement update step."""
        y = z - (self.H @ self.Xk)
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.Xk = self.Xk + (K @ y)
        self.P = (self.I - K @ self.H) @ self.P
        return self.Xk


class AutonomousNavigator:
    def __init__(self, rate):
        self.rate = rate
        self.EARTH_RADIUS = 6371000.0  # meters

    def geodetic_to_ned(self, lat, lon, ref_lat, ref_lon):
        """
        Converts Lat/Lon to local North and East distances in meters 
        relative to a reference Lat/Lon starting point.
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)

        delta_lat = lat_rad - ref_lat_rad
        delta_lon = lon_rad - ref_lon_rad

        pos_N = delta_lat * self.EARTH_RADIUS
        pos_E = delta_lon * self.EARTH_RADIUS * math.cos(ref_lat_rad)
        
        return pos_N, pos_E

    def navigate_to_target(self, current_lat, current_lon, current_alt, target_lat, target_lon, target_alt=0.0):
        """
        Simulates flying from current coordinates to target coordinates,
        generating noisy GPS data and filtering it with the EKF.
        """
        ekf = ExtendedKalmanFilter(self.rate)

        # 1. Convert everything to Local NED coordinates (Current loc is origin 0,0)
        start_N, start_E = 0.0, 0.0 
        target_N, target_E = self.geodetic_to_ned(target_lat, target_lon, current_lat, current_lon)

        print(f"Target is {target_N:.1f}m North, {target_E:.1f}m East of starting location.")

        # Calculate ideal heading and distance
        distance = math.sqrt(target_N**2 + target_E**2)
        ideal_heading = math.atan2(target_E, target_N)
        
        # Flight simulation parameters
        airspeed = 15.0  # m/s
        descent_rate = (target_alt - current_alt) / (distance / airspeed) if distance > 0 else 0
        total_steps = int((distance / airspeed) * self.rate)

        # Initialize EKF
        ekf.init_state(start_N, start_E, current_alt, psi0=ideal_heading)

        # Data storage for plotting
        history = {
            'noisy_pN': [], 'noisy_pE': [], 'noisy_pD': [],
            'est_pN': [], 'est_pE': [], 'est_pD': [], 
            'est_Va': [], 'est_vD': [], 'est_WN': [], 'est_WE': []
        }

        # Simulated True States
        true_N, true_E, true_D = start_N, start_E, current_alt

        print("Simulating flight and running EKF...")
        for i in range(total_steps):
            # --- SIMULATE MOVEMENT ---
            true_N += (airspeed * math.cos(ideal_heading)) * (1.0 / self.rate)
            true_E += (airspeed * math.sin(ideal_heading)) * (1.0 / self.rate)
            true_D += descent_rate * (1.0 / self.rate)

            # Generate noisy "GPS" measurements
            meas_N = true_N + np.random.normal(0, 3.0) # 3m GPS noise
            meas_E = true_E + np.random.normal(0, 3.0)
            meas_D = true_D + np.random.normal(0, 1.5) # 1.5m Baro/GPS alt noise
            
            # --- EKF LOOP ---
            omega_z = 0.0 # Assuming straight flight to target for demo
            ekf.predict(omega_z)

            z = np.array([[meas_N], [meas_E], [meas_D]])
            state = ekf.update(z)

            # --- LOGGING ---
            history['noisy_pN'].append(meas_N)
            history['noisy_pE'].append(meas_E)
            history['noisy_pD'].append(meas_D)
            
            history['est_pN'].append(state[0, 0])
            history['est_pE'].append(state[1, 0])
            history['est_pD'].append(state[2, 0])
            history['est_Va'].append(state[4, 0])
            history['est_vD'].append(state[5, 0])
            history['est_WN'].append(state[6, 0])
            history['est_WE'].append(state[7, 0])

        self.plot_results(history, total_steps)

    def plot_results(self, h, total_steps):
        t = np.arange(total_steps) / self.rate
        
        # --- Window 1: Position Filtering (Noisy vs Filtered) ---
        fig1, ax1 = plt.subplots(3, 1, figsize=(12, 10))
        fig1.canvas.manager.set_window_title('EKF: Position Filtering')
        
        ax1[0].plot(t, h['noisy_pN'], label='Noisy GPS North', color='red', alpha=0.4)
        ax1[0].plot(t, h['est_pN'], label='Filtered North', color='blue', linewidth=2)
        ax1[0].set_ylabel('North (m)')
        ax1[0].legend(); ax1[0].grid(True, alpha=0.3)
        
        ax1[1].plot(t, h['noisy_pE'], label='Noisy GPS East', color='red', alpha=0.4)
        ax1[1].plot(t, h['est_pE'], label='Filtered East', color='blue', linewidth=2)
        ax1[1].set_ylabel('East (m)')
        ax1[1].legend(); ax1[1].grid(True, alpha=0.3)
        
        ax1[2].plot(t, h['noisy_pD'], label='Noisy Altitude', color='red', alpha=0.4)
        ax1[2].plot(t, h['est_pD'], label='Filtered Altitude', color='blue', linewidth=2)
        ax1[2].set_ylabel('Altitude (m)'); ax1[2].set_xlabel('Time (s)')
        ax1[2].legend(); ax1[2].grid(True, alpha=0.3)
        
        fig1.tight_layout()

        # --- Window 2: 3D Trajectory ---
        fig2 = plt.figure(figsize=(10, 8))
        fig2.canvas.manager.set_window_title('EKF: 3D Flight Trajectory')
        ax2 = fig2.add_subplot(111, projection='3d')
        
        ax2.plot(h['noisy_pE'], h['noisy_pN'], h['noisy_pD'], 
                 label='Measured Path (Noisy)', color='red', alpha=0.3, linestyle='--')
        ax2.plot(h['est_pE'], h['est_pN'], h['est_pD'], 
                 label='EKF Filtered Path', color='blue', linewidth=2.5)
        
        ax2.scatter(h['est_pE'][0], h['est_pN'][0], h['est_pD'][0], 
                    color='green', s=100, label='Start Point', marker='o')
        ax2.scatter(h['est_pE'][-1], h['est_pN'][-1], h['est_pD'][-1], 
                    color='black', s=100, label='Target Point', marker='X')

        ax2.set_xlabel('East (m)'); ax2.set_ylabel('North (m)'); ax2.set_zlabel('Altitude (m)')
        ax2.legend()
        ax2.view_init(elev=20, azim=-45)

        plt.show()


if __name__ == '__main__':
    navigator = AutonomousNavigator(rate=1) # 10Hz loop rate

    # Define current location (e.g., somewhere in a field)
    CURRENT_LAT = 34.052235
    CURRENT_LON = -118.243683
    CURRENT_ALT = 1000.0  # meters

    # Define target location (e.g., landing zone ~1km away)
    TARGET_LAT = 34.060000
    TARGET_LON = -118.250000
    TARGET_ALT = 0.0

    print("Initializing Flight to Target...")
    navigator.navigate_to_target(
        CURRENT_LAT, CURRENT_LON, CURRENT_ALT,
        TARGET_LAT, TARGET_LON, TARGET_ALT
    )