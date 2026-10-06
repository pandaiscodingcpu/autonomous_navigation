import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 1. NAVIGATION MATH & CONTROL
# ==========================================
def calculate_heading_error(current_pos, current_heading, target_pos):
    """Calculates the angular difference between current heading and the target."""
    # Calculate absolute angle to the target using atan2
    target_heading = np.arctan2(target_pos[1] - current_pos[1], 
                                target_pos[0] - current_pos[0])
    
    # Calculate difference
    error = target_heading - current_heading
    
    # Wrap the error to [-pi, pi] so the canopy always takes the shortest turn
    error = (error + np.pi) % (2 * np.pi) - np.pi
    return error

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error):
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (accumulated error over time)
        self.integral += error * self.dt
        
        # Anti-windup: clamp the integral to prevent massive over-steering 
        # if the CanSat gets stuck circling the target
        self.integral = np.clip(self.integral, -2.0, 2.0)
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        d_term = self.kd * ((error - self.prev_error) / self.dt)
        self.prev_error = error
        
        return p_term + i_term + d_term

# ==========================================
# 2. EXTENDED KALMAN FILTER
# ==========================================
class DescentEKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros(8)
        self.x[2] = 1000.0  # Initial altitude
        
        self.P = np.eye(8) * 0.2
        self.Q = np.eye(8) * 0.5
        
        self.R_gps = np.eye(2) * 25.0  # High noise for simulated GPS
        self.R_baro = np.array([[1.0]])
        
        self.H_gps = np.zeros((2, 8))
        self.H_gps[0, 0] = 1.0
        self.H_gps[1, 1] = 1.0
        
        self.H_baro = np.zeros((1, 8))
        self.H_baro[0, 2] = 1.0

    def predict(self, accel_xyz, gyro_z):
        # State: [x, y, z, vx, vy, vz, psi, psi_dot]
        self.x[0] += self.x[3] * self.dt
        self.x[1] += self.x[4] * self.dt
        self.x[2] += self.x[5] * self.dt
        
        self.x[3] += accel_xyz[0] * self.dt
        self.x[4] += accel_xyz[1] * self.dt
        self.x[5] += accel_xyz[2] * self.dt 
        
        self.x[6] += self.x[7] * self.dt
        self.x[7] = gyro_z 
        
        F = np.eye(8)
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt
        F[6, 7] = self.dt
        
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, meas_x, meas_y):
        z = np.array([meas_x, meas_y])
        y = z - (self.H_gps @ self.x)
        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps
        K = self.P @ self.H_gps.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(8) - K @ self.H_gps) @ self.P

    def update_baro(self, meas_z):
        z = np.array([meas_z])
        y = z - (self.H_baro @ self.x)
        S = self.H_baro @ self.P @ self.H_baro.T + self.R_baro
        K = self.P @ self.H_baro.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(8) - K @ self.H_baro) @ self.P

# ==========================================
# 3. KINEMATIC PARAGLIDER SIMULATOR (The "Truth")
# ==========================================
class ParagliderTruth:
    def __init__(self):
        # True State: x, y, z, vx, vy, vz, psi, psi_dot
        self.state = np.array([0.0, 0.0, 1000.0, 0.0, 0.0, -5.0, np.pi/4, 0.0])
        self.v_forward = 12.5 # Canopy forward airspeed (m/s)
        self.wind = np.array([0.5, -1.0]) # Crosswind pushing off course
        
    def step(self, commanded_yaw_rate, dt):
        # Apply steering command to turn rate
        self.state[7] = commanded_yaw_rate
        psi = self.state[6]
        
        # Calculate new velocities based on heading + wind
        self.state[3] = self.v_forward * np.cos(psi) + self.wind[0]
        self.state[4] = self.v_forward * np.sin(psi) + self.wind[1]
        self.state[5] = -10.0 # Constant terminal descent velocity
        
        # Update positions
        self.state[0] += self.state[3] * dt
        self.state[1] += self.state[4] * dt
        self.state[2] += self.state[5] * dt
        self.state[6] += self.state[7] * dt # Update heading
        
        return self.state.copy()

# ==========================================
# 4. MAIN ANIMATION LOOP
# ==========================================
def main():
    dt = 1 # 20Hz Flight Computer Loop
    
    # Initialize Systems
    truth_model = ParagliderTruth()
    ekf = DescentEKF(dt)
    # Tuned PID (Kp for response, Kd for damping, Ki for steady wind drift)
    pid = PIDController(kp=0.7, ki=0.001, kd=0.2, dt=dt)
    # Mission Targets
    start_pos = np.array([0, 0])
    target_pos = np.array([800, 800])
    
    # Data Tracking for Plotting
    true_history = []
    est_history = []
    gps_measurements = []
    
    # Setup Matplotlib 3D Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("CanSat EKF vs True Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_xlim(-200, 1000)
    ax.set_ylim(-200, 1000)
    ax.set_zlim(0, 1000)
    
    # Plot elements
    ax.plot([start_pos[0], target_pos[0]], [start_pos[1], target_pos[1]], [1000, 0], 
            'k--', alpha=0.5, label='Target Flight Path')
            
    true_line, = ax.plot([], [], [], 'g-', linewidth=2, label='True Flight Path')
    est_line, = ax.plot([], [], [], 'b-', linewidth=2, label='EKF Estimated Path')
    gps_scatter, = ax.plot([], [], [], 'rx', markersize=4, alpha=0.3, label='Noisy GPS Data')
    
    ax.legend()
    
    sim_time = [0.0]
    yaw_cmd = [0.0]

    def update(frame):
        # Run 4 physics steps per animation frame to speed up the visual
        for _ in range(4):
            if truth_model.state[2] <= 0:
                # Calculate final landing distance error
                final_x = truth_model.state[0]
                final_y = truth_model.state[1]
                landing_error = np.linalg.norm(np.array([final_x, final_y]) - target_pos)
                
                # Update the plot title dynamically
                ax.set_title(f"CanSat Trajectory | Landing Error: {landing_error:.2f} m", 
                             color='red', fontweight='bold')
                             
                return true_line, est_line, gps_scatter # Reached Ground
                
            # 1. Physics Step
            true_state = truth_model.step(yaw_cmd[0], dt)
            
            # 2. Synthesize Sensors
            # IMU: approximate true acceleration as 0 + noise, gyro is true yaw_rate + noise
            sim_accel = np.random.normal(0, 0.2, 3) 
            sim_gyro_z = true_state[7] + np.random.normal(0, 0.05)
            
            ekf.predict(sim_accel, sim_gyro_z)
            
            # GPS Update (5Hz -> every 4 loops of 20Hz)
            if frame % 4 == 0:
                sim_gps_x = true_state[0] + np.random.normal(0, 5.0)
                sim_gps_y = true_state[1] + np.random.normal(0, 5.0)
                ekf.update_gps(sim_gps_x, sim_gps_y)
                gps_measurements.append([sim_gps_x, sim_gps_y, true_state[2]]) # store for plotting
            
            # Baro Update (20Hz)
            sim_baro_z = true_state[2] + np.random.normal(0, 0.5)
            ekf.update_baro(sim_baro_z)
            
            # 3. Control Step (using EKF state)
            # 3. Control Step (using EKF state)
            est_state = ekf.x.copy()
            
            # Use estimated position and estimated heading to find the error
            heading_error = calculate_heading_error(
                current_pos=est_state[:2],
                current_heading=est_state[6],
                target_pos=target_pos
            )
            
            # PID computes required yaw rate to turn towards the target
            yaw_cmd[0] = pid.update(heading_error)
            
            # Store data
            true_history.append(true_state[:3].copy())
            est_history.append(est_state[:3].copy())
            sim_time[0] += dt

        # Update Plots
        if len(true_history) > 0:
            true_arr = np.array(true_history)
            est_arr = np.array(est_history)
            gps_arr = np.array(gps_measurements)
            
            true_line.set_data(true_arr[:, 0], true_arr[:, 1])
            true_line.set_3d_properties(true_arr[:, 2])
            
            est_line.set_data(est_arr[:, 0], est_arr[:, 1])
            est_line.set_3d_properties(est_arr[:, 2])
            
            if len(gps_arr) > 0:
                gps_scatter.set_data(gps_arr[:, 0], gps_arr[:, 1])
                gps_scatter.set_3d_properties(gps_arr[:, 2])
                
        return true_line, est_line, gps_scatter

    ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()