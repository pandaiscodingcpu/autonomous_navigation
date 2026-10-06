import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- Configuration ---
# Update this to match your Arduino's COM port (e.g., 'COM3', 'COM4')
SERIAL_PORT = 'COM9' 
BAUD_RATE = 115200
MAX_POINTS = 200  # Number of data points to keep on the screen at once

# --- Data Storage ---
# deques act as sliding windows; they automatically drop old data when full
t_data = deque(maxlen=MAX_POINTS)
raw_alt_data = deque(maxlen=MAX_POINTS)
filt_alt_data = deque(maxlen=MAX_POINTS)
filt_vel_data = deque(maxlen=MAX_POINTS)

# Initialize Serial Connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# --- Figure Setup ---
fig, (ax_alt, ax_vel) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Live 1D Kalman Filter Telemetry')

# Altitude Plot Lines
line_raw_alt, = ax_alt.plot([], [], label='Noisy Altitude', color='red', alpha=0.5)
line_filt_alt, = ax_alt.plot([], [], label='Filtered Altitude', color='blue', linewidth=2)
ax_alt.set_title('Altitude')
ax_alt.set_ylabel('Meters')
ax_alt.legend(loc='upper right')

# Velocity Plot Line
line_filt_vel, = ax_vel.plot([], [], label='Filtered Velocity', color='green', linewidth=2)
ax_vel.set_title('Velocity (dx/dt)')
ax_vel.set_xlabel('Samples')
ax_vel.set_ylabel('Meters / Second')
ax_vel.legend(loc='upper right')

plt.tight_layout()

sample_count = 0

def update_plot(frame):
    global sample_count
    
    # Read all available lines from the serial buffer
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
                
            # Parse the comma-separated values: raw_alt, filt_alt, filt_vel
            data = line.split(',')
            if len(data) == 3:
                raw_alt = float(data[0])
                filt_alt = float(data[1])
                filt_vel = float(data[2])
                
                t_data.append(sample_count)
                raw_alt_data.append(raw_alt)
                filt_alt_data.append(filt_alt)
                filt_vel_data.append(filt_vel)
                
                sample_count += 1
        except (ValueError, UnicodeDecodeError):
            # Ignore garbled serial lines that often happen on startup
            pass

    # Update plot data if we have collected points
    if len(t_data) > 0:
        line_raw_alt.set_data(t_data, raw_alt_data)
        line_filt_alt.set_data(t_data, filt_alt_data)
        line_filt_vel.set_data(t_data, filt_vel_data)
        
        # Dynamically adjust the X-axis to scroll with the data
        ax_alt.set_xlim(t_data[0], t_data[-1])
        ax_vel.set_xlim(t_data[0], t_data[-1])
        
        # Dynamically adjust the Y-axis based on current window min/max
        ax_alt.set_ylim(min(min(raw_alt_data), min(filt_alt_data)) - 2, 
                        max(max(raw_alt_data), max(filt_alt_data)) + 2)
        ax_vel.set_ylim(min(filt_vel_data) - 1, max(filt_vel_data) + 1)

    return line_raw_alt, line_filt_alt, line_filt_vel

# Run the animation loop at a 50ms interval to match the 50Hz sensor rate
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("Plotting stopped.")
finally:
    ser.close()