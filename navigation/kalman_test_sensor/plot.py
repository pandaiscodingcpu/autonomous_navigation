import sys
import time
import collections
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

COM_PORT = 'COM14'
BAUD_RATE = 115200
MAX_SAMPLES = 60  # Displays last 60 seconds of data

# Buffers for time, raw, and filtered values
time_buf = collections.deque(maxlen=MAX_SAMPLES)
raw_gyr_buf = collections.deque(maxlen=MAX_SAMPLES)
filt_gyr_buf = collections.deque(maxlen=MAX_SAMPLES)
raw_acc_buf = collections.deque(maxlen=MAX_SAMPLES)
filt_acc_buf = collections.deque(maxlen=MAX_SAMPLES)

start_time = time.time()

# Connect to Serial
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"\n[ERROR] Access Denied or Port Error on {COM_PORT}: {e}")
    print("Ensure the Arduino Serial Monitor / Serial Plotter is closed!\n")
    sys.exit(1)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig.canvas.manager.set_window_title("ICM-20948: Raw vs. Kalman Filter Comparison")

# Gyroscope Plots
line_raw_gyr, = ax1.plot([], [], 'r:', alpha=0.6, linewidth=1.5, marker='o', markersize=3, label='Noisy Raw Gyro X')
line_filt_gyr, = ax1.plot([], [], '#00FF00', linewidth=2.5, label='Kalman Filtered Gyro X')
ax1.set_ylabel('Gyro X (deg/s)')
ax1.set_title('Gyroscope X Axis (100 Hz Filtered, 1 Hz Output)')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right')

# Accelerometer Plots
line_raw_acc, = ax2.plot([], [], 'm:', alpha=0.6, linewidth=1.5, marker='o', markersize=3, label='Noisy Raw Accel Z')
line_filt_acc, = ax2.plot([], [], '#00E5FF', linewidth=2.5, label='Kalman Filtered Accel Z')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Accel Z (mg)')
ax2.set_title('Accelerometer Z Axis (100 Hz Filtered, 1 Hz Output)')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right')

def update(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            # Expecting: RawGyrX:val,FiltGyrX:val,RawAccZ:val,FiltAccZ:val
            parts = dict(item.split(':') for item in line.split(','))

            rgx = float(parts['RawGyrX'])
            fgx = float(parts['FiltGyrX'])
            raz = float(parts['RawAccZ'])
            faz = float(parts['FiltAccZ'])

            curr_t = time.time() - start_time

            time_buf.append(curr_t)
            raw_gyr_buf.append(rgx)
            filt_gyr_buf.append(fgx)
            raw_acc_buf.append(raz)
            filt_acc_buf.append(faz)
        except Exception:
            pass

    if len(time_buf) > 0:
        t_data = list(time_buf)

        line_raw_gyr.set_data(t_data, list(raw_gyr_buf))
        line_filt_gyr.set_data(t_data, list(filt_gyr_buf))

        line_raw_acc.set_data(t_data, list(raw_acc_buf))
        line_filt_acc.set_data(t_data, list(filt_acc_buf))

        ax1.set_xlim(min(t_data), max(t_data) + 1.0)
        ax2.set_xlim(min(t_data), max(t_data) + 1.0)

        ax1.relim()
        ax1.autoscale_view(scalex=False, scaley=True)
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)

    return line_raw_gyr, line_filt_gyr, line_raw_acc, line_filt_acc

ani = animation.FuncAnimation(fig, update, interval=200, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()

if ser.is_open:
    ser.close()