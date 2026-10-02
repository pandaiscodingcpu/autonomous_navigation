'''
This physics environment is contributed by Rishabh Palany
'''
import math
import random
import numpy as np
import matplotlib.pyplot as plt

DT = 0.1
ALTITUDE = 650
TARGET = (0, 0)

MONTE_CARLO_RUNS = 500

ZONE_LENGTH = 61
ZONE_WIDTH = 12.2

GLIDE_RATIO = 1.4
DESCENT_RATE_MEAN = 5

MAX_TURN = math.radians(70)

TAU_ACT = 0.5
TAU_TURN = 0.6

TURN_DRAG_GAIN = 0.9
MIN_FORWARD_SPEED = 3.5

GPS_NOISE = 2
DEPLOY_STD = 40

BASE_WIND = 5
WIND_DIR = math.radians(230)

ALPHA = 0.18
TURB_TAU = 2.5
TURB_SIGMA = 0.8

ALIGN_GAIN = 0.3

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def wind_profile(z):

    mag = BASE_WIND * (z / ALTITUDE) ** ALPHA

    wx = mag * math.cos(WIND_DIR)
    wy = mag * math.sin(WIND_DIR)

    return wx, wy

def turbulence(gx, gy):

    gx += (-gx / TURB_TAU) * DT + TURB_SIGMA * np.sqrt(DT) * np.random.randn()
    gy += (-gy / TURB_TAU) * DT + TURB_SIGMA * np.sqrt(DT) * np.random.randn()

    return gx, gy

def wind_model(z, gx, gy):

    wx, wy = wind_profile(z)

    gx, gy = turbulence(gx, gy)

    wx += gx
    wy += gy

    wz = np.random.normal(0, 0.2)

    return wx, wy, wz, gx, gy

def estimate_airspeed(Vg1, Vg2, heading_change):

    dx = Vg1[0] - Vg2[0]
    dy = Vg1[1] - Vg2[1]

    delta_v = math.sqrt(dx * dx + dy * dy)

    if abs(heading_change) < math.radians(10):
        return None

    Va = delta_v / (2 * math.sin(abs(heading_change) / 2))

    return Va

def forward_speed(descent, turn_rate):

    base = GLIDE_RATIO * descent

    turn_drag = TURN_DRAG_GAIN * abs(turn_rate)

    speed = base - turn_drag

    return max(speed, MIN_FORWARD_SPEED)

def controller(state, airspeed):

    omega_cmd = 0.0

    return omega_cmd

def run_single():

    wind_est_x = 0
    wind_est_y = 0
    airspeed_est = 7

    x = random.uniform(200, 500)
    y = random.uniform(-150, 150)

    x += np.random.normal(0, DEPLOY_STD)
    y += np.random.normal(0, DEPLOY_STD)

    z = ALTITUDE

    heading = random.uniform(-math.pi, math.pi)

    omega = 0

    gx, gy = 0, 0

    prev_v = (0, 0)
    prev_heading = heading

    prev_x = x
    prev_y = y

    traj_x = []
    traj_y = []

    while z > 0:

        wx, wy, wz, gx, gy = wind_model(z, gx, gy)

        descent = np.random.normal(DESCENT_RATE_MEAN, 0.4)

        v = forward_speed(descent, omega)

        vx_air = v * math.cos(heading)
        vy_air = v * math.sin(heading)

        vx_ground = vx_air + wx
        vy_ground = vy_air + wy

        x += vx_ground * DT
        y += vy_ground * DT

        vx_est = (x - prev_x) / DT
        vy_est = (y - prev_y) / DT

        prev_x = x
        prev_y = y

        z -= (descent - wz) * DT

        gps_x = x + np.random.normal(0, GPS_NOISE)
        gps_y = y + np.random.normal(0, GPS_NOISE)

        heading_change = heading - prev_heading

        Va_est = estimate_airspeed(prev_v, (vx_est, vy_est), heading_change)

        if Va_est is not None:
            airspeed_est = 0.8 * airspeed_est + 0.2 * Va_est
            airspeed_est = max(4, min(airspeed_est, 9))

        prev_v = (vx_est, vy_est)
        prev_heading = heading

        state = (gps_x, gps_y, heading, vx_est, vy_est, z)

        omega_cmd = controller(state, airspeed_est)

        omega += (omega_cmd - omega) * DT / TAU_ACT

        wind_align = ALIGN_GAIN * math.sin(wrap(WIND_DIR - heading))

        heading += (omega + wind_align) * DT / TAU_TURN

        traj_x.append(x)
        traj_y.append(y)

    error = math.hypot(x, y)

    inside = (abs(x) <= ZONE_LENGTH / 2) and (abs(y) <= ZONE_WIDTH / 2)

    return error, inside, x, y, traj_x, traj_y

def monte_carlo():

    errors = []
    inside_count = 0

    xs = []
    ys = []

    for i in range(MONTE_CARLO_RUNS):

        e, inside, x, y, tx, ty = run_single()

        errors.append(e)

        xs.append(x)
        ys.append(y)

        if inside:
            inside_count += 1

    errors = np.array(errors)

    print("\nRESULTS")
    print("Mean error:", round(np.mean(errors), 2))
    print("95% error:", round(np.percentile(errors, 95), 2))
    print("Landing zone hit probability:",
          round(inside_count / MONTE_CARLO_RUNS, 3))

    return xs, ys



xs, ys = monte_carlo()

plt.figure(figsize=(6, 6))

plt.scatter(xs, ys, s=10)

rect = plt.Rectangle(
        (-ZONE_LENGTH / 2, -ZONE_WIDTH / 2),
        ZONE_LENGTH,
        ZONE_WIDTH,
        fill=False,
        color='red',
        linewidth=2
    )

plt.gca().add_patch(rect)

plt.axis('equal')

plt.title("Landing Dispersion")

plt.show()
