import math
import random
from plot import render_3d_flight

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_target_bearing(x_curr, y_curr, x_target, y_target):
    angle_rad = math.atan2(y_target - y_curr, x_target - x_curr)
    return math.degrees(angle_rad)


def get_heading_error(current_heading, target_bearing):
    error = target_bearing - current_heading
    error = (error + 180) % 360 - 180
    return round(error, 2)


def simulate_cansat_3d(x_start, y_start, z_start, x_target, y_target, initial_heading):
    x, y, z = x_start, y_start, z_start
    heading = initial_heading

    step_size = 2    # Horizontal speed (m/s)
    descent_rate = 20 + random.random()    # Vertical sink rate (m/s)

    # Tracking arrays for 3D plotting
    x_history = [x]
    y_history = [y]
    z_history = [z]

    print(f"Target Launch Pad Coords: ({x_target}, {y_target}, 0)")

    for second in range(1, 600):
        distance = calculate_distance(x, y, x_target, y_target)

        current_step = min(step_size, distance)
        if z <= 0:
            print(f"\n[LANDED] CanSat touched down on the ground at {second}s")
            print(f"Final distance deviation from Pad: {distance:.2f} meters.")
            break

        target_bearing = get_target_bearing(x, y, x_target, y_target)
        steering_error = get_heading_error(heading, target_bearing)

        # Proportional Steering Adjustment
        heading += steering_error * 0.2
        heading_rad = math.radians(heading)

        x += current_step * math.cos(heading_rad)
        y += current_step * math.sin(heading_rad)
        z -= descent_rate + random.uniform(0,2)

        x_history.append(x)
        y_history.append(y)
        z_history.append(max(0, z))

        if second % 10 == 0 or distance < 2:
            print(f"Time: {second}s  Pos: ({x:.1f}, {y:.1f})  Alt: {z:.1f}m  Distance to Pad: {distance:.1f}m")

    render_3d_flight(x_history, y_history, z_history, target_coords=(x_target, y_target), start_coords=(x_start, y_start, z_start))


if __name__ == "__main__":
    simulate_cansat_3d(x_start=-5,y_start=0,z_start=1000,x_target=0,y_target=-100,initial_heading=90)