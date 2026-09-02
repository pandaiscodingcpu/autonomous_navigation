from dubin_planner import *
import matplotlib.pyplot as plt
from utils.plot import *

start_x = 1.0  # [m]
start_y = 1.0  # [m]
start_yaw = np.deg2rad(45.0)  # [rad]
end_x = -3.0  # [m]
end_y = -3.0  # [m]
end_yaw = np.deg2rad(-45.0)  # [rad]
curvature = 1.0
path_x, path_y, path_yaw, mode, _ = plan_dubins_path(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
plt.plot(path_x, path_y, label="final course " + "".join(mode))
plot_arrow(start_x, start_y, start_yaw)
plot_arrow(end_x, end_y, end_yaw)
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()