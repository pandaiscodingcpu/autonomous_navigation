import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def render_3d_flight(x_history, y_history, z_history, target_coords, start_coords):
    """
    Handles all Matplotlib 3D plotting configurations.
    """
    x_start, y_start, z_start = start_coords
    x_target, y_target = target_coords

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D flight trajectory curve
    ax.plot(
        x_history,
        y_history,
        z_history,
        label="CanSat Descent Path",
        color="royalblue",
        linewidth=2.5,
    )

    # Highlight the ejection/start point in the sky
    ax.scatter(
        x_start,
        y_start,
        z_start,
        color="darkorange",
        marker="^",
        s=100,
        label="Ejection Start",
    )

    # Highlight the Ground Target Launch Pad
    ax.scatter(
        x_target,
        y_target,
        0,
        color="crimson",
        marker="X",
        s=150,
        label="Launch Pad Target (Ground)",
    )

    # Visual aids: Draw a vertical drop-line directly beneath the start position
    ax.plot(
        [x_start, x_start],
        [y_start, y_start],
        [0, z_start],
        color="gray",
        linestyle=":",
        alpha=0.5,
    )

    # Labels and Titles
    ax.set_title("CanSat 3D Auto-Steering Tracking Profile", fontsize=14)
    ax.set_xlabel("X Coordinate (meters)", fontsize=10)
    ax.set_ylabel("Y Coordinate (meters)", fontsize=10)
    ax.set_zlabel("Altitude / Z Coordinate (meters)", fontsize=10)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_box_aspect([1, 1, 1])

    plt.show()