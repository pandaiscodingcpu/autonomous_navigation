import numpy as np


def wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class ExtendedKalman:
    """
    EKF estimating position, heading, airspeed, wind and gyro bias.

    State x = [pN, pE, h, psi, Va, wN, wE, b_gyro]
      pN, pE : position North/East of the release point [m]
      h      : altitude [m]
      psi    : heading, clockwise from North [rad]
      Va     : horizontal airspeed [m/s]
      wN, wE : wind velocity components [m/s]
      b_gyro : gyro yaw-rate bias [rad/s]

    Input u = gyro yaw rate r [rad/s] (plus optional sink rate Vs [m/s]).
    """

    PN, PE, H, PSI, VA, WN, WE, BG = range(8)

    def __init__(self, dT,
                 gyro_noise=0.01,       # gyro noise density [rad/s/sqrt(Hz)]
                 gyro_bias_walk=1e-4,   # bias random walk [rad/s^2/sqrt(Hz)]
                 va_walk=0.2,           # airspeed random walk [m/s^2/sqrt(Hz)]
                 wind_walk=0.05,        # wind random walk [m/s^2/sqrt(Hz)]
                 pos_noise=0.05,        # small position process noise [m/sqrt(Hz)]
                 alt_noise=0.1):        # altitude process noise [m/sqrt(Hz)]
        self.dt = dT
        self.rate = 1.0 / dT            # filter rate [Hz]
        self.n = 8

        # ---- initial state and covariance (override with set_initial_state) ----
        self.x = np.zeros(self.n)
        self.x[self.VA] = 8.0
        self.P = np.diag([5.0**2, 5.0**2, 5.0**2,
                          np.deg2rad(30)**2,    # heading
                          3.0**2,               # airspeed
                          5.0**2, 5.0**2,       # wind: start very uncertain
                          np.deg2rad(2)**2])    # gyro bias

        # ---- process noise (continuous spectral density * dt) ----
        q = np.zeros(self.n)
        q[self.PN] = q[self.PE] = pos_noise**2
        q[self.H] = alt_noise**2
        q[self.PSI] = gyro_noise**2           # gyro noise integrates into heading
        q[self.VA] = va_walk**2
        q[self.WN] = q[self.WE] = wind_walk**2
        q[self.BG] = gyro_bias_walk**2
        self.Q = np.diag(q) * dT

    # ------------------------------------------------------------------
    def set_initial_state(self, pN=0.0, pE=0.0, h=0.0, psi=0.0, Va=8.0,
                          wN=0.0, wE=0.0, bias=0.0, P=None):
        self.x = np.array([pN, pE, h, wrap(psi), Va, wN, wE, bias], float)
        if P is not None:
            self.P = np.array(P, float)

    # ------------------------------------------------------------------
    def predict(self, gyro_r, sink_rate=0.0):
        """Call at the gyro rate (e.g. 100 Hz). sink_rate is positive downward."""
        dt = self.dt
        pN, pE, h, psi, Va, wN, wE, bg = self.x
        c, s = np.cos(psi), np.sin(psi)

        # nonlinear process model f(x, u)
        self.x = np.array([
            pN + (Va * c + wN) * dt,
            pE + (Va * s + wE) * dt,
            h - sink_rate * dt,
            wrap(psi + (gyro_r - bg) * dt),
            Va, wN, wE, bg,
        ])

        # Jacobian F = df/dx
        F = np.eye(self.n)
        F[self.PN, self.PSI] = -Va * s * dt
        F[self.PN, self.VA] = c * dt
        F[self.PN, self.WN] = dt
        F[self.PE, self.PSI] = Va * c * dt
        F[self.PE, self.VA] = s * dt
        F[self.PE, self.WE] = dt
        F[self.PSI, self.BG] = -dt

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    def _update(self, z, z_pred, H, R, angle_rows=()):
        """Generic EKF measurement update (Joseph form for numerical safety)."""
        z = np.atleast_1d(z).astype(float)
        z_pred = np.atleast_1d(z_pred).astype(float)
        H = np.atleast_2d(H)
        R = np.atleast_2d(R)

        y = z - z_pred                      # innovation
        for i in angle_rows:                # wrap angular innovations
            y[i] = wrap(y[i])

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[self.PSI] = wrap(self.x[self.PSI])

        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return y, S                         # handy for innovation (consistency) checks

    # ---------------- measurement updates -----------------------------
    def update_gps_position(self, pN, pE, sigma=2.0):
        H = np.zeros((2, self.n)); H[0, self.PN] = 1; H[1, self.PE] = 1
        return self._update([pN, pE], self.x[[self.PN, self.PE]], H,
                            np.eye(2) * sigma**2)

    def update_gps_velocity(self, vN, vE, sigma=0.1):
        """The key measurement for wind: v_ground = Va*[cos psi, sin psi] + wind."""
        psi, Va = self.x[self.PSI], self.x[self.VA]
        c, s = np.cos(psi), np.sin(psi)
        z_pred = [Va * c + self.x[self.WN], Va * s + self.x[self.WE]]
        H = np.zeros((2, self.n))
        H[0, self.PSI] = -Va * s;  H[0, self.VA] = c;  H[0, self.WN] = 1
        H[1, self.PSI] = Va * c;   H[1, self.VA] = s;  H[1, self.WE] = 1
        return self._update([vN, vE], z_pred, H, np.eye(2) * sigma**2)

    def update_altitude(self, h, sigma=0.5):
        H = np.zeros((1, self.n)); H[0, self.H] = 1
        return self._update([h], [self.x[self.H]], H, [[sigma**2]])

    def update_heading(self, psi_meas, sigma=np.deg2rad(3)):
        """Tilt-compensated magnetometer heading (with declination applied)."""
        H = np.zeros((1, self.n)); H[0, self.PSI] = 1
        return self._update([psi_meas], [self.x[self.PSI]], H, [[sigma**2]],
                            angle_rows=(0,))

    def update_airspeed(self, sink_rate, glide_ratio, sigma=0.5):
        """Pseudo-measurement: Va ~= (G/D) * sink rate. Helps separate Va from wind."""
        H = np.zeros((1, self.n)); H[0, self.VA] = 1
        return self._update([glide_ratio * sink_rate], [self.x[self.VA]], H,
                            [[sigma**2]])

    # ---------------- convenient outputs -------------------------------
    @property
    def heading(self):
        return self.x[self.PSI]

    @property
    def wind(self):
        return self.x[self.WN], self.x[self.WE]

    @property
    def airspeed(self):
        return self.x[self.VA]


# ----------------------------------------------------------------------
# Quick simulation to sanity-check convergence
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dt = 0.01
    ekf = ExtendedKalman(dt)
    ekf.set_initial_state(psi=0.5, Va=6.0)        # deliberately wrong guesses

    # truth
    Va_true, wind_true, bias_true = 8.0, np.array([3.0, -2.0]), 0.02
    GR, sink = 2.0, 4.0                            # glide ratio, sink rate
    pos = np.zeros(2)
    psi_true = 0.0

    for k in range(int(120 / dt)):
        t = k * dt
        r_true = 0.25 * np.sin(0.15 * t)           # gentle S-turns
        psi_true = wrap(psi_true + r_true * dt)
        vel = Va_true * np.array([np.cos(psi_true), np.sin(psi_true)]) + wind_true
        pos = pos + vel * dt

        gyro = r_true + bias_true + rng.normal(0, 0.01 / np.sqrt(dt) * 0.1)
        ekf.predict(gyro, sink_rate=sink)

        if k % 10 == 0:                            # GPS at 10 Hz
            ekf.update_gps_velocity(*(vel + rng.normal(0, 0.1, 2)), sigma=0.1)
            ekf.update_gps_position(*(pos + rng.normal(0, 2.0, 2)), sigma=2.0)
            ekf.update_heading(psi_true + rng.normal(0, np.deg2rad(3)))
            ekf.update_airspeed(sink + rng.normal(0, 0.2), GR, sigma=0.5)

    wN, wE = ekf.wind
    print(f"wind estimate : N={wN:.2f} E={wE:.2f}   (true {wind_true[0]}, {wind_true[1]})")
    print(f"airspeed      : {ekf.airspeed:.2f}   (true {Va_true})")
    print(f"heading error : {np.rad2deg(wrap(ekf.heading - psi_true)):.2f} deg")
    print(f"gyro bias     : {ekf.x[ekf.BG]:.4f}   (true {bias_true})")
