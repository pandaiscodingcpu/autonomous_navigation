"""State estimator: EKF (position, heading, airspeed, wind, gyro bias) + 1D descent KF."""
import numpy as np
from extended_kalman import ExtendedKalman
from descent_kf import DescentKF


class StateEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ekf = ExtendedKalman(
            cfg.dt,
            gyro_noise=cfg.ekf_gyro_noise,
            gyro_bias_walk=cfg.ekf_gyro_bias_walk,
            va_walk=cfg.ekf_va_walk,
            wind_walk=cfg.ekf_wind_walk,
            pos_noise=cfg.ekf_pos_noise,
            alt_noise=cfg.ekf_alt_noise,
        )
        self.dkf = DescentKF(cfg.dt, cfg.dkf_accel_sigma, cfg.baro_sigma)

    # ---------- initialisation from the first measurements ----------
    def initialize(self, gps, mag_psi, baro_h):
        c = self.cfg
        pN, pE, vN, vE = gps
        va = c.va0                                   # nominal airspeed (design value)
        # first wind guess: GPS velocity minus nominal air velocity along the measured heading
        wN = vN - va * np.cos(mag_psi)
        wE = vE - va * np.sin(mag_psi)
        self.ekf.set_initial_state(pN, pE, baro_h, mag_psi, va, wN, wE, 0.0)
        E = self.ekf
        E.P[E.PSI, E.PSI] = np.deg2rad(c.mag_sigma_deg) ** 2
        E.P[E.PN, E.PN] = E.P[E.PE, E.PE] = c.gps_pos_sigma ** 2
        E.P[E.WN, E.WN] = E.P[E.WE, E.WE] = 3.0 ** 2
        E.P[E.H, E.H] = c.baro_sigma ** 2
        self.dkf.reset(baro_h, va / c.glide_ratio_est)

    # ---------- 100 Hz ----------
    def predict(self, gyro_r):
        self.dkf.predict()
        self.ekf.predict(gyro_r, sink_rate=self.dkf.vd)   # sink rate from the 1D KF, not dh/dt

    # ---------- sensor updates ----------
    def update_gps(self, pN, pE, vN, vE):
        c = self.cfg
        self.ekf.update_gps_velocity(vN, vE, sigma=c.gps_vel_sigma)
        self.ekf.update_gps_position(pN, pE, sigma=c.gps_pos_sigma)
        # Va ~= (G/D) * sink rate: helps separate airspeed from wind in straight flight
        self.ekf.update_airspeed(self.dkf.vd, c.glide_ratio_est, sigma=c.airspeed_sigma)

    def update_mag(self, psi_meas):
        self.ekf.update_heading(psi_meas, sigma=np.deg2rad(self.cfg.mag_sigma_deg))

    def update_baro(self, h_meas):
        self.dkf.update(h_meas)
        self.ekf.update_altitude(h_meas, sigma=self.cfg.baro_sigma)

    # ---------- outputs ----------
    @property
    def position(self):
        return self.ekf.x[[self.ekf.PN, self.ekf.PE]]

    @property
    def heading(self):
        return self.ekf.heading

    @property
    def wind(self):
        return np.array(self.ekf.wind)

    @property
    def airspeed(self):
        return self.ekf.airspeed

    @property
    def altitude(self):
        return self.dkf.h

    @property
    def descent_velocity(self):
        return self.dkf.vd

    @property
    def gyro_bias(self):
        return self.ekf.x[self.ekf.BG]

    @property
    def sigmas(self):
        d = np.sqrt(np.diag(self.ekf.P))
        E = self.ekf
        return dict(wN=d[E.WN], wE=d[E.WE], Va=d[E.VA], psi=d[E.PSI])
