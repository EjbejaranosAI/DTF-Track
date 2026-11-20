"""2D constant-velocity EKF (parity with Spencer srl_nearest_neighbor_tracker / 4D EKF)."""

from __future__ import annotations

import numpy as np


class ConstantVelocityEKF:
    """State [x, y, vx, vy], measurement [x, y]. Linear dynamics => equivalent to Spencer's KF."""

    def __init__(
        self,
        map_x: float,
        map_y: float,
        *,
        process_noise_diag: tuple[float, float, float, float] = (1e-2, 1e-2, 0.55, 0.55),
        measurement_noise: float = 5e-2,
    ) -> None:
        self.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.transitionMatrix = np.eye(4, dtype=np.float32)
        self.processNoiseCov = np.diag(process_noise_diag).astype(np.float32)
        self._R_base = np.eye(2, dtype=np.float32) * float(measurement_noise)
        self.measurementNoiseCov = self._R_base.copy()
        self.statePost = np.array(
            [[float(map_x)], [float(map_y)], [0.0], [0.0]], np.float32
        )
        self.statePre = self.statePost.copy()
        self.errorCovPost = np.eye(4, dtype=np.float32)
        self.errorCovPre = self.errorCovPost.copy()

    def predict(self, dt_sec: float) -> np.ndarray:
        dt = max(float(dt_sec), 1e-3)
        f = np.eye(4, dtype=np.float32)
        f[0, 2] = dt
        f[1, 3] = dt
        self.transitionMatrix = f
        self.statePre = f @ self.statePost
        self.errorCovPre = f @ self.errorCovPost @ f.T + self.processNoiseCov
        return self.statePre

    def correct(self, map_x: float, map_y: float) -> None:
        h = self.measurementMatrix
        r = self.measurementNoiseCov
        z = np.array([[float(map_x)], [float(map_y)]], np.float32)
        s = h @ self.errorCovPre @ h.T + r
        k = self.errorCovPre @ h.T @ np.linalg.inv(s)
        innov = z - h @ self.statePre
        self.statePost = self.statePre + k @ innov
        i = np.eye(4, dtype=np.float32)
        self.errorCovPost = (i - k @ h) @ self.errorCovPre

    def occluded_update(self) -> None:
        """Spencer EKF::updateOccludedTrack — posterior = prediccion."""
        np.copyto(self.statePost, self.statePre)
        np.copyto(self.errorCovPost, self.errorCovPre)
