"""Constant-velocity EKF (Spencer parity)."""

import numpy as np

from dynamic_tracking.ekf2d import ConstantVelocityEKF


def test_predict_advances_position_with_velocity():
    kf = ConstantVelocityEKF(0.0, 0.0)
    kf.statePost[2, 0] = 1.0
    kf.predict(0.5)
    assert abs(float(kf.statePre[0, 0]) - 0.5) < 1e-5


def test_occluded_update_copies_prediction():
    kf = ConstantVelocityEKF(1.0, 2.0)
    kf.predict(0.1)
    kf.occluded_update()
    assert float(kf.statePost[0, 0]) == float(kf.statePre[0, 0])
    np.testing.assert_allclose(kf.errorCovPost, kf.errorCovPre)
