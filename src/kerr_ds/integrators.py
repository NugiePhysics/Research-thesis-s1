"""Integrator numerik untuk persamaan Hamilton dx/dτ = ∂H/∂p, dp/dτ = -∂H/∂x."""
from .base import Integrator


class RungeKutta4(Integrator):
    def __init__(self, sptm):
        self.sptm = sptm

    def step(self, x, p, dt):
        grad = self.sptm.grad

        # Step 1
        dHx1, dHp1 = grad(x, p)
        k1 = -dHx1 * dt
        l1 = dHp1 * dt

        # Step 2
        dHx2, dHp2 = grad(x + 0.5*l1, p + 0.5*k1)
        k2 = -dHx2 * dt
        l2 = dHp2 * dt

        # Step 3
        dHx3, dHp3 = grad(x + 0.5*l2, p + 0.5*k2)
        k3 = -dHx3 * dt
        l3 = dHp3 * dt

        # Step 4
        dHx4, dHp4 = grad(x + l3, p + k3)
        k4 = -dHx4 * dt
        l4 = dHp4 * dt

        p_new = p + (k1 + 2*k2 + 2*k3 + k4) / 6.
        x_new = x + (l1 + 2*l2 + 2*l3 + l4) / 6.

        return x_new, p_new
