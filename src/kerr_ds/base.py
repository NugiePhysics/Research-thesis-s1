"""Abstract interfaces: spacetime (Hamiltonian) and integrator."""
from abc import ABC, abstractmethod


class Spacetime(ABC):
    @abstractmethod
    def H(self, x, p):
        pass

    @abstractmethod
    def dHdx(self, x, p):
        pass

    @abstractmethod
    def dHdp(self, x, p):
        pass

    def grad(self, x, p):
        """Return (dH/dx, dH/dp) together.

        The default calls dHdx and dHdp separately; subclasses may override it
        with a version that computes both in a single call.
        """
        return self.dHdx(x, p), self.dHdp(x, p)


class Integrator(ABC):
    @abstractmethod
    def step(self, x, p, dt):
        pass
