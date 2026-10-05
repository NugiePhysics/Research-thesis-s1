"""Antarmuka abstrak: ruang-waktu (Hamiltonian) dan integrator."""
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
        """Kembalikan (dH/dx, dH/dp) sekaligus.

        Default memanggil dHdx dan dHdp terpisah; subclass boleh override
        dengan versi yang menghitung keduanya dalam satu panggilan.
        """
        return self.dHdx(x, p), self.dHdp(x, p)


class Integrator(ABC):
    @abstractmethod
    def step(self, x, p, dt):
        pass
