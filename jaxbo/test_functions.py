from abc import ABC, abstractmethod
from typing import Callable, final
import jax.numpy as np
from jaxbo.input_priors import uniform_prior, gaussian_prior, Prior

class TestFunction(ABC):
    """
    Abstract base class for test functions with prior, dimension, and domain bounds.

    Note: all functions are designed for single point evaluation (use vmap to vectorize)
    """

    def __init__(self, dim: int, lb: np.ndarray, ub: np.ndarray, prior: Prior):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.prior = prior

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at a given point x."""
        pass

    def __iter__(self) -> tuple[Callable, Prior, int, np.ndarray, np.ndarray]:
        """
        Returns:
        f (Callable): Target function.
        p_x (Prior): Gaussian prior.
        dim (int): Input dimension.
        lb (np.ndarray): Lower bounds.
        ub (np.ndarray): Upper bounds.
        """
        return iter((self.evaluate, self.prior, self.dim, self.lb, self.ub))

class MultiFidelityTestFunction(TestFunction):
    """
    Abstract class for test functions with both low- and high-fidelity variants.
    """

    def __init__(self, dim: int, lb: np.ndarray, ub: np.ndarray, prior: Prior):
        super().__init__(dim, lb, ub, prior)

    @abstractmethod
    def evaluate_low(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def evaluate_high(self, x: np.ndarray) -> float:
        pass

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError("Use evaluate_low or evaluate_high for multi-fidelity functions.")

    def __iter__(self) -> tuple[tuple[Callable, Callable], Prior, int, np.ndarray, np.ndarray]:
        return iter(((self.evaluate_low, self.evaluate_high), self.prior, self.dim, self.lb, self.ub))

@final
class OakleyFunction(TestFunction):
    """
    Oakley function (2D) with a Gaussian prior centered at the origin.
    """
    def __init__(self):
        dim = 2
        lb = -4.0 * np.ones(dim)
        ub =  4.0 * np.ones(dim)
        prior = gaussian_prior(np.zeros(dim), np.eye(dim))
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        return 5.0 + x[0] + x[1] + 2.0 * np.cos(x[0]) + 2.0 * np.sin(x[1])

@final
class MichalewiczFunction(TestFunction):
    """
    Michalewicz function (arbitrary dimension) with a Gaussian prior.

    Attributes:
        dim (int): Input dimensionality.
    """

    def __init__(self, dim: int = 2):
        lb = 0.0 * np.ones(dim)
        ub = np.pi * np.ones(dim)
        prior = gaussian_prior(np.full(dim, 0.5 * np.pi), 0.1 * np.eye(dim))
        self._m = 10.0
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Michalewicz function at a given point.

        Args:
            x (np.ndarray): Input vector of shape (dim,).

        Returns:
            float: Function value.
        """
        y = sum(np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * self._m) for i in range(self.dim))
        return -y

@final
class AckleyFunction(TestFunction):
    """
    Ackley function (commonly used for optimization benchmarking).

    Attributes:
        dim (int): Input dimensionality.
    """

    def __init__(self, dim: int = 2):
        lb = -32.768 * np.ones(dim)
        ub = 32.768 * np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Ackley function at a given point.

        Args:
            x (np.ndarray): Input vector of shape (dim,).

        Returns:
            float: Function value.
        """
        a = 20.0
        b = 0.2
        c = 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / self.dim))
        term2 = -np.exp(np.sum(np.cos(c * x)) / self.dim)
        return term1 + term2 + a + np.exp(1.0)

@final
class BirdFunction(TestFunction):
    """
    Bird function (2D) with a non-convex landscape and multiple local minima.
    """

    def __init__(self):
        dim = 2
        lb = -2.0 * np.pi * np.ones(dim)
        ub = 2.0 * np.pi * np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Bird function at a given point.

        Args:
            x (np.ndarray): Input vector of shape (2,).

        Returns:
            float: Function value.
        """
        x1, x2 = x[0], x[1]
        return (
            np.sin(x1) * np.exp((1 - np.cos(x2))**2)
            + np.cos(x2) * np.exp((1 - np.sin(x1))**2)
            + (x1 - x2)**2
        )

@final
class RosenbrockFunction(TestFunction):
    """
    Rosenbrock function (2D) with an additional Gaussian bump.
    """

    def __init__(self):
        dim = 2
        lb = np.array([-1.0, -1.0])
        ub = np.array([0.5, 1.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        y = 74.0 + 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        y -= 400.0 * np.exp(-((x[0] + 1.0) ** 2 + (x[1] + 1.0) ** 2) / 0.1)
        return y

@final
class BraninFunction(TestFunction):
    """
    Branin function (2D) with a uniform prior.
    """
    def __init__(self):
        dim = 2
        lb = np.array([-5.0, 0.0])
        ub = np.array([10.0, 15.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        x1, x2 = x[0], x[1]
        y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return y

@final
class ModifiedBraninFunction(TestFunction):
    """
    Modified Branin function (2D) with a uniform prior.
    """
    def __init__(self):
        dim = 2
        lb = np.array([-5.0, 0.0])
        ub = np.array([10.0, 15.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        x1, x2 = x[0], x[1]
        f1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        f2 = s * (1 - t) * np.cos(x1) * np.cos(x2)
        f3 = np.log(x1 ** 2 + x2 ** 2 + 1)
        y = -1 / (f1 + f2 + f3 + s)
        return y

@final
class UrsemWavesFunction(TestFunction):
    """
    Ursem Waves function (2D) with a uniform prior.
    """
    def __init__(self):
        dim = 2
        lb = np.array([-0.9, -1.2])
        ub = np.array([1.2, 1.2])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        u = -0.9 * x1 ** 2
        v = (x2 ** 2 - 4.5 * x2 ** 2) * x1 * x2
        w = 4.7 * np.cos(3 * x1 - x2 ** 2 * (2 + x1)) * np.sin(2.5 * np.pi * x1)
        return u + v + w

@final
class HimmelblauFunction(TestFunction):
    """
    Himmelblau function (2D) with a uniform prior.
    """
    def __init__(self):
        dim = 2
        lb = np.array([-6.0, -6.0])
        ub = np.array([6.0, 6.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        y = (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2
        return y

@final
class BukinFunction(TestFunction):
    """
    Bukin function (2D) with a uniform prior.
    """
    def __init__(self):
        dim = 2
        lb = np.array([-15.0, -3.0])
        ub = np.array([-5.0, 3.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        y = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)
        return y

@final
class Hartmann6Function(TestFunction):
    """
    Hartmann 6D function with a uniform prior.
    """
    def __init__(self):
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y

@final
class ForresterFunction(MultiFidelityTestFunction):
    """
    Forrester 1D function (returns both low and high fidelity).
    """
    def __init__(self):
        dim = 1
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        y = (6.0*x - 2.0)**2 * np.sin(12.0*x - 4.0)
        return y[0]

    def evaluate_low(self, x: np.ndarray) -> float:
        x = x.flatten()
        y = 0.5 * self.evaluate_high(x) + 10.0 * (x - 0.5) - 5.0
        return y[0]

@final
class JumpForresterFunction(MultiFidelityTestFunction):
    """
    Jump Forrester function with discontinuities across the input domain.
    """

    def __init__(self):
        dim = 1
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_low(self, x: np.ndarray) -> float:
        x = x.flatten()
        y1 = (6.0 * x - 2.0)**2 * np.sin(12.0 * x - 4.0)
        y2 = y1 + 3.0
        y = (x < 0.5) * y1 + (x > 0.5) * y2
        return y[0]

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        f_L_val = self.evaluate_low(x)
        y1 = 2.0 * f_L_val - 20.0 * x + 20.0
        y2 = y1 + 4.0
        y = (x < 0.5) * y1 + (x > 0.5) * y2
        return y[0]

@final
class HeterogeneousForresterFunction(MultiFidelityTestFunction):
    """
    Heterogeneous Forrester function (2D version of the classic 1D function).
    
    The high-fidelity function only depends on the first input dimension.
    The low-fidelity function is a scaled version of the high-fidelity one.
    """

    def __init__(self):
        dim = 2
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        x1 = x[0]
        return (6.0 * x1 - 2.0)**2 * np.sin(12.0 * x1 - 4.0)

    def evaluate_low(self, x: np.ndarray) -> float:
        return 0.5 * self.evaluate_high(x)

@final
class StepFunction(TestFunction):
    """
    Step function (1D).
    """
    def __init__(self):
        dim = 1
        lb = -np.ones(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        return np.heaviside(x, 0.5)[0]

@final
class MultiFidelityBraninFunction(MultiFidelityTestFunction):
    """
    Multi-fidelity version of the Branin function.
    """

    def __init__(self):
        dim = 2
        lb = np.array([-5.0, 0.0])
        ub = np.array([10.0, 15.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi)
        x1, x2 = x[0], x[1]
        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    def evaluate_low(self, x: np.ndarray) -> float:
        return 0.5 * self.evaluate_high(x) + 10.0 * (np.sum(x) - 0.5) - 5.0

@final
class SinglefidelityBraninFunction(TestFunction):
    """
    Singlefidelity Branin function (2D).
    """
    def __init__(self):
        dim = 2
        lb = np.array([-5.0, 0.0])
        ub = np.array([10.0, 15.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1, x2 = x[0], x[1]
        y = a * (x2 - b*x1**2 + c*x1 -r)**2 + s * (1-t) * np.cos(x1) + s
        return y

@final
class MultiFidelityCamelbackFunction(MultiFidelityTestFunction):
    """
    Multi-fidelity version of the Six-Hump Camelback function.
    """

    def __init__(self):
        dim = 2
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        x1, x2 = x[0], x[1]
        return (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2

    def evaluate_low(self, x: np.ndarray) -> float:
        x = x.flatten()
        # Perturb the input
        x1, x2 = x[0] + 0.1, x[1] - 0.1
        z = (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2
        return 0.75 * self.evaluate_high(x) + 0.375 * z - 0.125

@final
class SinglefidelityCamelbackFunction(TestFunction):
    """
    Singlefidelity Six-Hump Camelback function (2D).
    """
    def __init__(self):
        dim = 2
        lb = np.array([-2.0, -1.0])
        ub = np.array([2.0, 1.0])
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        y = (4.0 - 2.1 * x1 ** 2 + x1 ** 4 / 3.0) * x1 ** 2 + x1 * x2 + (-4.0 + 4.0 * x2 ** 2) * x2 ** 2
        return y

@final
class MultiFidelitySingerCoxFunction(MultiFidelityTestFunction):
    """
    Multi-fidelity version of the Singer-Cox function (4D).

    The high-fidelity function is nonlinear and non-separable.
    The low-fidelity function introduces additional transformations and bias.
    """

    def __init__(self):
        dim = 4
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        y = 0.5 * (np.sqrt(x1**2 + (x2 + x3**2) * x4) - x1) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
        return -y  # maximize y → minimize -y

    def evaluate_low(self, x: np.ndarray) -> float:
        x = x.flatten()
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        high_val = -self.evaluate_high(x)
        y = (1 + np.sin(x1) / 10.0) * high_val - 2 * x1**2 + x2**2 + x3**2 + 0.5
        return -y

@final
class SinglefidelitySingerCoxFunction(TestFunction):
    """
    Singlefidelity Singer-Cox function (4D).
    """
    def __init__(self):
        dim = 4
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        y = 0.5 * (np.sqrt(x1**2 + (x2 + x3**2) * x4) - x1) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
        return -y  # maximize y → minimize -y

@final
class MultiFidelityHartmann3Function(MultiFidelityTestFunction):
    """
    Multi-fidelity Hartmann function in 3D.
    
    The high-fidelity version uses the standard Hartmann formulation.
    The low-fidelity version introduces perturbations to the alpha coefficients.
    """

    def __init__(self):
        dim = 3
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        x = x.flatten()
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [3, 10, 30],
            [0.1, 10, 35],
            [3, 10, 30],
            [0.1, 10, 35]
        ])
        P = 1e-4 * np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ])
        arg = np.sum(A * (x - P) ** 2, axis=1)
        return -np.sum(alpha * np.exp(-arg))

    def evaluate_low(self, x: np.ndarray) -> float:
        x = x.flatten()
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        delta = np.array([0.01, -0.01, -0.1, 0.1])
        alpha2 = alpha + (3 - 2) * delta
        A = np.array([
            [3, 10, 30],
            [0.1, 10, 35],
            [3, 10, 30],
            [0.1, 10, 35]
        ])
        P = 1e-4 * np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ])
        arg = np.sum(A * (x - P) ** 2, axis=1)
        return -np.sum(alpha2 * np.exp(-arg))

@final
class SinglefidelityHartmann3Function(TestFunction):
    """
    Singlefidelity Hartmann 3D function.
    """
    def __init__(self):
        dim = 3
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3, 10, 30],
                        [0.1, 10, 35],
                        [3, 10, 30],
                        [0.1, 10, 35]])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1091, 8732, 5547],
                                [381, 5743, 8828]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y

@final
class MultifidelityHartmann6Function(MultiFidelityTestFunction):
    """
    Multifidelity Hartmann 6D function.
    """
    def __init__(self):
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y

    def evaluate_low(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        delta = np.array([0.01, -0.01, -0.1, 0.1])
        alpha2 = alpha + (3 - 2) * delta
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha2, np.diag(np.exp(-arg)))
        return y

@final
class SinglefidelityHartmann6Function(TestFunction):
    """
    Singlefidelity Hartmann 6D function.
    """
    def __init__(self):
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y

@final
class MultifidelityHartmann6LevelsFunction(MultiFidelityTestFunction):
    """
    Multifidelity Hartmann 6D function with levels.
    """
    def __init__(self):
        dim = 6
        lb = np.zeros(dim)
        ub = np.ones(dim)
        prior = uniform_prior(lb, ub)
        super().__init__(dim, lb, ub, prior)

    def evaluate_high(self, x: np.ndarray) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y

    def evaluate_low(self, x: np.ndarray, dm: float) -> float:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        delta = np.array([0.01, -0.01, -0.1, 0.1])
        alpha2 = alpha + dm * delta
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha2, np.diag(np.exp(-arg)))
        return y

