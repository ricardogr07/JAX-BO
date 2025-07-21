import jax.numpy as jnp
from jaxbo.test_functions import StepFunction, ForresterFunction


def test_step_function():
    f = StepFunction()
    assert f.evaluate(jnp.array([-0.1])) == 0
    assert f.evaluate(jnp.array([0.1])) == 1


def test_forrester_high_low():
    f = ForresterFunction()
    x = jnp.array([0.2])
    high = f.evaluate_high(x)
    low = f.evaluate_low(x)
    assert high != low

