import jax.numpy as jnp
from jax import random
from jaxbo.input_priors import uniform_prior, gaussian_prior


def test_uniform_prior_sample_pdf():
    prior = uniform_prior(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
    key = random.PRNGKey(0)
    samples = prior.sample(key, 5)
    assert samples.shape == (5, 2)
    pdf_vals = prior.pdf(samples)
    assert jnp.all(pdf_vals > 0)


def test_gaussian_prior_sample_pdf():
    prior = gaussian_prior(jnp.zeros(2), jnp.eye(2))
    key = random.PRNGKey(1)
    samples = prior.sample(key, 3)
    assert samples.shape == (3, 2)
    pdf_vals = prior.pdf(samples)
    assert jnp.all(pdf_vals > 0)

