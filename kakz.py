import jax
import jax.numpy as jnp

# Define a simple function
def f(x):
  return jnp.sin(x)

# Compute the gradient of the function using JAX's grad function
grad_f = jax.grad(f)

# Evaluate the gradient at x = 1
x = 1.0
print(grad_f(x))