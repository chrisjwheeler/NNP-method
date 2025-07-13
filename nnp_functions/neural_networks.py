import jax
import equinox as eqx

class FeedForwardNetwork(eqx.Module):
    """Feedforward neural network predicting mean and std of a normal distribution.

    Input: 2D (X_{t-1}, R_t)
    Output: mean, std (std > 0 via softplus)
    """
    layers: list[eqx.nn.Linear]

    def __init__(self, n_layers: int, hidden_dim: int, key: jax.random.PRNGKey):
        subkeys = jax.random.split(key, n_layers + 1)
        layers = [eqx.nn.Linear(2, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 2, key=subkeys[-1]))
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        mean, log_std = x[0], x[1]
        std = jax.nn.softplus(log_std) + 1e-24
        return mean, std
    
class t_FeedForwardNetwork(eqx.Module):
    """Feedforward neural network predicting mean and std of a normal distribution.

    Input: 2D (X_{t-1}, R_t)
    Output: mean, std (std > 0 via softplus)
    """
    layers: list[eqx.nn.Linear]

    def __init__(self, n_layers: int, hidden_dim: int, key: jax.random.PRNGKey):
        subkeys = jax.random.split(key, n_layers + 1)
        layers = [eqx.nn.Linear(2, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 3, key=subkeys[-1]))
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        mean, log_std, def_freedom = x[0], x[1], x[2]
        std = jax.nn.softplus(log_std) + 1e-24
        def_freedom = jax.nn.softplus(def_freedom) + 1e-24
        return mean, std, def_freedom
