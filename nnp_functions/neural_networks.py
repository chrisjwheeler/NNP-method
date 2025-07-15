import jax
import jax.numpy as jnp
import equinox as eqx

class FeedForwardNetwork(eqx.Module):
    """
    Feedforward neural network for predicting the mean and standard deviation of a normal distribution.

    Architecture:
        - Input: 2D vector (X_{t-1}, R_t)
        - Several hidden layers with ReLU activation
        - Output: 2D vector (mean, log_std)

    Output:
        - mean: Predicted mean of the normal distribution (scalar)
        - std: Predicted standard deviation (scalar, enforced positive via softplus + epsilon)
    """
    layers: list[eqx.nn.Linear]

    def __init__(self, n_layers: int, hidden_dim: int, key: jax.random.PRNGKey):
        subkeys = jax.random.split(key, n_layers + 1)
        layers = [eqx.nn.Linear(2, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 2, key=subkeys[-1]))
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass through the network.

        Args:
            x: jax.Array of shape (2,)

        Returns:
            mean: Predicted mean (scalar)
            std: Predicted standard deviation (scalar, >0)
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        mean, log_std = x[0], x[1]
        std = jax.nn.softplus(log_std) + 1e-24
        return mean, std
    
class t_FeedForwardNetwork(eqx.Module):
    """
    Feedforward neural network for predicting the mean, standard deviation, and degrees of freedom of a Student's t-distribution.

    Architecture:
        - Input: 2D vector (X_{t-1}, R_t)
        - Several hidden layers with ReLU activation
        - Output: 3D vector (mean, log_std, log_df)

    Output:
        - mean: Predicted mean (scalar)
        - std: Predicted standard deviation (scalar, enforced positive via softplus + epsilon)
        - def_freedom: Predicted degrees of freedom (scalar, enforced positive via softplus + epsilon)
    """
    layers: list[eqx.nn.Linear]

    def __init__(self, n_layers: int, hidden_dim: int, key: jax.random.PRNGKey):
        subkeys = jax.random.split(key, n_layers + 1)
        layers = [eqx.nn.Linear(2, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 3, key=subkeys[-1]))
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass through the network.

        Args:
            x: jax.Array of shape (2,)

        Returns:
            mean: Predicted mean (scalar)
            std: Predicted standard deviation (scalar, >0)
            def_freedom: Predicted degrees of freedom (scalar, >0)
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        mean, log_std, def_freedom = x[0], x[1], x[2]
        std = jax.nn.softplus(log_std) + 1e-24
        def_freedom = jax.nn.softplus(def_freedom) + 1e-24
        return mean, std, def_freedom

class MixFeedForwardNetwork(eqx.Module):
    """
    Feedforward neural network for predicting the parameters of a mixture of normal distributions.

    Architecture:
        - Input: 2D vector (X_{t-1}, R_t)
        - Several hidden layers with ReLU activation
        - Output: 3 * n_mix vector (means, log_stds, logits for mixture weights)

    Output:
        - mean: Predicted means for each mixture component (jax.Array of shape (n_mix,))
        - std: Predicted standard deviations for each component (jax.Array of shape (n_mix,), >0)
        - weights: Mixture weights (jax.Array of shape (n_mix,), sum to 1 via softmax)
    """
    layers: list[eqx.nn.Linear]
    n_mix: int

    def __init__(self, n_layers: int, hidden_dim: int, n_mix: int, key: jax.random.PRNGKey):
        self.n_mix = n_mix
        subkeys = jax.random.split(key, n_layers + 1)
        
        layers = [eqx.nn.Linear(2, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 3 * n_mix, key=subkeys[-1]))
        
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass through the network.

        Args:
            x: jax.Array of shape (2,)

        Returns:
            mean: Predicted means for each mixture component (jax.Array, shape (n_mix,))
            std_val: Predicted standard deviations for each component (jax.Array, shape (n_mix,), >0)
            weights: Mixture weights (jax.Array, shape (n_mix,), sum to 1)
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        mean, log_std, weights = x[0:self.n_mix], x[self.n_mix:2*self.n_mix], x[2*self.n_mix:]
        
        std_val = jax.nn.softplus(log_std) + 1e-24
        weights = jax.nn.softmax(weights)
        
        return mean, std_val, weights

class MixMultiChannelNetwork(eqx.Module):
    """
    Feedforward neural network for predicting the parameters of a mixture of normal distributions.

    Architecture:
        - Input: 2D vector (X_{t-1}, R_t)
        - Several hidden layers with ReLU activation
        - Output: 3 * n_mix vector (means, log_stds, logits for mixture weights)

    Output:
        - mean: Predicted means for each mixture component (jax.Array of shape (n_mix,))
        - std: Predicted standard deviations for each component (jax.Array of shape (n_mix,), >0)
        - weights: Mixture weights (jax.Array of shape (n_mix,), sum to 1 via softmax)
    """
    layer_tuple: tuple[list[eqx.nn.Linear]]
    n_mix: int

    def __init__(self, n_layers: int, hidden_dim: int, n_mix: int, key: jax.random.PRNGKey):
        self.n_mix = n_mix
        
        # Creating the multiple channels.
        total_layers_list = []
        for _ in range(3):
            layer_list = []
            subkeys = jax.random.split(key, n_layers + 2)
            key = subkeys[0]
            
            layer_list.append(eqx.nn.Linear(2, hidden_dim, key=subkeys[1]))
            layer_list.extend([eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)])
            layer_list.append(eqx.nn.Linear(hidden_dim, n_mix, key=subkeys[-1]))

            total_layers_list.append(layer_list)

        self.layer_tuple = tuple(total_layers_list)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass through the network.

        Args:
            x: jax.Array of shape (2,)

        Returns:
            mean: Predicted means for each mixture component (jax.Array, shape (n_mix,))
            std_val: Predicted standard deviations for each component (jax.Array, shape (n_mix,), >0)
            weights: Mixture weights (jax.Array, shape (n_mix,), sum to 1)
        """
        
        inital_x = x
        
        final_outputs = []
        for channel_idx in range(3):
            x = inital_x
            for layer in self.layer_tuple[channel_idx][:-1]:
                x = jax.nn.relu(layer(x))
            final_outputs.append(self.layer_tuple[channel_idx][-1](x))
       
        mean, log_std, weights = final_outputs[0], final_outputs[1], final_outputs[2]
        std_val = jax.nn.softplus(log_std) + 1e-24
        weights = jax.nn.softmax(weights)
        
        return mean, std_val, weights
    
class MixGivFeedForwardNetwork(eqx.Module):
    """
    Feedforward neural network for predicting the parameters of a mixture of normal distributions.

    Architecture:
        - Input: 2D vector (X_{t-1}, R_t)
        - Several hidden layers with ReLU activation
        - Output: 3 * n_mix vector (means, log_stds, logits for mixture weights)

    Output:
        - mean: Predicted means for each mixture component (jax.Array of shape (n_mix,))
        - std: Predicted standard deviations for each component (jax.Array of shape (n_mix,), >0)
        - weights: Mixture weights (jax.Array of shape (n_mix,), sum to 1 via softmax)
    """
    layers: list[eqx.nn.Linear]
    n_mix: int
    giv_func: callable

    def __init__(self, n_layers: int, hidden_dim: int, n_mix: int, giv_func: callable, key: jax.random.PRNGKey):
        self.n_mix = n_mix
        self.giv_func = giv_func

        subkeys = jax.random.split(key, n_layers + 1)
        
        layers = [eqx.nn.Linear(4, hidden_dim, key=subkeys[0])]
        layers += [eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[i]) for i in range(1, n_layers)]
        layers.append(eqx.nn.Linear(hidden_dim, 3 * n_mix, key=subkeys[-1]))
        
        self.layers = layers

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass through the network.

        Args:
            x: jax.Array of shape (2,)

        Returns:
            mean: Predicted means for each mixture component (jax.Array, shape (n_mix,))
            std_val: Predicted standard deviations for each component (jax.Array, shape (n_mix,), >0)
            weights: Mixture weights (jax.Array, shape (n_mix,), sum to 1)
        """
        trans_x = self.giv_func(x)
        x = jnp.concatenate([x, trans_x], axis=0)
        
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        
        x = self.layers[-1](x)
        mean, log_std, weights = x[0:self.n_mix], x[self.n_mix:2*self.n_mix], x[2*self.n_mix:]
        
        std_val = jax.nn.softplus(log_std) + 1e-24
        weights = jax.nn.softmax(weights)
        
        return mean, std_val, weights