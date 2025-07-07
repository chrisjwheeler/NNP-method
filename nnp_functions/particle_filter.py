from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import equinox as eqx


## ChatGPT's weighted quantile implementation.

def weighted_quantile(values, quantiles, sample_weight=None, axis=None):
    """
    Compute the weighted quantiles of `values` along the specified axis.
    If axis is None, the input is flattened.

    Args:
        values: Input data.
        quantiles: Float or array of floats in [0, 1].
        sample_weight: Same shape as `values`, or broadcastable.
        axis: Axis to compute quantile along.
    
    Returns:
        Weighted quantiles.
    """
    values = jnp.asarray(values)
    quantiles = jnp.atleast_1d(quantiles)

    if sample_weight is None:
        sample_weight = jnp.ones_like(values)
    else:
        sample_weight = jnp.asarray(sample_weight)

    # If axis is None, flatten inputs
    if axis is None:
        values = values.flatten()
        sample_weight = sample_weight.flatten()
        axis = 0

    # Sort values and weights along the given axis
    sorted_indices = jnp.argsort(values, axis=axis)
    sorted_values = jnp.take_along_axis(values, sorted_indices, axis=axis)
    sorted_weights = jnp.take_along_axis(sample_weight, sorted_indices, axis=axis)

    # Compute normalized cumulative weights
    cum_weights = jnp.cumsum(sorted_weights, axis=axis)
    total_weight = jnp.sum(sorted_weights, axis=axis, keepdims=True)
    norm_cum_weights = cum_weights / total_weight

    # Interpolate for each quantile
    def interp(q):
        mask = norm_cum_weights >= q
        idx = jnp.argmax(mask, axis=axis)

        def get_value(i):
            expanded_idx = jnp.expand_dims(i, axis=axis)
            taken = jnp.take_along_axis(sorted_values, expanded_idx, axis=axis)
            return jnp.squeeze(taken, axis=axis)

        return get_value(idx)

    return jax.vmap(interp)(quantiles) if quantiles.shape[0] > 1 else interp(quantiles[0])


@dataclass(frozen=True)
class ParticleFilter:

    # Constants
    sample_fn: callable
    weight_fn: callable

    final_reweight: callable = lambda *args: args

    ESS_COND: float = 0.5
    N_PARTICLES: int = 2500
    Y_LOOK_FORWARD: int = 0
    switch_resampling_in_step: bool = False
    needs_final_reweight: bool = False

    def __post_init__(self):

        object.__setattr__(self, 'category_two_metric_names', ['ess', 'resample_flag', 'normalised_entropy', 'marginal_likelihood'])
        object.__setattr__(self, 'category_one_metric_names', ['mean_squared_error', 'tail_coverage'])

    @staticmethod    
    def multinomial_resample(subkey, weights, N_particles):
        """
        Perform multinomial resampling of particles based on their weights.
        
        Args:
            subkey: JAX random key for sampling
            weights: Array of particle weights (must sum to 1)
            N_particles: Number of particles to resample
            
        Returns:
            tuple: (resample_indices, log_weights, resample_flag)
                - resample_indices: Indices of resampled particles
                - log_weights: Log weights after resampling (uniform)
                - resample_flag: 1 if resampling occurred, 0 otherwise
        """
        resample_indices = jax.random.choice(
            subkey, N_particles, p=weights, shape=(N_particles,)
        )
        log_weights = jnp.log(1 / N_particles) * jnp.ones_like(weights)
        return resample_indices, log_weights, 1
   
    @staticmethod
    def calculate_ess(log_weights):
        """
        Calculate the effective sample size (ESS) of the particle weights.
        
        Args:
            log_weights: Log weights of particles
            
        Returns:
            float: Effective sample size
        """
        return 1.0 / jnp.exp(jax.scipy.special.logsumexp(2 * log_weights))
    
    @staticmethod
    def get_category_one_metrics(X_val, sampled_particles, log_weights, unormalised_log_weights, n_particles):

        true_scale_weights = jnp.exp(log_weights)
        
        mean_squared_error = (X_val - jnp.average(sampled_particles, weights=true_scale_weights))**2
        
        q5 = weighted_quantile(sampled_particles, 0.05, sample_weight=true_scale_weights)
        q95 = weighted_quantile(sampled_particles, 0.95, sample_weight=true_scale_weights)
        tail_coverage = (X_val > q5) * (X_val < q95)
        
        return mean_squared_error, tail_coverage

    @staticmethod
    def get_category_two_metrics(log_weights, unormalised_log_weights, n_particles):
        """
        Calculate online diagnostic metrics for the particle filter.
        
        Args:
            log_weights: Current log weights of particles
            unormalised_log_weights: Unnormalized log weights before normalization
            n_particles: Number of particles
            
        Returns:
            tuple: (entropy, marginal_likelihood)
                - entropy: Normalized entropy of the particle weights
                - marginal_likelihood: Marginal likelihood estimate
        """
        entropy = jnp.sum(jnp.exp(log_weights) * log_weights) + jnp.log(n_particles)
        marginal_likelihood = jnp.sum(jnp.exp(unormalised_log_weights))
        
        return entropy, marginal_likelihood

    @eqx.filter_jit
    def simulate(
            self,
            key: jax.random.PRNGKey,
            initial_particles: jnp.ndarray, 
            initial_log_weights: jnp.ndarray,
            Y_array: jnp.ndarray,
            X_array: jnp.ndarray
    ):
        """
        Run the particle filter simulation over the entire time series.
        
        This method implements a sequential Monte Carlo particle filter that can
        switch between different proposal distributions and weight functions based
        on the tau_id_array. It performs resampling when the effective sample size
        falls below the threshold.
        
        Args:
            key: JAX random key for the simulation
            initial_particles: Initial particle states
            initial_log_weights: Initial log weights of particles
            Y_array: Array of observations
            tau_id_array: Array of time period identifiers for each time step
            
        Returns:
            tuple: (final_particles, final_log_weights, filter_diagnostics)
                - final_particles: Final particle states
                - final_log_weights: Final log weights
                - filter_diagnostics: Dictionary containing diagnostic metrics
                    (ess, resample_flag, normalised_entropy, marginal_likelihood)
                    
        Raises:
            AssertionError: If Y_array shape doesn't match tau_id_array shape + Y_LOOK_FORWARD
            NotImplementedError: If final reweighting is requested but not implemented
        """
        

        def particle_filter_step(carry, time_slice):
            idt, true_X_val = time_slice
            key, Y_array, particles, log_weights = carry
            
            if self.switch_resampling_in_step:
                key, subkey = jax.random.split(key)

                ess = self.calculate_ess(log_weights)
                particle_indices, log_weights, resample_flag = jax.lax.cond(
                    ess/self.N_PARTICLES < self.ESS_COND,
                    lambda k, log_w: self.multinomial_resample(k, jnp.exp(log_w), self.N_PARTICLES),
                    lambda _, log_w: (jnp.arange(self.N_PARTICLES), log_w, 0),
                    *(subkey, log_weights)
                )
                
                particles = sampled_particles[particle_indices]
           

            # 2. Classic particle filter logic

             # Sample new particles using the proposal distribution
            key, subkey = jax.random.split(key)
            sampled_particles = self.sample_fn(subkey, particles, Y_array, idt)

            # Update particle weights
            weight_update = self.weight_fn(sampled_particles, particles, Y_array, idt)
            unormalised_log_weights = log_weights + weight_update
            log_weights = unormalised_log_weights - jsp.special.logsumexp(unormalised_log_weights)

            category_two_metric_args = self.get_category_two_metrics(log_weights, unormalised_log_weights, self.N_PARTICLES)

            # IF IMPLEMENTING AUX: THIS SHOULD HAPPEN BEFORE RESAMPLING
            category_one_metric_args = self.get_category_one_metrics(true_X_val, sampled_particles, log_weights, unormalised_log_weights, self.N_PARTICLES)

            if not self.switch_resampling_in_step:
                key, subkey = jax.random.split(key)

                ess = self.calculate_ess(log_weights)
                particle_indices, log_weights, resample_flag = jax.lax.cond(
                    ess/self.N_PARTICLES < self.ESS_COND,
                    lambda k, log_w: self.multinomial_resample(k, jnp.exp(log_w), self.N_PARTICLES),
                    lambda _, log_w: (jnp.arange(self.N_PARTICLES), log_w, 0),
                    *(subkey, log_weights)
                )
                particles = sampled_particles[particle_indices]

            return (key, Y_array, particles, log_weights), (ess, resample_flag, *category_two_metric_args, *category_one_metric_args)
        
        # 1. Run scan.

        Y_idt = jnp.arange(Y_array.shape[0])

        # Run filter
        final_carry, online_metric_list = jax.lax.scan(
            particle_filter_step, 
            (key, Y_array, initial_particles, initial_log_weights),
            (Y_idt, X_array)
        )

        # 2. Process output of scan.
        _, _, final_particles, final_log_weights = final_carry
        filter_diagnostics = {diagnostic_name: val for diagnostic_name, val in zip(self.category_two_metric_names + self.category_one_metric_names, online_metric_list)}

        # AUX final reweight
        if self.needs_final_reweight:
            raise NotImplementedError("Final reweighting not implemented")
            # We are going to have issues as we need the previous particles, which we dont have without writing them into scan or terminating one early.
            # When re-writing terminate one step early.

        return final_particles, final_log_weights, filter_diagnostics
