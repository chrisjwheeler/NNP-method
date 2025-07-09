from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx


@dataclass(frozen=True)
class SimulatingForward:
    """
    A class for simulating forward particle paths and evaluating their likelihood.s
    
    This class implements forward simulation of particle paths from initial particle states
    and evaluates the likelihood of observed data using kernel density estimation on
    realized volatility metrics.
    
    Attributes:
        min_f_from_noise: Function to evolve state for minute-level periods
        overnight_f_from_noise: Function to evolve state for overnight periods  
        observation_from_noise: Function to generate observations from state and noise
        weekend_f_from_noise: Function to evolve state for weekend periods (defaults to overnight)
        other_f_from_noise: Function to evolve state for other periods (defaults to weekend)
        N_SAMPLES: Number of forward simulation samples (default: 2500)
    """
    min_f_from_noise: callable
    overnight_f_from_noise: callable
    observation_from_noise: callable
    weekend_f_from_noise: callable = None
    other_f_from_noise: callable = None
    N_SAMPLES: int = 2500

    def __post_init__(self):
        """
        Post-initialization setup for the forward simulator.
        
        Sets default functions for weekend and other periods if not provided.
        """
        # Workaround for frozen=True: use object.__setattr__
        if self.weekend_f_from_noise is None:
            object.__setattr__(self, 'weekend_f_from_noise', self.overnight_f_from_noise)
        if self.other_f_from_noise is None:
            object.__setattr__(self, 'other_f_from_noise', self.weekend_f_from_noise)

    def evaluation_metrics_from_paths(self, sampled_forward_particles, sampled_observation_paths, 
                                    Y_array, tau_id_array, total_tau_time):
        """
        Calculate likelihood of observed data given simulated forward paths.
        
        Uses kernel density estimation on realized volatility to estimate the likelihood
        of the target realized volatility given the distribution of simulated realized volatilities.
        
        Args:
            sampled_forward_particles: Simulated particle states
            sampled_observation_paths: Simulated observation paths
            Y_array: Target observation array
            tau_id_array: Time period identifiers
            total_tau_time: Total time duration in minutes
            
        Returns:
            float: Estimated likelihood of target realized volatility
        """
        # Calculate realized volatility with time normalization
        tau_normalized_constant = 1 / total_tau_time
        target_realized_vol = tau_normalized_constant * jnp.sum(Y_array**2)
        sampled_realized_vol = tau_normalized_constant * jnp.sum(sampled_observation_paths**2, axis=1)

        # Estimate likelihood using kernel density estimation
        kde = jsp.stats.gaussian_kde(sampled_realized_vol, bw_method="scott")
        likelihood = kde.logpdf(target_realized_vol)
    
        return likelihood

    def combined_f_from_noise(self, tau_id, last_val, noise):
        """
        Select and apply the appropriate state evolution function based on tau_id.
        
        Args:
            tau_id: Time period identifier (0=minute, 1=overnight, 2=weekend, 3=other)
            last_val: Previous state value
            noise: Random noise for state evolution
            
        Returns:
            array: New state value
        """
        return jax.lax.switch(
            tau_id,
            [
                lambda: self.min_f_from_noise(last_val, noise),
                lambda: self.overnight_f_from_noise(last_val, noise),
                lambda: self.weekend_f_from_noise(last_val, noise),
                lambda: self.other_f_from_noise(last_val, noise),
            ]
        )

    @eqx.filter_jit
    def simulate_forward(self, key: jax.random.PRNGKey, initial_particles: jnp.ndarray,
                        initial_log_weights: jnp.ndarray, Y_array: jnp.ndarray,
                        tau_id_array: jnp.ndarray, total_tau_time: float):
        """
        Simulate forward particle paths and evaluate their likelihood.
        
        This method samples particles from the initial distribution, simulates their
        forward evolution, generates corresponding observations, and evaluates the
        likelihood of the target observations using realized volatility metrics.
        
        Args:
            key: JAX random key for the simulation
            initial_particles: Initial particle states
            initial_log_weights: Initial log weights of particles
            Y_array: Target observation array
            tau_id_array: Time period identifiers for each time step
            total_tau_time: Total time duration in minutes for normalization
            
        Returns:
            tuple: ((sampled_forward_particles, sampled_observation_paths), likelihood)
                - sampled_forward_particles: Simulated particle state paths
                - sampled_observation_paths: Simulated observation paths
                - likelihood: Estimated likelihood of target observations
        """
        sample_key, path_key, observation_key = jax.random.split(key, 3)

        # Sample starting points from initial particle distribution
        starting_point_indices = jax.random.choice(
            sample_key, initial_particles.shape[0], (self.N_SAMPLES,), 
            p=jnp.exp(initial_log_weights)
        )
        starting_points = initial_particles[starting_point_indices]

        # Define state evolution functions
        def individual_jump_body(carry, time_slice):
            """Single step of state evolution"""
            tau_id, noise = time_slice
            next_val = self.combined_f_from_noise(tau_id, carry, noise)
            return next_val, next_val
    
        def scan_fn(initial_points, tau_id_array, jump_noise):
            """Evolve states over multiple time steps"""
            _, hidden_state_evolution = jax.lax.scan(individual_jump_body, initial_points, (tau_id_array, jump_noise))
            return hidden_state_evolution

        # Generate forward particle paths
        jump_noises = jax.random.normal(path_key, (self.N_SAMPLES, Y_array.shape[0]))
        sampled_forward_particles = jax.vmap(scan_fn, in_axes=(0, None, 0))(
            starting_points, tau_id_array, jump_noises
        )

        # Generate observations from the particle paths
        def observation_jump_body(all_particle_array, time_slice):
            """Single step of state evolution"""
            idt, noise = time_slice
            new_y = self.observation_from_noise(all_particle_array.at[idt-1].get(), all_particle_array.at[idt].get(), noise)
            return all_particle_array, new_y

        def observation_scan_fn(all_points, jump_noise):
            """Evolve states over multiple time steps"""

            # Expand the all_points array.
            all_points = jnp.concatenate((jnp.expand_dims(all_points[0], axis=0), all_points))

            # Create the xs vals, which is a tuple of arrays.
            time_slice = jnp.arange(1, all_points.shape[0], dtype=jnp.int32)

            _, hidden_state_evolution = jax.lax.scan(observation_jump_body, all_points, (time_slice, jump_noise))
            return hidden_state_evolution

        observation_noise = jax.random.normal(observation_key, sampled_forward_particles.shape)
        sampled_observation_paths = observation_scan_fn(sampled_forward_particles, observation_noise)

        # Calculate evaluation metrics
        result_metrics = self.evaluation_metrics_from_paths(
            sampled_forward_particles, sampled_observation_paths, 
            Y_array, tau_id_array, total_tau_time
        )

        return (sampled_forward_particles, sampled_observation_paths), result_metrics
