from .particle_filter import *
from .simulate_forward import *

import jax
import jax.numpy as jnp
import polars as pl
import equinox as eqx

class PFVehicle:

    def __init__(self, 
                 #initial_model: eqx.Module,
                 f_from_noise: callable,
                 g_from_noise: callable,
                 state_transition_single_weight: callable,
                 state_observation_single_weight: callable):


        '''Both data generation should be partialed so that we dont handel the function passing inside the class.
        state and observation should be for singles. '''
        
        #self.initial_model = initial_model
        #self.model = initial_model
            
        self.f_from_noise = f_from_noise
        self.g_from_noise = g_from_noise

        self.state_transition_single_weight = jax.jit(jax.vmap(state_transition_single_weight, in_axes=(0, 0, None, None)))
        self.state_observation_single_weight = jax.jit(jax.vmap(state_observation_single_weight, in_axes=(0, 0, None, None)))
    
    def generate_data(self, 
                      key, 
                      N_runs: int, 
                      N_time_steps: int, 
                      starting_point: float
                      ):
        path_key, observation_key = jax.random.split(key, 2)

        # Creating starting points.
        starting_points = jnp.ones(N_runs) * starting_point

        # Define state evolution functions
        def individual_jump_body(carry, noise):
            """Single step of state evolution"""
            next_val = self.f_from_noise(carry, noise)
            return next_val, next_val
    
        def scan_fn(initial_points, jump_noise):
            """Evolve states over multiple time steps"""
            _, hidden_state_evolution = jax.lax.scan(individual_jump_body, initial_points, jump_noise)
            return hidden_state_evolution

        # Generate forward particle paths
        jump_noises = jax.random.normal(path_key, (N_runs, N_time_steps))
        sampled_forward_particles = jax.vmap(scan_fn, in_axes=(0, 0))(
            starting_points, jump_noises
        )

        # Generate observations from the particle paths
        def observation_jump_body(all_particle_array, time_slice):
            """Single step of state evolution"""
            idt, noise = time_slice
            new_y = self.g_from_noise(all_particle_array.at[idt-1].get(), all_particle_array.at[idt].get(), noise)
            return all_particle_array, new_y

        def observation_scan_fn(all_points, jump_noise):
            """Evolve states over multiple time steps"""

            # Expand the all_points array, we double up the first point.
            all_points = jnp.concatenate((jnp.expand_dims(all_points[0], axis=0), all_points))

            # Create the xs vals, which is a tuple of arrays.
            time_slice = jnp.arange(1, all_points.shape[0], dtype=jnp.int32)

            _, hidden_state_evolution = jax.lax.scan(observation_jump_body, all_points, (time_slice, jump_noise))
            return hidden_state_evolution

        observation_noise = jax.random.normal(observation_key, sampled_forward_particles.shape)
        sampled_observation_paths = observation_scan_fn(sampled_forward_particles, observation_noise)

        return sampled_forward_particles, sampled_observation_paths


    def generate_training_data(
            self, 
            key: jax.random.PRNGKey, 
            N_batches: int, 
            batch_size: int,
            starting_point: float
        ):

        all_X_vals, all_Y_vals = self.generate_data(key, N_batches, batch_size, starting_point)
        
        def generate_batch(X_vals, Y_vals):
            prev_X = X_vals[:-1]
            true_X = X_vals[1:]

            Y_vals = Y_vals[1:]

            assert len(prev_X) == len(true_X) == len(Y_vals), "Length mismatch in data preparation"
            
            input_data = jnp.stack([prev_X, Y_vals], axis=-1)  # shape: (batch_size, 2)
            output_data = true_X  # shape: (batch_size,)
            
            return input_data, output_data

        # Vectorize over batches
        batched_data = jax.vmap(generate_batch)(all_X_vals, all_Y_vals)

        return batched_data  # shape: (N_batches, batch_size, 2), (N_batches, batch_size)  



    # def run_from_particle_filter(self,
    #                            key: jax.random.PRNGKey,
    #                            particle_filter: ParticleFilter,
    #                            simulate_from_func: SimulatingForward = None,
    #                            include_raw_paths: bool = False,
    #                            break_after: int = -1,
    #                            verbose: bool = True):
    #     """
    #     Run the complete particle filter pipeline.
        
    #     Args:
    #         key: JAX random key for reproducibility
    #         particle_filter: ParticleFilter instance to use
    #         break_after: Maximum number of segments to process (-1 for all)
    #         verbose: Whether to print progress information
            
    #     Returns:
    #         Tuple of (final_diagnostics, final_fit_metrics)
    #     """
    #     # Split key for separate processing and forward simulation
    #     processing_key, forward_key = jax.random.split(key)
        
    #     # Run particle filtering
    #     diagnostic_from_segment_dict, particle_and_weights_at_flag_idx = processing(
    #         processing_key, 
    #         particle_filter, 
    #         self.pre_processed_data_base, 
    #         self.initial_particles_weights, 
    #         break_after, 
    #         verbose
    #     )
        
    #     # Run forward simulations

    #     if simulate_from_func is None:
    #         simulate_from_func = self.simulate_from_func

    #     raw_path_dict, raw_fit_metrics = simulate_forward_processing(
    #         forward_key, 
    #         simulate_from_func, 
    #         self.pre_processed_data_base, 
    #         particle_and_weights_at_flag_idx, 
    #         self.forecast_negative_increments, 
    #         break_after, 
    #         verbose
    #     )
        
    #     # Apply final processing step
    #     final_diagnostic_from_segment_dict, final_raw_fit_metrics = self.final_step_processing(
    #         diagnostic_from_segment_dict, 
    #         raw_fit_metrics
    #     )
    #     if include_raw_paths:
    #         return final_diagnostic_from_segment_dict, final_raw_fit_metrics, raw_path_dict
    #     else:
    #         return final_diagnostic_from_segment_dict, final_raw_fit_metrics
