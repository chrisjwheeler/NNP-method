from .particle_filter import ParticleFilter
from .simulate_forward import *

import jax
import optax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


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

    def _generate_paths(self, 
                        key,
                        starting_points, 
                        N_runs,
                        N_time_steps):

        path_key, observation_key = jax.random.split(key, 2)
        
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
    
    def generate_data(self, 
                      key, 
                      N_runs: int, 
                      N_time_steps: int, 
                      starting_point: float
                      ):

        # Creating starting points.
        starting_points = jnp.ones(N_runs) * starting_point

        return self._generate_paths(key, starting_points, N_runs, N_time_steps)

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

    def train_model(
            self, 
            key: jax.random.PRNGKey,
            initial_model: eqx.Module,
            batch_size: int,
            training_params: tuple[int, int] = (1000, 5000),
            testing_params: tuple[int, int] = (100, 2500),
            learning_rate: float = 1e-3,
            steps: int = 1000,
            eval_frequency: int = 100,
            X_bar: float = 0.0,
            verbose: bool = True,
            transition_model: callable = None
            ):


        train_key, data_key, test_data_key = jax.random.split(key, 3)

        input_batches, target_batches = self.generate_training_data(
            data_key, 
            training_params[0], 
            training_params[1],
            X_bar
        )

        inputs = jnp.vstack(input_batches)
        targets = jnp.hstack(target_batches)

        test_input_batches, test_target_batches = self.generate_training_data(
            test_data_key, 
            testing_params[0], 
            testing_params[1],
            X_bar
        )

        test_inputs = jnp.vstack(test_input_batches)
        test_targets = jnp.hstack(test_target_batches)

        model = initial_model

        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        @eqx.filter_value_and_grad
        def loss(model, inputs, z_i):
            pred_mean, pred_std = jax.vmap(model)(inputs)
            log_likelihood = jax.scipy.stats.norm.logpdf(z_i, loc=pred_mean, scale=pred_std)
            return -jnp.mean(log_likelihood)

        @eqx.filter_jit
        def batched_train_step(model, x, y, opt_state, optimizer):
            neg_ll, grads = loss(model, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, neg_ll

        @eqx.filter_jit
        def evaluate(model, x, y):
            return loss(model, x, y)

        losses = []
        test_losses = []
        loop = tqdm(range(steps))

        shuffle_key, train_key = jax.random.split(train_key)
        shuffled_ix = jax.random.permutation(shuffle_key, inputs.shape[0])

        batch_ix = 0
        for step in loop:
            if (batch_ix + 1) * batch_size > inputs.shape[0]:
                shuffle_key, train_key = jax.random.split(train_key)
                shuffled_ix = jax.random.permutation(shuffle_key, inputs.shape[0])
                batch_ix = 0
            
            batch_idx = shuffled_ix[batch_ix * batch_size:(batch_ix + 1) * batch_size]
            batch_ix += 1

            model, opt_state, neg_ll = batched_train_step(
                model, inputs[batch_idx], targets[batch_idx], opt_state, optimizer
            )
            losses.append(neg_ll)
            
            # Evaluate on test set periodically
            if step % eval_frequency == 0:
                test_loss = evaluate(model, test_inputs, test_targets)
                test_losses.append(test_loss)
                loop.set_postfix({
                    'train_loss': f'{neg_ll:.4f}',
                    'test_loss': f'{test_loss[0]:.4f}',
                    'step': step
                })
            else:
                loop.set_postfix({'train_loss': f'{neg_ll:.4f}', 'step': step})

        if verbose:
            plt.figure(figsize=(8, 3))
            plt.plot(losses)
            plt.axhline(test_loss[0], linestyle='--', c='green', label=f'test_loss: {test_loss[0]:.4f}')

            if transition_model is not None:
                transition_loss, _ = loss(transition_model, test_inputs, test_targets)
                plt.axhline(transition_loss, linestyle='--', c='black', label=f'transition_loss: {transition_loss:.4f}')

            plt.xlabel('steps')
            plt.ylabel('neg ll')
            # Set y-limits based on the bottom 20% quantile of the losses
            lower = jnp.quantile(jnp.array(losses), 0.0)
            upper = jnp.quantile(jnp.array(losses), 0.5)
            plt.ylim([lower, upper])
            plt.legend()
            plt.grid()
            plt.show()

        self.has_trained_flag = True
        self.trained_model = jax.vmap(model)
        
        self.losses = losses
        self.test_losses = test_losses
    


    def _simulate_forward(
            self, 
            key: jax.random.PRNGKey,
            initial_particles: jnp.ndarray,
            initial_log_weights: jnp.ndarray,
            Y_array: jnp.ndarray,
            X_array: jnp.ndarray,
            tau: float,
            n_particles: int = 2500
    ):

        # 1. Prepare For the forward simulation. 

        key, sample_key = jax.random.split(key, 2)

        # Sample starting points from initial particle distribution
        starting_point_indices = jax.random.choice(
            sample_key, initial_particles.shape[0], (n_particles,), 
            p=jnp.exp(initial_log_weights)
        )
        starting_points = initial_particles[starting_point_indices]

        # 2. Get the paths. 

        sampled_forward_particles, sampled_observation_paths = self._generate_paths(
            key,
            starting_points,
            n_particles,
            Y_array.shape[0]
        )

        # 3. Calculate metrics on the paths. 
        forecast_metrics = {}

        # Calculate realized volatility with time normalization
        tau_normalized_constant = 1 / (Y_array.shape[0] * tau)
        target_realized_vol = tau_normalized_constant * jnp.sum(Y_array**2)
        sampled_realized_vol = tau_normalized_constant * jnp.sum(sampled_observation_paths**2, axis=1)

        # Estimate likelihood using kernel density estimation
        kde = jsp.stats.gaussian_kde(sampled_realized_vol, bw_method="scott")
        forecast_metrics['likelihood'] = kde.logpdf(target_realized_vol)

        kde_mse = jnp.mean((sampled_realized_vol - target_realized_vol)**2)
        forecast_metrics['kde_mse'] = kde_mse

        # Calculating the mean squared error. 
        total_mse = jnp.mean((sampled_forward_particles - X_array)**2)
        forecast_metrics['total_mse'] = total_mse

        # Calculating coverage
        lower_quantile = jnp.quantile(sampled_forward_particles, 0.2, axis=0)
        upper_quantile = jnp.quantile(sampled_forward_particles, 0.8, axis=0)
        forecast_metrics['total_coverage'] = jnp.mean((X_array > lower_quantile) & (X_array < upper_quantile))

        # 4. 

        return (sampled_forward_particles, sampled_observation_paths), forecast_metrics

    def run_from_particle_filter(
        self,
        key: jax.random.PRNGKey,
        particle_filter: ParticleFilter,
        Y_array: jnp.ndarray,
        X_array: jnp.ndarray,
        initial_particles: jnp.ndarray,
        simulate_at: ArrayLike,
        tau: float,
        verbose: bool = True
    ):
        """
        Runs a particle filter in segments, simulates forward, and collects diagnostics.

        Args:
            key: JAX random key.
            particle_filter: ParticleFilter instance.
            data_tuple: Tuple of (X_array, Y_array).
            initial_particles: Initial particle states.
            simulate_at: List/array of fractions (0-1) indicating forecast points.
            tau: Time normalization constant.
            verbose: If True, show progress bar.

        Returns:
            merged_diagnostics: Dict of concatenated diagnostics.
            particle_and_weights_at_flag_idx: Dict of (particles, weights) at each segment.
            total_forecast_metrics: Dict of forward simulation metrics at each segment.
        """
        adjusted_prediction_length = X_array.shape[0] - 1
        forecast_at = [int(f_ratio * adjusted_prediction_length) for f_ratio in simulate_at]

        output_diagnostics = (
            particle_filter.category_two_metric_names + particle_filter.category_one_metric_names
        )
        merged_diagnostics = {k: jnp.array([]) for k in output_diagnostics}
        diagnostic_from_segment_dict = {}
        particle_and_weights_at_flag_idx = {}
        total_forecast_metrics = {}

        start_idx = 0
        last_particles = initial_particles
        last_weights = -jnp.log(particle_filter.N_PARTICLES) * jnp.ones_like(initial_particles)

        pbar = tqdm(forecast_at, desc="Processing segments", unit="segment") if verbose else forecast_at

        for forecast_idx in pbar:
            key, forward_key, pf_step_key = jax.random.split(key, 3)

            # Particle filter segment
            pf_Y_segment = Y_array[start_idx:forecast_idx]
            pf_X_segment = X_array[start_idx:forecast_idx]

            out_particles, out_weights, diagnostics = particle_filter.simulate(
                pf_step_key, last_particles, last_weights, pf_Y_segment, pf_X_segment
            )

            diagnostic_from_segment_dict[forecast_idx] = diagnostics
            particle_and_weights_at_flag_idx[forecast_idx] = (out_particles, out_weights)
            merged_diagnostics = {
                k: jnp.concatenate([merged_diagnostics[k], diagnostics[k]])
                for k in output_diagnostics
            }

            last_particles, last_weights = out_particles, out_weights

            # Forward simulation
            pf_Y_forward = Y_array[forecast_idx + 1 :]
            pf_X_forward = X_array[forecast_idx + 1 :]
            (sampled_forward_particles, sampled_observation_paths), forecast_metric_dict = self._simulate_forward(
                forward_key, out_particles, out_weights, pf_Y_forward, pf_X_forward, tau
            )

            merged_diagnostics[f"forward_sim_{forecast_idx}"] = (sampled_forward_particles, sampled_observation_paths)
            total_forecast_metrics[forecast_idx] = forecast_metric_dict

            start_idx = forecast_idx + 1

        # Final segment
        key, final_pf_key = jax.random.split(key, 2)
        pf_Y_segment = Y_array[start_idx:]
        pf_X_segment = X_array[start_idx:]

        out_particles, out_weights, diagnostics = particle_filter.simulate(
            final_pf_key, last_particles, last_weights, pf_Y_segment, pf_X_segment
        )

        diagnostic_from_segment_dict["final"] = diagnostics
        particle_and_weights_at_flag_idx["final"] = (out_particles, out_weights)
        
        merged_diagnostics = {
            k: jnp.concatenate([merged_diagnostics[k], diagnostics[k]])
            for k in output_diagnostics
        }

        return merged_diagnostics, particle_and_weights_at_flag_idx, total_forecast_metrics
