import itertools
import time

import numpy as np
import numba as nb

from model.core.model import Model
from model.core.signal import InputSignal
from service.behavioral_processing import BehavioralProcessing
from service.model_service import ModelService
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel, MAX_SCORE


class DDMstable(Model):
    """
    Drift-diffusion model with leaky integration, a refractory ("inactive")
    period, and a partial reset after each bout. Subclasses Model to plug
    into the shared fitting/scoring infrastructure. The integrator
    accumulates noisy drift driven by the stimulus until it crosses a
    threshold, emitting a discrete "bout" (decision event), then resets
    partially and enters a refractory period before it can fire again.
    """
    model_label = 'drift_diffusion_Dkl*'
    # Per-trial bout count bounds, later scaled by trials_per_simulation.
    min_bout_per_simulation = 1
    max_bout_per_simulation = np.inf
    max_response_time_allowed = 0.5
    time_experimental_trial = 30  # seconds

    def __init__(self, parameters, trials_per_simulation=50,
                 scaling_factor_input=100, time_experimental_trial=None,
                 size_loss_memory=1, fitting_resolution=40, dt=None,
                 multiple_individuals=False, analysed_parameter=StimulusParameterLabel.COHERENCE.value,
                 smooth_loss=False, noise_sigma_sensitivity=0):
        # Base Model class handles parameter storage/validation.
        super(DDMstable, self).__init__(parameters)

        self.trials_per_simulation = trials_per_simulation
        # Scale class-level per-trial bout bounds to the full simulation.
        self.min_bout_per_simulation *= trials_per_simulation
        self.max_bout_per_simulation *= trials_per_simulation
        # Rescales raw stimulus values (e.g. % coherence) to drift-rate units.
        self.scaling_factor_input = scaling_factor_input
        # Number of past scores averaged to smooth the fitting objective.
        self.size_loss_memory = size_loss_memory
        # Bin count for discretizing response-time distributions in the loss.
        self.fitting_resolution = fitting_resolution
        self.multiple_individuals = multiple_individuals
        self.analysed_parameter = analysed_parameter
        self.smooth_loss = smooth_loss
        # Std-dev of an optional loss perturbation, used for sensitivity analysis.
        self.noise_sigma_sensitivity = noise_sigma_sensitivity
        if time_experimental_trial is not None:
            self.time_experimental_trial = time_experimental_trial
        if dt is not None:
            self.dt = dt
        self.resolution_distribution = None  # just initialize blank attribute, to populate when fit is prepared
        self.perturbation = None  # just initialize blank attribute, to update if needed when fit is prepared

    def prepare_for_fitting(self, input_signal=None, focus_scope=None, options=None):
        # Histogram resolution for the RT distribution, derived from the
        # focus-window width at a fixed 5-timesteps-per-bin granularity.
        if focus_scope is not None:
            self.resolution_distribution = int((focus_scope[1] - focus_scope[0]) / (self.parameters.dt.value * 5))
        # Only "fit"/"constant" signals iterate over discrete condition keys.
        if isinstance(input_signal, InputSignal) and input_signal.label in ("fit", "constant"):
            if self.noise_sigma_sensitivity:
                # Draw one fixed random perturbation per condition (correct/error),
                # reused across evaluations for a controlled sensitivity analysis.
                self.perturbation = {}
                for key in input_signal.value.keys():
                    self.perturbation[key] = {"corr": np.random.normal(scale=self.noise_sigma_sensitivity, size=self.resolution_distribution),
                                              "err": np.random.normal(scale=self.noise_sigma_sensitivity, size=self.resolution_distribution)}
            else:
                # No sensitivity analysis: perturbations are zero (no-op).
                self.perturbation = {}
                for key in input_signal.value.keys():
                    self.perturbation[key] = {"corr": np.zeros(self.resolution_distribution),
                                              "err": np.zeros(self.resolution_distribution)}

    def simulate(self, input_signal=None, dt=None, number_trials=None):
        # Track wall-clock time for the debug print at the end.
        time0 = time.time_ns()
        if number_trials is None:
            number_trials = self.trials_per_simulation
        if input_signal is not None:
            self.input_signal = input_signal
        # Per-trial containers, used only in the default (else) branch below.
        response_time_list = [[] for i in range(number_trials)]
        bout_decision_list = [[] for i in range(number_trials)]
        time_list = [[] for i in range(number_trials)]

        output = {}
        if self.input_signal.label == "fit":
            # Multiple named conditions (e.g. one per coherence level),
            # each simulated separately and stored under its own key.
            for key in self.input_signal.value.keys():
                if ModelService.is_window(key):
                    # Window conditions hold a constant value over a fixed
                    # time span, repeated across trials_per_simulation.
                    duration_input = (self.input_signal.value[key]["time_end"] - self.input_signal.value[key]["time_start"]) * self.input_signal.value[key]["trials_per_simulation"]
                    response_time_list, bout_decision_list, time_list, _ = self.simulate_trial_constant(
                        self.input_signal.value[key]["value"],
                        duration_input,
                        dt)
                else:
                    response_time_list, bout_decision_list, time_list, _ = self.simulate_trial(self.input_signal.value[key], dt)
                output[key] = {
                    "response_time_list": response_time_list,
                    "bout_decision_list": bout_decision_list,
                    "time_list": time_list
                }
        elif self.input_signal.label == "constant":
            # Same per-condition loop as "fit", but always uses the
            # constant-input simulation routine (value + fixed duration).
            for key in self.input_signal.value.keys():
                if ModelService.is_window(key):
                    current_input_value = self.input_signal.value[key]["value"]
                    current_input_duration = (self.input_signal.value[key]["time_end"] - self.input_signal.value[key]["time_start"]) * self.input_signal.value[key]["trials_per_simulation"]
                else:
                    current_input_value = self.input_signal.value[key]["value"]
                    current_input_duration = self.input_signal.value[key]["duration"]
                response_time_list, bout_decision_list, time_list, _ = self.simulate_trial_constant(
                    current_input_value,
                    current_input_duration,
                    dt)
                output[key] = {
                    "response_time_list": response_time_list,
                    "bout_decision_list": bout_decision_list,
                    "time_list": time_list
                }
        else:
            # Default: simulate number_trials independent trials, drawing a
            # fresh stimulus each time if fitting, then pool all results
            # into a single flattened output under one implicit condition.
            for index_trial in range(number_trials):
                if self.is_fitting:
                    self.compute_input_signal(index=self.counter_input_signal, scaling_factor=self.scaling_factor_input)
                    self._update_counter_input_signal(random=False)
                response_time_list_trial, bout_decision_list_trial, time_list_trial, _ = self.simulate_trial(self.input_signal.value, dt)
                response_time_list[index_trial] = response_time_list_trial
                bout_decision_list[index_trial] = bout_decision_list_trial
                time_list[index_trial] = time_list_trial

            # Flatten per-trial lists-of-lists into pooled bout-event lists.
            response_time_list = list(itertools.chain.from_iterable(response_time_list))
            bout_decision_list = list(itertools.chain.from_iterable(bout_decision_list))
            time_list = list(itertools.chain.from_iterable(time_list))
            output = {
                self.input_signal.label: {
                    "response_time_list": response_time_list,
                    "bout_decision_list": bout_decision_list,
                    "time_list": time_list
                }
            }
        print(f"DEBUG | {(time.time_ns()-time0) / (1e9)}")
        return output

    def simulate_trial(self, input_signal, dt=None):
        # Unpacks fittable parameters and forwards them, with the input
        # signal cast to a numba-typed list, to the JIT-compiled routine.
        if dt is None:
            dt = self.parameters.dt.value
        return self.simulate_trial_computation(
                self.parameters.scaling_factor.value,
                self.parameters.threshold.value,
                self.parameters.noise_sigma.value,
                self.parameters.leak.value,
                self.parameters.inactive_time.value,
                self.parameters.residual_after_bout.value,
                nb.typed.List(input_signal),
                dt
            )

    @staticmethod
    @nb.njit()
    def simulate_trial_computation(
            scaling_factor,
            threshold,
            noise_sigma,
            leak,
            inactive_time,
            residual_after_bout,
            input_signal, dt):
        # JIT-compiled integration loop for a time-varying input signal.
        # Kept static (no `self`) so numba can compile it efficiently.
        xs = np.zeros(len(input_signal))
        ts = np.zeros(len(input_signal))
        bout_counter = 0
        bout_decision_list = np.full(len(input_signal), np.nan)
        response_time_list = np.full(len(input_signal), np.nan)
        time_list = np.full(len(input_signal), np.nan)

        time_last_bout = 0
        # Pre-draw all noise increments (Euler-Maruyama scaling by sqrt(dt)).
        noise = np.random.normal(0, noise_sigma * np.sqrt(dt), len(input_signal))
        for i in range(1, len(input_signal)):
            dx = scaling_factor * np.sqrt(input_signal[i]) - leak * xs[i - 1]

            # Euler-Maruyama update: drift term + stochastic noise term.
            xs[i] = xs[i - 1] + dx * dt + noise[i]
            ts[i] = ts[i - 1] + dt

            time_since_last_bout = ts[i] - time_last_bout  # no refractory period for the first bout

            if (time_since_last_bout > inactive_time):  # refractory period 500 ms
                if abs(xs[i]) >= threshold:
                    # Threshold crossed: register a bout; sign of xs sets
                    # decision as error (0) or correct (1).
                    response_time_list[bout_counter] = time_since_last_bout
                    bout_decision_list[bout_counter] = 0 if xs[i] < 0 else 1
                    time_list[bout_counter] = ts[i]
                    time_last_bout = ts[i]
                    bout_counter += 1
                    # Partial reset toward threshold, scaled by residual_after_bout.
                    xs[i] = np.sign(xs[i]) * threshold * abs(residual_after_bout)
            else:  # commenting this block, prevents the inactive_time to affect also the integrator, impacting only the effector
                # Refractory period: freeze integrator state (no update).
                xs[i] = xs[i - 1]

        # Drop unused (NaN) preallocated slots, keeping only actual bouts.
        response_time_list = response_time_list[~np.isnan(response_time_list)]
        bout_decision_list = bout_decision_list[~np.isnan(bout_decision_list)]
        time_list = time_list[~np.isnan(time_list)]

        return response_time_list, bout_decision_list, time_list, xs

    def simulate_trial_constant(self, value_input_signal, duration_input_signal, dt=None):
        # Analogous to simulate_trial but for a constant input value held
        # over a fixed duration (used for "window"/"constant" conditions).
        if dt is None:
            dt = self.parameters.dt.value
        return self.simulate_trial_computation_constant(
                self.parameters.scaling_factor.value,
                self.parameters.threshold.value,
                self.parameters.noise_sigma.value,
                self.parameters.leak.value,
                self.parameters.inactive_time.value,
                self.parameters.residual_after_bout.value,
                value_input_signal,
                duration_input_signal,
                dt
            )

    @staticmethod
    @nb.njit()
    def simulate_trial_computation_constant(
            scaling_factor,
            threshold,
            noise_sigma,
            leak,
            inactive_time,
            residual_after_bout,
            value_input_signal,
            duration_input_signal,
            dt):
        # JIT-compiled integration loop for a constant input, run for
        # duration_input_signal/dt steps. Scalar state instead of arrays,
        # since the input value never changes within the trial.
        xs_old = 0
        ts = 0
        bout_counter = 0
        duration_input_signal += dt
        bout_decision_list = np.full(int(duration_input_signal/dt), np.nan)
        response_time_list = np.full(int(duration_input_signal/dt), np.nan)
        time_list = np.full(int(duration_input_signal/dt), np.nan, dtype=np.float32)

        time_last_bout = 0
        noise = np.random.normal(0, noise_sigma * np.sqrt(dt), int(duration_input_signal/dt))
        for t_i in range(0, int(duration_input_signal/dt)):
            dx = scaling_factor * np.sqrt(value_input_signal) - leak * xs_old

            xs = xs_old + dx * dt + noise[t_i]
            ts += dt

            time_since_last_bout = ts - time_last_bout  # no refractory period for the first bout

            if (time_since_last_bout > inactive_time):
                if abs(xs) >= threshold:
                    # Same bout-registration logic as the time-varying
                    # version, operating on scalars instead of arrays.
                    response_time_list[bout_counter] = time_since_last_bout
                    bout_decision_list[bout_counter] = 0 if xs < 0 else 1
                    time_list[bout_counter] = ts
                    time_last_bout = ts
                    bout_counter += 1
                    xs = np.sign(xs) * threshold * abs(residual_after_bout)
            else:  # commenting this block, prevents the inactive_time to affect also the integrator, impacting only the effector
                xs = xs_old
            xs_old = xs

        response_time_list = response_time_list[~np.isnan(response_time_list)]
        bout_decision_list = bout_decision_list[~np.isnan(bout_decision_list)]
        time_list = time_list[~np.isnan(time_list)]

        return response_time_list, bout_decision_list, time_list, xs  # return xs as well just for consistency with non-constant method

    def evaluate_output_signal(self, output_signal, resolution=None, missing_data_score=100, focus_scope=None, time_selection=True):
        # Public scoring entry point; optionally smooths the score across
        # recent fitting iterations (loss_memory) to stabilize the optimizer.
        score_dict = self.evaluate_output_signal_computation(output_signal, resolution, missing_data_score, focus_scope, time_selection)
        score = score_dict["score"]

        if self.size_loss_memory is not None and self.size_loss_memory > 1:
            self.loss_memory.append(score)
            if len(self.loss_memory) > self.size_loss_memory:
                self.loss_memory.pop(0)
            if len(self.loss_memory) > 1:
                # Average of past scores blended with the latest one.
                score = (np.mean(self.loss_memory[:-1]) + self.loss_memory[-1]) / 2
                # score = sum([(index+1)*loss for index, loss in enumerate(self.loss_memory)]) / sum(range(1, len(self.loss_memory)+1))

        return score

    def evaluate_output_signal_computation(self, output_signal, resolution=None, missing_data_score=100, focus_scope=None, time_selection=True):
        # Core loss: for each condition, compares simulated vs experimental
        # correct/error response-time distributions via KL divergence and
        # sums across conditions into one scalar score (lower = better fit).
        score_dict = {}
        score = 0

        if focus_scope is None:
            print(f"WARNING | focus_scope parameter is now mandatory for loss function computation. If not provided, (0, 3) will be used as default")
            focus_scope = (0, 3)

        if resolution is None:
            resolution = self.resolution_distribution

        if self.smooth_loss:
            # Savitzky-Golay smoothing config applied to simulated histograms.
            smoothing = {"is_simmetric": False,
                         "label": "savitzky_golay",
                         "window_size": 11,
                         "polynomial_order": 3}
        else:
            smoothing = None

        try:
            if time_selection:
                # Restrict training data to the stimulus presentation window.
                df_train_time = self.data_train.query(
                    f"start_time > {self.stimulus['time_start_stimulus']} and end_time < {self.stimulus['time_end_stimulus']}")
            else:
                df_train_time = self.data_train
            for key in output_signal.keys():
                score_dict[key] = {}
                response_time_list = np.array(output_signal[key]["response_time_list"])
                bout_decision_list = np.array(output_signal[key]["bout_decision_list"])
                if len(response_time_list) > self.max_bout_per_simulation:
                    # Runaway/unstable simulation: short-circuit with max penalty.
                    score = MAX_SCORE
                    score_dict["score"] = score
                    return score_dict

                if len(response_time_list) == 0:
                    response_time_list = []
                    bout_decision_list = []

                if ModelService.is_window(key):
                    # Window condition: filter to its exact time span.
                    time_start = self.input_signal.value[key]["time_start"]
                    time_end = self.input_signal.value[key]["time_end"]
                    duration_window = time_end - time_start
                    df_train_filtered = self.data_train.query(f"start_time > {time_start} and end_time < {time_end}")
                else:
                    # Regular condition: filter by matching stimulus value.
                    df_train_filtered = df_train_time[df_train_time[self.analysed_parameter] == key]
                df_correct = df_train_filtered[df_train_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1]
                data_correct = [] if df_correct.empty else df_correct[ConfigurationExperiment.ResponseTimeColumn]
                df_error = df_train_filtered[df_train_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0]
                data_error = [] if df_error.empty else df_error[ConfigurationExperiment.ResponseTimeColumn]

                # Determine number of individuals contributing to this
                # condition, trying several possible ID column/index names
                # since datasets may store fish identity differently.
                try:
                    number_individuals = len(df_train_filtered["fish_ID"].unique()) if self.multiple_individuals else 1
                except KeyError:
                    try:
                        number_individuals = len(
                            df_train_filtered["experiment_ID"].unique()) if self.multiple_individuals else 1
                    except KeyError:
                        try:
                            number_individuals = len(
                                df_train_filtered.index.unique("experiment_ID")) if self.multiple_individuals else 1
                        except (KeyError, ValueError):
                            try:
                                number_individuals = len(
                                    df_train_filtered.index.unique("fish_ID")) if self.multiple_individuals else 1
                            except (KeyError, ValueError):
                                number_individuals = 1

                # Total simulated duration, used to normalize simulated vs
                # experimental event rates.
                duration_simulation = self.time_experimental_trial * self.trials_per_simulation  # seconds
                if ModelService.is_window(key):
                    try:
                        duration_experiment = len(df_train_filtered['trial'].unique()) * duration_window * number_individuals  # seconds
                    except KeyError:
                        duration_experiment = np.sum(BehavioralProcessing.get_duration_trials_in_df(df_train_filtered, fixed_time_trial=duration_window)) * number_individuals  # seconds
                else:
                    if self.fit_options is not None and "duration_experiment" in self.fit_options.keys() and self.fit_options["duration_experiment"]:
                        # Externally supplied override for experimental duration.
                        duration_experiment = self.fit_options["duration_experiment"][key]
                    else:
                        try:
                            duration_experiment = len(
                                df_train_filtered['trial'].unique()) * self.time_experimental_trial * number_individuals  # seconds
                        except KeyError:
                            duration_experiment = np.sum(BehavioralProcessing.get_duration_trials_in_df(df_train_filtered,
                                                                                                        fixed_time_trial=self.time_experimental_trial))  # seconds
                        if self.fit_options is not None and "duration_correction" in self.fit_options.keys() and self.fit_options["duration_correction"]:
                            # Optional multiplicative correction to estimated duration.
                            duration_experiment *= self.fit_options["duration_correction"]

                if len(bout_decision_list) == 0:
                    simulation_correct = []
                    simulation_error = []
                else:
                    # Split simulated bouts into correct/error subsets,
                    # mirroring the split done on experimental data above.
                    index_response_time_list_correct = np.asarray(bout_decision_list == 1).nonzero()
                    simulation_correct = [] if len(index_response_time_list_correct) == 0 else response_time_list[
                        index_response_time_list_correct]
                    index_response_time_list_error = np.asarray(bout_decision_list == 0).nonzero()
                    simulation_error = [] if len(index_response_time_list_error) == 0 else response_time_list[
                        index_response_time_list_error]

                plot_distributions = False  # 1.85 < self.parameters.scaling_factor.value < 1.95

                if self.fit_options is not None and "disable_corr_optimization" in self.fit_options.keys() and self.fit_options["disable_corr_optimization"]:
                    # Correct-response loss term explicitly disabled.
                    KL_divergence_response_correct = 0
                    # fitting_now[key]["loss_corr"] = None
                elif self.fit_options is not None and "overlook_empty_condition" in self.fit_options.keys() and \
                        self.fit_options["overlook_empty_condition"] and len(data_correct) == 0:
                    # No experimental correct data for this condition and
                    # empty conditions are configured to be overlooked.
                    KL_divergence_response_correct = 0
                else:
                    if self.perturbation is not None:
                        perturbation = self.perturbation[key]["corr"]
                    else:
                        perturbation = None
                    # Main loss term: KL divergence between experimental and
                    # simulated correct RT distributions, duration-normalized.
                    KL_divergence_response_correct = BehavioralProcessing.kl_divergence_rt_distribution_weight(data_correct,
                                                                                                        simulation_correct,
                                                                                                        resolution=resolution,
                                                                                                        focus_scope=focus_scope,
                                                                                                        duration_0=duration_experiment,
                                                                                                        duration_1=duration_simulation,
                                                                                                        smoothing=smoothing,
                                                                                                        order_max_result=True,
                                                                                                        correct_by_area=False,
                                                                                                        plot_distributions=plot_distributions,
                                                                                                        perturbation=perturbation)
                score_dict[key]["corr"] = KL_divergence_response_correct
                if self.fit_options is not None and "disable_err_optimization" in self.fit_options.keys() and self.fit_options["disable_err_optimization"]:
                    KL_divergence_response_error = 0
                elif self.fit_options is not None and "overlook_empty_condition" in self.fit_options.keys() and \
                        self.fit_options["overlook_empty_condition"] and len(data_error) == 0:
                    KL_divergence_response_error = 0
                else:
                    if self.perturbation is not None:
                        perturbation = self.perturbation[key]["err"]
                    else:
                        perturbation = None
                    # Same divergence computation, for error RT distributions.
                    KL_divergence_response_error = BehavioralProcessing.kl_divergence_rt_distribution_weight(data_error,
                                                                                                      simulation_error,
                                                                                                      resolution=resolution,
                                                                                                      focus_scope=focus_scope,
                                                                                                      duration_0=duration_experiment,
                                                                                                      duration_1=duration_simulation,
                                                                                                      smoothing=smoothing,
                                                                                                      order_max_result=True,
                                                                                                      correct_by_area=False,
                                                                                                      plot_distributions=plot_distributions,
                                                                                                      perturbation=perturbation)
                score_dict[key]["err"] = KL_divergence_response_error

                # Accumulate this condition's divergence into the total score.
                score += KL_divergence_response_correct + KL_divergence_response_error

        except ValueError:
            # Fallback for degenerate/failed computations: assign the max
            # penalty score, perturbed slightly so the optimizer sees a
            # non-flat (still informative) landscape.
            score = MAX_SCORE + np.random.normal(scale=missing_data_score)
        score_dict["score"] = score

        return score_dict

    def store_history_fitting_de(self, x, convergence):
        # Callback invoked by the (differential evolution) optimizer at
        # each generation: re-simulates with current best parameters,
        # scores the result, and appends a snapshot to history_fitting
        # for later inspection of the fitting trajectory.
        output_signal = self.simulate(self.input_signal, dt=None)
        score_dict = self.evaluate_output_signal_computation(output_signal, focus_scope=self.focus_scope)
        fitting_now = {"n_fitting": self.n_fitting,
                       "score": score_dict["score"],
                       "convergence": convergence,
                       "x": [p.value for k, p in self.parameters_fittable]}
        for key in output_signal.keys():
            try:
                fitting_now[key] = {"loss_corr": score_dict[key]["corr"],
                                    "loss_err": score_dict[key]["err"]}
            except KeyError:
                fitting_now[key] = {"loss_corr": None,
                                    "loss_err": None}

        if self.debug:
            print(f"DEBUG | temporary solution: {x}")
            print(f"DEBUG | convergence: {convergence}")
            print(f"DEBUG | score: {score_dict['score']}")
        self.history_fitting.append(fitting_now)

    def define_stimulus(self, time_end_stimulus, time_start_stimulus=0):
        # Stores the stimulus presentation window (start/end times), used
        # elsewhere to filter training data and time-varying input signals.
        self.stimulus = {
            "time_start_stimulus": time_start_stimulus,
            "time_end_stimulus": time_end_stimulus
        }

    def compute_input_signal(self, index=0, scaling_factor=1):
        # Builds a constant-valued input signal from a single training-data
        # row's stimulus parameter (e.g. coherence at that trial index),
        # used to drive one simulated trial during random-order fitting.
        try:
            parameter = self.data_train.iloc[index][self.analysed_parameter]
        except KeyError:
            # Fallback if the index isn't directly positional (e.g. a
            # MultiIndex): reset the index first before positional access.
            parameter = self.data_train.reset_index().iloc[index][self.analysed_parameter]
        except IndexError:
            # Requested row doesn't exist: mark fitting as failed and use
            # a neutral placeholder value (0) rather than crashing.
            self.failed_fitting = True
            print(f"ERROR | {self.model_label} | fitting failed: impossible to find line {index} in the train dataframe")
            parameter = 0
        # Build a flat time vector spanning the stimulus window and a
        # constant input signal scaled by scaling_factor over that span.
        time_list = np.arange(0, self.stimulus['time_end_stimulus'], self.parameters.dt.value)
        input_signal = np.ones(len(time_list)) * parameter / scaling_factor
        self.input_signal = InputSignal(value=input_signal, label=parameter)