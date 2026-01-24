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
    model_label = 'drift_diffusion_Dkl*'
    min_bout_per_simulation = 1
    max_bout_per_simulation = np.inf
    max_response_time_allowed = 0.5
    time_experimental_trial = 30  # seconds

    def __init__(self, parameters, trials_per_simulation=50,
                 scaling_factor_input=100, time_experimental_trial=None,
                 size_loss_memory=1, fitting_resolution=40, dt=None,
                 multiple_individuals=False, analysed_parameter=StimulusParameterLabel.COHERENCE.value,
                 smooth_loss=False, noise_sigma_sensitivity=0):
        super(DDMstable, self).__init__(parameters)

        self.trials_per_simulation = trials_per_simulation
        self.min_bout_per_simulation *= trials_per_simulation
        self.max_bout_per_simulation *= trials_per_simulation
        self.scaling_factor_input = scaling_factor_input
        self.size_loss_memory = size_loss_memory
        self.fitting_resolution = fitting_resolution
        self.multiple_individuals = multiple_individuals
        self.analysed_parameter = analysed_parameter
        self.smooth_loss = smooth_loss
        self.noise_sigma_sensitivity = noise_sigma_sensitivity
        if time_experimental_trial is not None:
            self.time_experimental_trial = time_experimental_trial
        if dt is not None:
            self.dt = dt
        self.resolution_distribution = None  # just initialize blank attribute, to populate when fit is prepared
        self.perturbation = None  # just initialize blank attribute, to update if needed when fit is prepared

    def prepare_for_fitting(self, input_signal=None, focus_scope=None, options=None):
        if focus_scope is not None:
            self.resolution_distribution = int((focus_scope[1] - focus_scope[0]) / (self.parameters.dt.value * 5))
        if isinstance(input_signal, InputSignal) and input_signal.label in ("fit", "constant"):
            if self.noise_sigma_sensitivity:
                self.perturbation = {}
                for key in input_signal.value.keys():
                    self.perturbation[key] = {"corr": np.random.normal(scale=self.noise_sigma_sensitivity, size=self.resolution_distribution),
                                              "err": np.random.normal(scale=self.noise_sigma_sensitivity, size=self.resolution_distribution)}
            else:
                self.perturbation = {}
                for key in input_signal.value.keys():
                    self.perturbation[key] = {"corr": np.zeros(self.resolution_distribution),
                                              "err": np.zeros(self.resolution_distribution)}

    def simulate(self, input_signal=None, dt=None, number_trials=None):
        time0 = time.time_ns()
        if number_trials is None:
            number_trials = self.trials_per_simulation
        if input_signal is not None:
            self.input_signal = input_signal
        response_time_list = [[] for i in range(number_trials)]
        bout_decision_list = [[] for i in range(number_trials)]
        time_list = [[] for i in range(number_trials)]

        output = {}
        if self.input_signal.label == "fit":
            for key in self.input_signal.value.keys():
                if ModelService.is_window(key):
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
            for index_trial in range(number_trials):
                if self.is_fitting:
                    self.compute_input_signal(index=self.counter_input_signal, scaling_factor=self.scaling_factor_input)
                    self._update_counter_input_signal(random=False)
                response_time_list_trial, bout_decision_list_trial, time_list_trial, _ = self.simulate_trial(self.input_signal.value, dt)
                response_time_list[index_trial] = response_time_list_trial
                bout_decision_list[index_trial] = bout_decision_list_trial
                time_list[index_trial] = time_list_trial

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
        xs = np.zeros(len(input_signal))
        ts = np.zeros(len(input_signal))
        bout_counter = 0
        bout_decision_list = np.full(len(input_signal), np.nan)  # np.full(len(input_signal), np.nan)
        response_time_list = np.full(len(input_signal), np.nan)  # np.full(len(input_signal), np.nan)
        time_list = np.full(len(input_signal), np.nan)

        time_last_bout = 0
        noise = np.random.normal(0, noise_sigma * np.sqrt(dt), len(input_signal))
        for i in range(1, len(input_signal)):
            # dx = self.parameters_fittable.scaling_factor.value * input_signal[i] + random.gauss(0, self.parameters_fittable.noise_sigma.value) - xs[i - 1]
            dx = scaling_factor * input_signal[i] - leak * xs[i - 1]  # NOTE: I changed np.sqrt(input_signal[i]) input_signal[i]
            # dx = scaling_factor * input_signal[i] - leak * xs[i - 1]

            xs[i] = xs[i - 1] + dx * dt + noise[i]  # / (1/leak + difficulty_factor * (1 - input_signal[i]))
            ts[i] = ts[i - 1] + dt

            time_since_last_bout = ts[i] - time_last_bout  # no refractory period for the first bout

            if (time_since_last_bout > inactive_time):  # refractory period 500 ms
                if abs(xs[i]) >= threshold:
                    response_time_list[bout_counter] = time_since_last_bout
                    bout_decision_list[bout_counter] = 0 if xs[i] < 0 else 1
                    time_list[bout_counter] = ts[i]
                    time_last_bout = ts[i]
                    bout_counter += 1
                    xs[i] = np.sign(xs[i]) * threshold * abs(residual_after_bout)
            else:  # uncommenting this block, makes the inactive_time affect also the integrator, not only the effector
                xs[i] = xs[i - 1]

        response_time_list = response_time_list[~np.isnan(response_time_list)]
        bout_decision_list = bout_decision_list[~np.isnan(bout_decision_list)]
        time_list = time_list[~np.isnan(time_list)]

        return response_time_list, bout_decision_list, time_list, xs

    def simulate_trial_constant(self, value_input_signal, duration_input_signal, dt=None):
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
        xs_old = 0
        ts = 0
        bout_counter = 0
        duration_input_signal += dt  # from now on I will use this as end point of ranges, so I h
        bout_decision_list = np.full(int(duration_input_signal/dt), np.nan)  # np.full(len(input_signal), np.nan)
        response_time_list = np.full(int(duration_input_signal/dt), np.nan)  # np.full(len(input_signal), np.nan)
        time_list = np.full(int(duration_input_signal/dt), np.nan, dtype=np.float32)

        time_last_bout = 0
        noise = np.random.normal(0, noise_sigma * np.sqrt(dt), int(duration_input_signal/dt))
        for t_i in range(0, int(duration_input_signal/dt)):
            dx = scaling_factor * np.sqrt(value_input_signal) - leak * xs_old
            # dx = scaling_factor * value_input_signal - leak * xs_old

            xs = xs_old + dx * dt + noise[t_i]  # / (1/leak + difficulty_factor * (1 - input_signal[i]))
            ts += dt

            time_since_last_bout = ts - time_last_bout  # no refractory period for the first bout

            if (time_since_last_bout > inactive_time):  # refractory period 500 ms
                if abs(xs) >= threshold:
                    response_time_list[bout_counter] = time_since_last_bout
                    bout_decision_list[bout_counter] = 0 if xs < 0 else 1
                    time_list[bout_counter] = ts
                    time_last_bout = ts
                    bout_counter += 1
                    xs = np.sign(xs) * threshold * abs(residual_after_bout)
            else:  # uncommenting this block, makes the inactive_time affect also the integrator, not only the effector
                xs = xs_old
            xs_old = xs

        response_time_list = response_time_list[~np.isnan(response_time_list)]
        bout_decision_list = bout_decision_list[~np.isnan(bout_decision_list)]
        time_list = time_list[~np.isnan(time_list)]

        return response_time_list, bout_decision_list, time_list, xs  # return xs as well just for consistency with non-constant method

    def evaluate_output_signal(self, output_signal, resolution=None, missing_data_score=100, focus_scope=None, time_selection=True):
        score_dict = self.evaluate_output_signal_computation(output_signal, resolution, missing_data_score, focus_scope, time_selection)
        score = score_dict["score"]

        if self.size_loss_memory is not None and self.size_loss_memory > 1:
            self.loss_memory.append(score)
            if len(self.loss_memory) > self.size_loss_memory:
                self.loss_memory.pop(0)
            if len(self.loss_memory) > 1:
                score = (np.mean(self.loss_memory[:-1]) + self.loss_memory[-1]) / 2
                # score = sum([(index+1)*loss for index, loss in enumerate(self.loss_memory)]) / sum(range(1, len(self.loss_memory)+1))

        return score

    def evaluate_output_signal_computation(self, output_signal, resolution=None, missing_data_score=100, focus_scope=None, time_selection=True):
        score_dict = {}
        score = 0

        if focus_scope is None:
            print(f"WARNING | focus_scope parameter is now mandatory for loss function computation. If not provided, (0, 3) will be used as default")
            focus_scope = (0, 3)

        if resolution is None:
            resolution = self.resolution_distribution

        if self.smooth_loss:
            smoothing = {"is_simmetric": False,
                         "label": "savitzky_golay",
                         "window_size": 11,
                         "polynomial_order": 3}
        else:
            smoothing = None

        try:
            if time_selection:
                df_train_time = self.data_train.query(
                    f"start_time > {self.stimulus['time_start_stimulus']} and end_time < {self.stimulus['time_end_stimulus']}")
            else:
                df_train_time = self.data_train
            for key in output_signal.keys():
                score_dict[key] = {}
                response_time_list = np.array(output_signal[key]["response_time_list"])
                bout_decision_list = np.array(output_signal[key]["bout_decision_list"])
                if len(response_time_list) > self.max_bout_per_simulation:
                    score = MAX_SCORE
                    score_dict["score"] = score
                    return score_dict

                if len(response_time_list) == 0:
                    response_time_list = []
                    bout_decision_list = []

                if ModelService.is_window(key):
                    time_start = self.input_signal.value[key]["time_start"]
                    time_end = self.input_signal.value[key]["time_end"]
                    duration_window = time_end - time_start
                    df_train_filtered = self.data_train.query(f"start_time > {time_start} and end_time < {time_end}")
                else:
                    df_train_filtered = df_train_time[df_train_time[self.analysed_parameter] == key]
                df_correct = df_train_filtered[df_train_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1]
                data_correct = [] if df_correct.empty else df_correct[ConfigurationExperiment.ResponseTimeColumn]
                df_error = df_train_filtered[df_train_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0]
                data_error = [] if df_error.empty else df_error[ConfigurationExperiment.ResponseTimeColumn]

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

                duration_simulation = self.time_experimental_trial * self.trials_per_simulation  # seconds
                if ModelService.is_window(key):
                    try:
                        duration_experiment = len(df_train_filtered['trial'].unique()) * duration_window * number_individuals  # seconds
                    except KeyError:
                        duration_experiment = np.sum(BehavioralProcessing.get_duration_trials_in_df(df_train_filtered, fixed_time_trial=duration_window)) * number_individuals  # seconds
                else:
                    if self.fit_options is not None and "duration_experiment" in self.fit_options.keys() and self.fit_options["duration_experiment"]:
                        duration_experiment = self.fit_options["duration_experiment"][key]
                    else:
                        try:
                            duration_experiment = len(
                                df_train_filtered['trial'].unique()) * self.time_experimental_trial * number_individuals  # seconds
                        except KeyError:
                            duration_experiment = np.sum(BehavioralProcessing.get_duration_trials_in_df(df_train_filtered,
                                                                                                        fixed_time_trial=self.time_experimental_trial))  # seconds
                        if self.fit_options is not None and "duration_correction" in self.fit_options.keys() and self.fit_options["duration_correction"]:
                            duration_experiment *= self.fit_options["duration_correction"]

                if len(bout_decision_list) == 0:
                    simulation_correct = []
                    simulation_error = []
                else:
                    index_response_time_list_correct = np.asarray(bout_decision_list == 1).nonzero()
                    simulation_correct = [] if len(index_response_time_list_correct) == 0 else response_time_list[
                        index_response_time_list_correct]
                    index_response_time_list_error = np.asarray(bout_decision_list == 0).nonzero()
                    simulation_error = [] if len(index_response_time_list_error) == 0 else response_time_list[
                        index_response_time_list_error]

                plot_distributions = False  # 1.85 < self.parameters.scaling_factor.value < 1.95

                if self.fit_options is not None and "disable_corr_optimization" in self.fit_options.keys() and self.fit_options["disable_corr_optimization"]:
                    KL_divergence_response_correct = 0
                    # fitting_now[key]["loss_corr"] = None
                elif self.fit_options is not None and "overlook_empty_condition" in self.fit_options.keys() and \
                        self.fit_options["overlook_empty_condition"] and len(data_correct) == 0:
                    KL_divergence_response_correct = 0
                else:
                    # KL_divergence_response_correct = BehavioralProcessing.rmse_rt_distribution(
                    #     data_correct,
                    #     simulation_correct,
                    #     resolution=resolution,
                    #     focus_scope=focus_scope,
                    #     duration_0=duration_experiment,
                    #     duration_1=duration_simulation,
                    #     smoothing=None)
                    if self.perturbation is not None:
                        perturbation = self.perturbation[key]["corr"]
                    else:
                        perturbation = None
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
                    # KL_divergence_response_error = BehavioralProcessing.rmse_rt_distribution(
                    #     data_error,
                    #     simulation_error,
                    #     resolution=resolution,
                    #     focus_scope=focus_scope,
                    #     duration_0=duration_experiment,
                    #     duration_1=duration_simulation,
                    #     smoothing=None)
                    if self.perturbation is not None:
                        perturbation = self.perturbation[key]["err"]
                    else:
                        perturbation = None
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

                score += KL_divergence_response_correct + KL_divergence_response_error  # + np.abs(self.parameters.leak.value)**2  # + abs(np.log(area_target/area_simulation))  # + 10*(1-float(self.input_signal.label)/100) * self.parameters.scaling_factor.value  # + abs(area_simulation - area_target) + abs(median_rt_target - median_rt_simulation)

        except ValueError:
            score = MAX_SCORE + np.random.normal(scale=missing_data_score)
        score_dict["score"] = score

        return score_dict

    def store_history_fitting_de(self, x, convergence):
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
        self.stimulus = {
            "time_start_stimulus": time_start_stimulus,
            "time_end_stimulus": time_end_stimulus
        }

    def compute_input_signal(self, index=0, scaling_factor=1):
        try:
            parameter = self.data_train.iloc[index][self.analysed_parameter]
        except KeyError:
            parameter = self.data_train.reset_index().iloc[index][self.analysed_parameter]
        except IndexError:
            self.failed_fitting = True
            print(f"ERROR | {self.model_label} | fitting failed: impossible to find line {index} in the train dataframe")
            parameter = 0
        time_list = np.arange(0, self.stimulus['time_end_stimulus'], self.parameters.dt.value)
        input_signal = np.ones(len(time_list)) * parameter / scaling_factor
        # input_signal = np.zeros(len(time_list))
        # input_signal[np.where(np.logical_and(self.stimulus['time_start_stimulus'] < time_list, time_list < self.stimulus['time_end_stimulus']))] = parameter / scaling_factor
        self.input_signal = InputSignal(value=input_signal, label=parameter)
