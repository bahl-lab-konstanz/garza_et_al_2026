'''
Overview:
Fits a DDMstable drift-diffusion model to experimental zebrafish bout
data, using bootstrapped resampling and optionally multiple repeated
fits per bootstrap. Can also regenerate a synthetic dataset from the
fitted model, and saves fitted parameters, fitting-history logs, and
synthetic data to HDF5.

WARNING: this script is demanding on resources and time. With the current configuration
is usually run on a single CPU in around 20 hours.
'''

import sys
from datetime import datetime
import numpy as np
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd

from model.core.params import ParameterList, Parameter
from model.core.signal import InputSignal
from model.ddm import DDMstable
from service.behavioral_processing import BehavioralProcessing
from service.df_service import DFService
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel, Direction

if __name__ == '__main__':
    # PARAMETERS
    # data usage
    exclude_straight_bout = True  # drop fast, near-straight (non-decision) bouts
    exclude_border = True  # drop bouts too close to the arena wall (set False for synthetic data)
    compute_synthetic_dataset = False  # simulate a full synthetic bout dataset after fitting
    save_synthetic_dataframe = False  # write the synthetic dataset to disk
    save_model = True  # save aggregated fitted parameters at the end
    save_single_model = False  # also save each individual model's parameters separately
    save_error = True  # save aggregated fitting-history log at the end
    save_single_error = False  # also save each individual model's fitting history separately
    compute_drift_diffusion_model = True  # master flag, kept for compatibility
    set_parameters_from_model = False  # initialize/freeze some parameters from an existing model file
    # experiment configurations
    time_hours_offset = 0  # hours
    time_hours_to_analyse = 8  # hours
    time_start_stimulus = 12  # seconds
    time_end_stimulus = 40  # seconds
    time_experimental_trial = 28
    max_response_time = 10
    precompute_duration_experiment = False  # precompute per-condition durations upfront for loss normalization
    # modeling configurations
    analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
    analysed_parameter_list = [0, 25, 50, 100]  # None  # [1, 5, 6, 7.5, 10]  #  # fixed conditions, or infer from data if None
    trials_per_simulation = 2000  #
    sample_percentage_size = 1  # fraction of dataset drawn per bootstrap
    number_bootstraps = 1  #  # number of independent bootstrap resamples
    number_model_per_booststrap = 1  # number of independent fits per bootstrap
    fish_age = '5'
    mean_angle_bout = 50  # degrees, baseline turning angle for synthetic bouts
    maxiter = 1500
    popsize = 100
    focus_scope = (0, 2)  # time window used by the loss function, set to None to avoid constraining

    # OTHER PARAMETERS
    # env
    # allow passing the .env path as CLI arg, default to a relative path otherwise
    try:
        env_path = sys.argv[1]
    except IndexError:
        env_path = "../.env"
    env = dotenv_values(env_path)
    label_simulation = env["LABEL"]  # tags all output filenames for this run
    # path
    path_save = Path(env['PATH_SAVE'])
    path_data = Path(env['PATH_DATA'])
    # PATH_MODEL is optional: used for target-model comparison or parameter initialization
    try:
        path_model = Path(env['PATH_MODEL'])
    except KeyError:
        path_model = None

    # MODEL DEFINITION
    dt = 0.01
    response_time_label = ConfigurationExperiment.ResponseTimeColumn  # 'response_time'
    # parameters
    # DDM parameter set with bounds and fittable flags; threshold fixed at 1
    # since other parameters are expressed relative to it
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=dt))
    parameters.add_parameter("residual_after_bout", Parameter(min=0, max=1, value=0, fittable=True))
    parameters.add_parameter("noise_sigma", Parameter(min=0.1, max=3, value=1, fittable=True))
    parameters.add_parameter("leak", Parameter(min=-3, max=3, value=0, fittable=True))
    parameters.add_parameter("threshold", Parameter(min=0.01, max=2, value=1, fittable=False))
    parameters.add_parameter("scaling_factor", Parameter(min=-3, max=3, value=1, fittable=True))
    parameters.add_parameter("inactive_time", Parameter(min=0, max=1, value=0, fittable=True))

    if set_parameters_from_model:
        # override initial value (and freeze) of selected parameters using
        # the median value observed in an existing pre-fitted model population
        parameters_from_model = ["residual_after_bout"]  # "scaling_factor", "noise_sigma", "threshold", "inactive_time", "leak"
        path_model = Path(env['PATH_MODEL'])
        df_model = pd.read_hdf(path_model)
        for parameter in parameters_from_model:
            try:
                getattr(parameters, parameter).value = np.median(df_model[parameter])
                getattr(parameters, parameter).fittable = False
            except KeyError:
                pass

    # MODEL FITTING
    # fetch and filter df containing training data
    # load full dataset, restrict to bouts within the stimulus window
    df_0 = pd.read_hdf(str(path_data))
    query_time = f"start_time > {time_start_stimulus} and end_time < {time_end_stimulus}"
    df_0 = df_0.query(query_time)
    if exclude_straight_bout:
        df_0 = BehavioralProcessing.remove_fast_straight_bout(df_0, threshold_response_time=100)
    if exclude_border:
        df_0 = BehavioralProcessing.remove_border_bout(df_0, BehavioralProcessing.transform_arena_measure(5))
    df = df_0

    # fetch df containing target model
    # optional reference model used later to compute fitting error vs ground truth
    df_model_target = pd.read_hdf(str(path_model)) if path_model is not None else pd.DataFrame()

    # simulation
    # accumulators for results across all bootstraps/models: fitting history,
    # synthetic bout data, and fitted model parameters
    df_error_list = []
    df_output_data = []
    df_output_model = pd.DataFrame()

    # fitting
    # draw independent resampled copies of the training data, stratified by
    # the analysed stimulus parameter, with replacement
    df_fish_list = BehavioralProcessing.randomly_sample_df(df=df, sample_number=number_bootstraps, sample_percentage_size=sample_percentage_size, sample_per_column=analysed_parameter, with_replacement=True)
    for index_bootstrap in range(number_bootstraps):
        df_fish = df_fish_list[index_bootstrap]
        if analysed_parameter_list is None:
            # infer conditions from unique stimulus values in this bootstrap sample
            analysed_parameter_list = list(df_fish[analysed_parameter].unique())

        if precompute_duration_experiment:
            # precompute total experimental duration overall and per condition,
            # used to normalize simulated vs experimental event rates in the loss
            duration_experiment = {"tot": np.sum(
                BehavioralProcessing.get_duration_trials_in_df(df, fixed_time_trial=time_experimental_trial)
            ) * sample_percentage_size}
            for analysed_parameter_value in analysed_parameter_list:
                duration_experiment[analysed_parameter_value] = np.sum(
                    BehavioralProcessing.get_duration_trials_in_df(df[df[analysed_parameter] == analysed_parameter_value],
                                                                   fixed_time_trial=time_experimental_trial)
                ) * sample_percentage_size
        else:
            duration_experiment = None

        # ##### compute the input signal
        # constant input
        # one entry per stimulus condition, holding a fixed drift-input value
        # over the full simulated duration for that condition
        fitting_input_signal = InputSignal(label='constant', value={
            param: {"value": param / 100, "duration": trials_per_simulation * (time_end_stimulus - time_start_stimulus)} for
            param in analysed_parameter_list})

        for index_model in range(number_model_per_booststrap):
            # unique ID for this fit: simulation label + bootstrap/model index + timestamp
            model_id = f"{label_simulation}-{index_bootstrap}-{index_model}_{int(datetime.now().timestamp())}"
            ddm_model = DDMstable(parameters, trials_per_simulation=trials_per_simulation,
                                  time_experimental_trial=time_experimental_trial, fitting_resolution=100,
                                  multiple_individuals=True, scaling_factor_input=1, analysed_parameter=analysed_parameter, smooth_loss=False)
            ddm_model.define_stimulus(time_start_stimulus=time_start_stimulus, time_end_stimulus=time_end_stimulus)

            # fit all parameters but leak
            print(f"INFO | {ddm_model.model_label} | test {label_simulation} | compute model {index_model} for bootstrap {index_bootstrap}")
            # correction factor for the bootstrap sample being a fraction of
            # the full dataset, keeping duration-based normalization consistent
            duration_correction = df_fish.shape[0] / df_0.shape[0]
            ddm_model.fit(data_train=df_fish, method='bayesian', input_signal=fitting_input_signal,
                          max_number_iterations=maxiter, randomize_df_train_rows=True, size_loss_memory=1, focus_scope=focus_scope,
                          options={
                              "workers": 1,
                              "init": popsize,
                              "duration_experiment": duration_experiment,
                              "duration_correction": duration_correction,
                              "overlook_empty_condition": True
                          })

            if save_error:
                # build one row per optimizer iteration: score, fitted param
                # values (with error vs target model if available), and
                # per-condition correct/error loss components
                fitting_df_list = []
                for fitting_index, fitting_item in enumerate(ddm_model.history_fitting):
                    fitting_item_dict = {
                        'fish_id': label_simulation,
                        "model_id": [index_model],
                        "iteration": [fitting_index],
                        # "convergence": [fitting_item["convergence"]],
                        "score": [fitting_item["score"]],
                        "n_fitting": [fitting_item["n_fitting"]]
                    }
                    for i, param in enumerate(ddm_model.parameters_fittable):
                        fitting_item_dict[f"{param[0]}_value"] = [fitting_item["x"][i]]
                        try:
                            # normalized error vs target model, scaled by parameter range
                            fitting_item_dict[f"{param[0]}_error"] = [
                                (df_model_target[param[0]][0] - fitting_item["x"][i]) / (param[1].max - param[1].min)
                            ]
                        except (KeyError, ValueError):
                            pass  # no target model, or parameter missing from it
                    for analysed_parameter_value in analysed_parameter_list:
                        try:
                            fitting_item_dict[f"{analysed_parameter_value}_loss_corr"] = [fitting_item[analysed_parameter_value]["loss_corr"]]
                        except KeyError:
                            fitting_item_dict[f"{analysed_parameter_value}_loss_corr"] = None
                        try:
                            fitting_item_dict[f"{analysed_parameter_value}_loss_err"] = [fitting_item[analysed_parameter_value]["loss_err"]]
                        except KeyError:
                            fitting_item_dict[f"{analysed_parameter_value}_loss_err"] = None
                    fitting_df_list.append(pd.DataFrame(fitting_item_dict))
                    df_error_list.append(pd.DataFrame(fitting_item_dict))

            # collect this model's final fitted parameters/score into a
            # single-row summary, appended to the aggregated output table
            model_fish = {
                'name': ddm_model.model_label,
                'fish_id': label_simulation,
                'model_id': model_id,
                'score': ddm_model.score,
            }
            for label, param in ddm_model.parameters:
                model_fish[label] = param.value
            df_model = pd.DataFrame([model_fish])
            df_output_model = pd.concat([df_output_model, df_model], ignore_index=True)

            if save_single_model:
                # persist this individual model's parameters to its own HDF5 file
                DFService.update_df(
                    df_new=df_model,
                    df_start=pd.DataFrame(),
                    save_result=True,
                    path_save=path_save,
                    file_name_save=f"model_{label_simulation}-{index_bootstrap}-{index_model}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
                )

                if save_single_error:
                    # persist this individual model's fitting history to its own HDF5 file
                    DFService.update_df(
                        df_new=pd.concat(fitting_df_list),
                        df_start=pd.DataFrame(),
                        save_result=True,
                        path_save=path_save,
                        file_name_save=f"error_{label_simulation}-{index_bootstrap}-{index_model}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
                    )

            if compute_synthetic_dataset:
                # regenerate synthetic bout data from the fitted model, one
                # trial at a time per condition, so individual bouts can be
                # logged with per-trial metadata (unlike the batched fit simulation)
                for index_parameter, parameter in enumerate(analysed_parameter_list):
                    print(
                        f"INFO | drift_diffusion naive | simulate {trials_per_simulation} for test {label_simulation} at coherence {parameter}")
                    for trial in range(trials_per_simulation):
                        # print(f"INFO | drift_diffusion naive | simulate fish_id {fish_id} trial {trial} coherence {parameter}")
                        # constant input at this condition, simulate one trial
                        time_list = np.arange(time_start_stimulus, time_end_stimulus, dt)
                        input_signal = np.ones(len(time_list)) * parameter / 100
                        response_time_list, decision_list, time_list, _ = ddm_model.simulate_trial(
                            input_signal=input_signal)

                        # build one synthetic bout record per decision event,
                        # with a jittered turning angle and metadata linking
                        # it to this fish/model/trial
                        df_bout_list = [None for item in range(len(response_time_list))]
                        for index, time_item in enumerate(time_list):
                            flipped_bout_angle = mean_angle_bout + np.random.normal(scale=22.25)
                            bout = {
                                "estimated_orientation_change": flipped_bout_angle,
                                'start_time': time_item,
                                'end_time': time_item,
                                "correct_bout": decision_list[index],
                                StimulusParameterLabel.DIRECTION.value: Direction.LEFT.value,
                                analysed_parameter: parameter,
                                response_time_label: response_time_list[index],
                                "fish_ID": label_simulation,
                                "model_id": model_id,
                                "trial": trial + index_parameter * trials_per_simulation
                            }
                            df_bout_list[index] = pd.DataFrame([bout])
                        df_output_data.extend(df_bout_list)

    # SAVE RESULTS
    if save_synthetic_dataframe:
        # concatenate all synthetic per-bout DataFrames and write to one HDF5 file
        df_output_data = pd.concat(df_output_data, ignore_index=True)
        DFService.update_df(
            df_new=df_output_data,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            # file_name_save=f"data_synthetic_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"
            file_name_save=f"data_synthetic_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )

    if save_model:
        # save the aggregated fitted-model parameter table across all bootstraps/models
        DFService.update_df(
            df_new=df_output_model,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            file_name_save=f"model_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )

    if save_error:
        # concatenate and save the aggregated fitting-history/error log
        df_error = pd.concat(df_error_list, ignore_index=True)
        DFService.update_df(
            df_new=df_error,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            file_name_save=f"error_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )