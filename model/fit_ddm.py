import sys
from datetime import datetime
import numpy as np
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd

from analysis_helpers.analysis.personal_dirs.Roberto.model.drift_diffusion.core.ddm_stable import DDMstable
from analysis_helpers.analysis.personal_dirs.Roberto.model.utils.params import Parameter, ParameterList
from analysis_helpers.analysis.personal_dirs.Roberto.model.utils.signal import InputSignal
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, Direction, ResponseTimeColumn
from analysis_helpers.analysis.personal_dirs.Roberto.utils.pre_analysis import PreAnalysis

if __name__ == '__main__':
    # PARAMETERS
    debug = False
    # data usage
    exclude_straight_bout = True
    exclude_border = True
    compute_synthetic_dataset = False
    save_synthetic_dataframe = False
    save_model = True
    save_single_model = False
    save_error = True
    save_single_error = False
    compute_drift_diffusion_model = True
    set_parameters_from_model = False
    # experiment configurations
    time_hours_offset = 0  # hours
    time_hours_to_analyse = 8  # hours
    time_start_stimulus = 12  # seconds
    time_end_stimulus = 40  # seconds
    time_experimental_trial = 28
    max_response_time = 10
    precompute_duration_experiment = False
    # modeling configurations
    analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
    analysed_parameter_list = [0, 25, 50, 100]  # None  # [1, 5, 6, 7.5, 10]  #
    trials_per_simulation = 2000  # 20  #
    sample_percentage_size = 1
    number_bootstraps = 1  # 1  #
    number_model_per_booststrap = 1  # 1
    fish_age = '5'
    mean_angle_bout = 50  # degrees
    maxiter = 1500  # 2  #
    popsize = 100  # 5  #
    focus_scope = (0, 2)  # None

    # OTHER PARAMETERS
    # env
    try:
        env_path = sys.argv[1]
    except IndexError:
        env_path = "../.env"
    env = dotenv_values(env_path)
    label_simulation = env["LABEL"]
    # path
    path_save = Path(env['PATH_SAVE'])
    path_data = Path(env['PATH_DATA'])
    try:
        path_model = Path(env['PATH_MODEL'])
    except KeyError:
        path_model = None

    # MODEL DEFINITION
    dt = 0.01
    response_time_label = ResponseTimeColumn  # 'response_time'
    # parameters
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=dt))
    parameters.add_parameter("residual_after_bout", Parameter(min=0, max=1, value=0, fittable=True))
    parameters.add_parameter("noise_sigma", Parameter(min=0.1, max=3, value=1, fittable=True))
    parameters.add_parameter("leak", Parameter(min=-3, max=3, value=0, fittable=True))
    parameters.add_parameter("threshold", Parameter(min=0.01, max=2, value=1, fittable=False))
    parameters.add_parameter("scaling_factor", Parameter(min=-3, max=3, value=1, fittable=True))
    parameters.add_parameter("inactive_time", Parameter(min=0, max=1, value=0, fittable=True))

    if set_parameters_from_model:
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
    df_0 = pd.read_hdf(str(path_data))
    query_time = f"start_time > {time_start_stimulus} and end_time < {time_end_stimulus}"
    df_0 = df_0.query(query_time)
    if exclude_straight_bout:
        df_0 = BehavioralProcessing.remove_fast_straight_bout(df_0, threshold_response_time=100)
    if exclude_border:
        df_0 = BehavioralProcessing.remove_border_bout(df_0, BehavioralProcessing.transform_arena_measure(5))
    df = df_0
    # for i_row, row in df.iterrows():
    #     if row[StimulusParameterLabel.COHERENCE.value] < 12.5:
    #         df.loc[i_row, StimulusParameterLabel.COHERENCE.value] = 0
    #     elif row[StimulusParameterLabel.COHERENCE.value] > 12.5 and row[StimulusParameterLabel.COHERENCE.value] < 37.5:
    #         df.loc[i_row, StimulusParameterLabel.COHERENCE.value] = 25
    #     elif row[StimulusParameterLabel.COHERENCE.value] > 37.5 and row[StimulusParameterLabel.COHERENCE.value] < 62.5:
    #         df.loc[i_row, StimulusParameterLabel.COHERENCE.value] = 50
    #     elif row[StimulusParameterLabel.COHERENCE.value] > 62.5 and row[StimulusParameterLabel.COHERENCE.value] < 87.5:
    #         df.loc[i_row, StimulusParameterLabel.COHERENCE.value] = 75
    #     else:
    #         df.loc[i_row, StimulusParameterLabel.COHERENCE.value] = 100

    # fetch df containing target model
    df_model_target = pd.read_hdf(str(path_model)) if path_model is not None else pd.DataFrame()

    # simulation
    df_error_list = []
    df_output_data = []
    df_output_model = pd.DataFrame()

    # fitting
    df_fish_list = BehavioralProcessing.randomly_sample_df(df=df, sample_number=number_bootstraps, sample_percentage_size=sample_percentage_size, sample_per_column=analysed_parameter, with_replacement=True)
    for index_bootstrap in range(number_bootstraps):
        df_fish = df_fish_list[index_bootstrap]
        if analysed_parameter_list is None:
            analysed_parameter_list = list(df_fish[analysed_parameter].unique())

        if precompute_duration_experiment:
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
        fitting_input_signal = InputSignal(label='constant', value={
            param: {"value": param / 100, "duration": trials_per_simulation * (time_end_stimulus - time_start_stimulus)} for
            param in analysed_parameter_list})

        # # other fitting input
        # fitting_input_signal = InputSignal(label='fit', value={
        #     param: [param / 100 for _ in trials_per_simulation * (time_end_stimulus - time_start_stimulus)]
        #     for
        #     param in analysed_parameter_list})

        # # oscillating input
        # fitting_input_signal = InputSignal(label='fit', value={})
        # time = np.arange(time_start_stimulus, trials_per_simulation * time_end_stimulus,
        #                  dt)  # for now it only works properly if time_start_stimulus==0
        # for parameter in analysed_parameter_list:
        #     f = lambda t: (0.5 * (np.sin((2.0 * np.pi / parameter) * t - np.pi / 2.0) + 1))
        #     fitting_input_signal.value[parameter] = f(time)

        for index_model in range(number_model_per_booststrap):
            model_id = f"{label_simulation}-{index_bootstrap}-{index_model}_{int(datetime.now().timestamp())}"
            ddm_model = DDMstable(parameters, trials_per_simulation=trials_per_simulation,
                                  time_experimental_trial=time_experimental_trial, fitting_resolution=100,
                                  multiple_individuals=True, scaling_factor_input=1, analysed_parameter=analysed_parameter, smooth_loss=False)
            ddm_model.define_stimulus(time_start_stimulus=time_start_stimulus, time_end_stimulus=time_end_stimulus)

            # fit all parameters but leak
            print(f"INFO | {ddm_model.model_label} | test {label_simulation} | compute model {index_model} for bootstrap {index_bootstrap}")
            duration_correction = df_fish.shape[0] / df_0.shape[0]
            ddm_model.fit(data_train=df_fish, method='bayesian', input_signal=fitting_input_signal,
                          max_number_iterations=maxiter, randomize_df_train_rows=True, size_loss_memory=1, focus_scope=focus_scope,
                          options={
                              # "tol": 0.01,
                              "workers": 1,  # 1,  #
                              # "polish": False,
                              "init": popsize,  # np.transpose(
                              #     np.array([
                              #         np.random.exponential(scale=1 / 3, size=popsize),
                              #         np.random.exponential(scale=1 / 3, size=popsize),
                              #         np.random.exponential(scale=1 / 3, size=popsize),
                              #         np.random.exponential(scale=1 / 3, size=popsize),
                              #         # np.random.exponential(scale=1 / 3, size=popsize),
                              #     ])
                              # ),
                              "duration_experiment": duration_experiment,
                              "duration_correction": duration_correction,
                              "overlook_empty_condition": True
                          })

            if save_error:
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
                            fitting_item_dict[f"{param[0]}_error"] = [
                                (df_model_target[param[0]][0] - fitting_item["x"][i]) / (param[1].max - param[1].min)
                            ]
                        except (KeyError, ValueError):
                            pass
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

            # ddm_model.set_parameter("residual_after_bout", fittable=False)
            # ddm_model.set_parameter("noise_sigma", fittable=False)
            # ddm_model.set_parameter("scaling_factor", fittable=False)
            # ddm_model.set_parameter("inactive_time", fittable=False)
            # ddm_model.fit(data_train=df_fish, method='bayesian', input_signal=fitting_input_signal,
            #               max_number_iterations=maxiter_leak, randomize_df_train_rows=True, size_loss_memory=1,
            #               focus_scope=focus_scope,
            #               options={
            #                   # "tol": 0.01,
            #                   "workers": 19,  # 1
            #                   "init": popsize_leak,
            #                   "duration_experiment": duration_experiment,
            #                   "overlook_empty_condition": True
            #               })
            #
            # if save_error:
            #     # fitting_df_list = []
            #     for fitting_index, fitting_item in enumerate(ddm_model.history_fitting):
            #         fitting_item_dict = {
            #             'fish_id': label_simulation,
            #             "model_id": [index_model],
            #             "iteration": [fitting_index],
            #             # "convergence": [fitting_item["convergence"]],
            #             "score": [fitting_item["score"]],
            #             "n_fitting": [fitting_item["n_fitting"]]
            #         }
            #         for i, param in enumerate(ddm_model.parameters_fittable):
            #             fitting_item_dict[f"{param[0]}_value"] = [fitting_item["x"][i]]
            #             try:
            #                 fitting_item_dict[f"{param[0]}_error"] = [
            #                     (df_model_target[param[0]][0] - fitting_item["x"][i]) / (param[1].max - param[1].min)
            #                 ]
            #             except (KeyError, ValueError):
            #                 pass
            #         for analysed_parameter_value in analysed_parameter_list:
            #             try:
            #                 fitting_item_dict[f"{analysed_parameter_value}_loss_corr"] = [fitting_item[analysed_parameter_value]["loss_corr"]]
            #             except KeyError:
            #                 fitting_item_dict[f"{analysed_parameter_value}_loss_corr"] = None
            #             try:
            #                 fitting_item_dict[f"{analysed_parameter_value}_loss_err"] = [fitting_item[analysed_parameter_value]["loss_err"]]
            #             except KeyError:
            #                 fitting_item_dict[f"{analysed_parameter_value}_loss_err"] = None
            #         fitting_df_list.append(pd.DataFrame(fitting_item_dict))
            #         df_error_list.append(pd.DataFrame(fitting_item_dict))

            model_fish = {
                'name': ddm_model.model_label,
                'fish_id': label_simulation,
                'model_id': model_id,
                'score': ddm_model.score,
                # 'converged': len(ddm_model.history_fitting) != maxiter
            }
            for label, param in ddm_model.parameters:
                model_fish[label] = param.value
            df_model = pd.DataFrame([model_fish])
            df_output_model = pd.concat([df_output_model, df_model], ignore_index=True)

            if save_single_model:
                PreAnalysis.update_df(
                    df_new=df_model,
                    df_start=pd.DataFrame(),
                    save_result=True,
                    path_save=path_save,
                    file_name_save=f"model_{label_simulation}-{index_bootstrap}-{index_model}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
                )

                if save_single_error:
                    PreAnalysis.update_df(
                        df_new=pd.concat(fitting_df_list),
                        df_start=pd.DataFrame(),
                        save_result=True,
                        path_save=path_save,
                        file_name_save=f"error_{label_simulation}-{index_bootstrap}-{index_model}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
                    )

            if compute_synthetic_dataset:
                for index_parameter, parameter in enumerate(analysed_parameter_list):
                    print(
                        f"INFO | drift_diffusion naive | simulate {trials_per_simulation} for test {label_simulation} at coherence {parameter}")
                    for trial in range(trials_per_simulation):
                        # print(f"INFO | drift_diffusion naive | simulate fish_id {fish_id} trial {trial} coherence {parameter}")
                        time_list = np.arange(time_start_stimulus, time_end_stimulus, dt)
                        input_signal = np.ones(len(time_list)) * parameter / 100
                        # for index, time in enumerate(time_list):
                        #     if time_start_stimulus < time < time_end_stimulus:
                        #         input_signal[index] = parameter / 100
                        #     else:
                        #         input_signal[index] = 0
                        # time0 = time.time()
                        response_time_list, decision_list, time_list, _ = ddm_model.simulate_trial(
                            input_signal=input_signal)
                        # print(f"PROFILING | {time.time()-time0}")

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
        df_output_data = pd.concat(df_output_data, ignore_index=True)
        PreAnalysis.update_df(
            df_new=df_output_data,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            # file_name_save=f"data_synthetic_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"
            file_name_save=f"data_synthetic_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )

    if save_model:
        PreAnalysis.update_df(
            df_new=df_output_model,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            file_name_save=f"model_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )

    if save_error:
        df_error = pd.concat(df_error_list, ignore_index=True)
        PreAnalysis.update_df(
            df_new=df_error,
            df_start=pd.DataFrame(),
            save_result=True,
            path_save=path_save,
            file_name_save=f"error_{label_simulation}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_fit.hdf5"
        )
