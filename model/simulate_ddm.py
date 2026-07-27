import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd

import matplotlib as mpl

from model.core.params import ParameterList, Parameter
from model.ddm import DDMstable
from service.behavioral_processing import BehavioralProcessing
from service.fast_functions import count_entries_in_dict
from service.model_service import ModelService
from utils.configuration_experiment import ConfigurationExperiment
from utils.constants import StimulusParameterLabel, Direction

from service.df_service import DFService

mpl.use('TkAgg')

# =========================================================================
# SCRIPT OVERVIEW
# Generates synthetic zebrafish bout datasets by simulating a DDMstable
# drift-diffusion model, either with manually-set/randomized parameters or
# with parameters loaded from existing fitted models (single model, or a
# whole directory of them). For each parameter set it simulates trials
# across all analysed stimulus conditions, builds a synthetic bout-level
# dataset, optionally scores it against a reference dataset, and saves the
# synthetic data/model to disk. Ends with an optional scatter-plot summary
# of the parameter values used across all simulated models.
# =========================================================================

# PARAMETERS
# data usage
save_synthetic_dataframe = True  # write the synthetic bout dataset to disk
save_model = False  # write the model's parameter/score summary to disk
set_parameters_from_model = False  # load parameters from a single fitted model file
set_parameters_from_dir = True  # load parameters from every fitted model file in a directory
compute_score = True  # score the synthetic dataset against a reference dataset
display = True  # plot parameter value distributions at the end
# configuration
time_start_trial = 0
time_start_stimulus = 10  # 10  # seconds
time_end_stimulus = 40  # 40  # seconds
time_end_trial = 50  # 50
max_response_time = 10

# OTHER PARAMETERS
# env
env = dotenv_values()
# path
path_save = Path(env['PATH_SAVE'])
if set_parameters_from_dir or compute_score:
    path_dir = Path(env['PATH_DIR']) / "benchmark" / "base_dataset"

dt = 0.01
response_time_label = ConfigurationExperiment.ResponseTimeColumn
analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
analysed_parameter_list = [0, 25, 50, 100]  # [3.75, 4.286, 10, 15, 30]  # [15, 30]   #
try:
    label_save = env["LABEL"]
except KeyError:
    label_save = None
offset_label = 10
number_trial_per_model_coh = 30
fish_age = '5'
mean_angle_bout = 50  # degrees
number_of_simulation = 1  # 1

# default parameter values
noise_sigma_value = 0.6
scaling_factor_value = 1.5
leak_value = 0
residual_after_bout_value = 0.1
inactive_time_value = 0
threshold_value = 1

# what to vary
generate_random_parameters = False  # if True, it will generate random parameter for what is True in the following lines
fit_noise_sigma = True
fit_scaling_factor = True
fit_leak = True
fit_residual_after_bout = False
fit_inactive_time = True
fit_threshold = False

# initialize parameters-related structures
# metadata (bounds, display label) for each parameter, used both to build
# Parameter objects and to lay out the final scatter-plot panels
parameter_list = [
    {"param": "noise_sigma",
     "label": "diffusion",
     "min": 0,
     "max": 3},
    {"param": "scaling_factor",
     "label": "drift",
     "min": -3,
     "max": 3},
    {"param": "leak",
     "label": "leak",
     "min": -3,
     "max": 3},
    {"param": 'residual_after_bout',
     "label": "reset",
     "min": 0,
     "max": 1},
    {"param": 'inactive_time',
     "label": "delay",
     "min": 0,
     "max": 1},
]

if set_parameters_from_dir:
    # collect every fitted model file in the directory, labeling each by
    # parts of its filename, and prepare one ParameterList per model found
    model_list = []
    for model_filepath in path_dir.glob('model_*.hdf5'):
        model_filename = str(model_filepath.name)
        # if model_filename.split("_")[2].endswith("c"):
        #     model_list.append({"label": f"{model_filename.split('_')[1]}_{model_filename.split('_')[2]}", "fit": model_filepath})
        model_list.append({"label": f"{model_filename.split('_')[1]}_{model_filename.split('_')[2]}", "fit": model_filepath})
    parameters_list = [ParameterList() for _ in range(len(model_list))]
else:
    # otherwise prepare number_of_simulation independent ParameterLists,
    # to be filled with manual/random parameter values below
    parameters_list = [ParameterList() for _ in range(number_of_simulation)]


if set_parameters_from_dir:
    # storage for the final parameter value used in each simulated model,
    # one array per parameter, sized to the number of models found on disk
    parameter_dict = {p["param"]: np.zeros(len(model_list)) for p in parameter_list}
else:
    # same storage, sized to the number of manually-run simulations instead
    parameter_dict = {p["param"]: np.zeros(number_of_simulation) for p in parameter_list}

for index, parameters in enumerate(parameters_list):
    if not set_parameters_from_model:
        label_save = f"test_{index}_"
    if set_parameters_from_dir:
        print(f"INFO | produce synthetic dataset from model {model_list[index]['label']}")
        label_save = f"{model_list[index]['label']}_"
    # retry loop: keep resimulating with fresh random parameters until an
    # "accepted" dataset is produced (only relevant when parameters are
    # generated/manual rather than loaded from an existing fitted model)
    dataset_not_produced = True
    while dataset_not_produced:
        response_time_list = {}
        decision_list = {}
        time_list = {}
        time_trial_list = np.arange(time_start_trial, time_end_trial, dt)

        if generate_random_parameters:
            # draw a random value only for parameters flagged for fitting/variation
            if fit_noise_sigma:
                noise_sigma_value = random.uniform(0, 3)
            if fit_scaling_factor:
                scaling_factor_value = random.uniform(-3, 3)
            if fit_leak:
                leak_value = random.uniform(-3, 3)
            if fit_residual_after_bout:
                residual_after_bout_value = random.uniform(0, 1)
            if fit_inactive_time:
                inactive_time_value = random.uniform(0, 1)
            if fit_threshold:
                threshold_value = np.random.uniform(0, 2)

        if not (set_parameters_from_model or set_parameters_from_dir):
            # log the parameter values actually used for this manual/random run
            print(f'''
                threshold: {threshold_value}
                scaling_factor: {scaling_factor_value}
                inactive_time: {inactive_time_value}
                leak: {leak_value}
                noise_sigma: {noise_sigma_value}
                residual_after_bout: {residual_after_bout_value}
            ''')

        # build the DDM parameter set for this run using the current values
        # (manual defaults, randomized, or later overwritten from a model file)
        parameters.add_parameter("dt", Parameter(value=dt))
        parameters.add_parameter("inactive_time", Parameter(min=0, max=1, value=inactive_time_value, fittable=fit_inactive_time))
        parameters.add_parameter("residual_after_bout", Parameter(min=0, max=1, value=residual_after_bout_value, fittable=fit_residual_after_bout))
        parameters.add_parameter("leak", Parameter(min=-3, max=3, value=leak_value, fittable=fit_leak))
        parameters.add_parameter("scaling_factor", Parameter(min=-3, max=3, value=scaling_factor_value, fittable=fit_scaling_factor))
        parameters.add_parameter("threshold", Parameter(min=0.001, max=2, value=threshold_value, fittable=fit_threshold))
        parameters.add_parameter("noise_sigma", Parameter(min=0, max=3, value=noise_sigma_value, fittable=fit_noise_sigma))

        if set_parameters_from_model or set_parameters_from_dir:
            # override the parameter values above with those read from an
            # existing fitted model file, and freeze them as non-fittable
            from_median_model = False
            from_best_model = True
            index_model = 0
            parameters_from_model = ["threshold", "inactive_time", "scaling_factor", "noise_sigma", "residual_after_bout", "leak"]  #
            if set_parameters_from_model:
                path_model = Path(env['PATH_MODEL_FIT'])
            elif set_parameters_from_dir:
                path_model = Path(model_list[index]["fit"])
            # path_model = Path(env['PATH_MODEL'])
            df_model = pd.read_hdf(path_model)
            for parameter in parameters_from_model:
                if from_median_model:
                    # use the median value across all fits stored in the model file
                    param_median = np.median(df_model[parameter])
                elif from_best_model:
                    # use the value from the single best-scoring fit
                    best_score = np.min(df_model['score'])
                    df_model_best = df_model.loc[df_model['score'] == best_score]
                    param_median = float(df_model_best[parameter].iloc[0])
                else:
                    # use the value at a specific fixed row index
                    param_median = np.array(df_model[parameter])[index_model]
                getattr(parameters, parameter).value = param_median
                getattr(parameters, parameter).fittable = False


        # simulate
        # unique ID for this simulated model, combining the loop index and a timestamp
        model_id = f"{index}_{int(datetime.now().timestamp())}"
        ddm_model = DDMstable(parameters, trials_per_simulation=number_trial_per_model_coh, time_experimental_trial=time_end_trial, scaling_factor_input=1)
        ddm_model.define_stimulus(time_start_stimulus=time_start_stimulus, time_end_stimulus=time_end_stimulus)

        # record the final parameter values used in this model, for the
        # end-of-script summary plot
        for p in parameter_list:
            parameter_dict[p["param"]][index] = getattr(ddm_model.parameters, p["param"]).value

        for index_parameter, parameter in enumerate(analysed_parameter_list):
            response_time_list[parameter] = {}
            decision_list[parameter] = {}
            time_list[parameter] = {}
            print(f"INFO | drift_diffusion naive | simulate model_id {model_id} for {analysed_parameter}={parameter}")
            for trial in range(number_trial_per_model_coh):
                # # oscillating input
                # input_signal = np.zeros(len(time_trial_list))
                # for i_time, time in enumerate(time_trial_list):
                #     if time >= time_start_stimulus and time <= time_end_stimulus:
                #         f = lambda t: (0.5 * (np.sin((2.0 * np.pi / parameter) * (t-5) - np.pi / 2.0) + 1))
                #         input_signal[i_time] = f(time)

                # constant input
                # build a stepwise input signal: zero outside the stimulus
                # window, constant (parameter/100) during it
                input_signal = np.zeros(len(time_trial_list))
                for i_time, time in enumerate(time_trial_list):
                    if time >= time_start_stimulus and time <= time_end_stimulus:
                        input_signal[i_time] += parameter / 100

                # label the trial
                # unique trial label across all conditions, so trials from
                # different stimulus parameters don't collide
                trial_label = trial + index_parameter * number_trial_per_model_coh

                # simulate
                response_time_list[parameter][trial_label], decision_list[parameter][trial_label], time_list[parameter][trial_label], _ = ddm_model.simulate_trial(
                    input_signal=input_signal)

        if not (set_parameters_from_model or set_parameters_from_dir):
            # only for manual/random parameter runs: check whether this
            # simulated dataset meets acceptance criteria (e.g. plausible
            # bout rates); if not, loop back and resimulate with new parameters
            try:
                duration = (time_end_trial - time_start_trial) * number_trial_per_model_coh
                dataset_accepted = BehavioralProcessing.check_dataset_accepted(response_time_list, decision_list,
                                                                               parameter_list=analysed_parameter_list, duration=duration)
            except (KeyError, ValueError):
                continue
            if dataset_accepted:
                dataset_not_produced = False
            else:
                continue
        else:
            # model-derived parameters are always accepted as-is
            dataset_not_produced = False

        # flatten all simulated bout events (across conditions and trials)
        # into a list of one-row DataFrames, one per synthetic bout
        df_bout_list = [np.nan for _ in range(count_entries_in_dict(time_list))]
        index_bout = -1
        for index_parameter, parameter in enumerate(response_time_list.keys()):
            for index_trial, trial in enumerate(response_time_list[parameter].keys()):
                for index_time, time in enumerate(time_list[parameter][trial]):
                    index_bout += 1
                    # random turning angle jitter around the baseline mean
                    flipped_bout_angle = mean_angle_bout + np.random.normal(scale=22.25)
                    time_adjusted = time - time_end_trial * trial
                    bout = {
                        "estimated_orientation_change": flipped_bout_angle,
                        'start_time': time,
                        'end_time': time,
                        "correct_bout": decision_list[parameter][trial][index_time],
                        StimulusParameterLabel.DIRECTION.value: Direction.LEFT.value,
                        analysed_parameter: parameter,
                        response_time_label: response_time_list[parameter][trial][index_time],
                        "fish_ID": model_id,
                        "model_id": model_id,
                        "trial": trial
                    }
                    df_bout = pd.DataFrame([bout])
                    df_bout_list[index_bout] = df_bout
        df_output_data = pd.concat(df_bout_list, ignore_index=True)

        if save_synthetic_dataframe:
            # choose an output filename depending on whether parameters came
            # from a model file (labeled run) or were manually/randomly set
            if set_parameters_from_model or set_parameters_from_dir:
                # file_name_save = f"data_synthetic_{label_save}{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_bestfit.hdf5"
                file_name_save = f"data_synthetic_{label_save}{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"
            else:
                file_name_save = f"data_synthetic_test_{(offset_label+index):03d}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"

            DFService.update_df(
                df_new=df_output_data,
                df_start=pd.DataFrame(),
                save_result=True,
                path_save=path_save,
                file_name_save=file_name_save
            )

        if compute_score:
            # compare this synthetic dataset against a matching reference
            # dataset found on disk (same offset/index-based filename),
            # skipping any files that are themselves fit results
            for path_data in path_dir.glob(f"data_synthetic_test_{(offset_label+index):03d}*.hdf5"):
                if "fit" in path_data.name:
                    continue
                else:
                    df_data = pd.read_hdf(str(path_data))
                    df_data_sample = df_data
                    score_dict = ModelService.compute_score_fit(df_data_sample, df_output_data)
                    ddm_model.score = score_dict["score"]

        # build a single-row summary of this model's identity and parameter values
        model_fish = {
            'name': ddm_model.model_label,
            'fish_id': model_id,
            'model_id': model_id,
            'score': ddm_model.score
        }
        for label, param in ddm_model.parameters:
            model_fish[label] = param.value
        df_output_model = pd.DataFrame([model_fish])

        if save_model:
            # choose an output filename following the same labeling
            # convention as the synthetic dataset above
            if set_parameters_from_model or set_parameters_from_dir:
                file_name_save = f"model_{label_save}{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"
            else:
                file_name_save = f"model_test_{(offset_label+index):03d}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5"

            DFService.update_df(
                df_new=df_output_model,
                df_start=pd.DataFrame(),
                save_result=True,
                path_save=path_save,
                file_name_save=file_name_save
            )

if display:
    # summary scatter plot: one panel per parameter, showing the spread of
    # final parameter values used across all simulated models (x-axis has
    # no meaning, just spreads points for visibility)
    fig, axs = plt.subplots(1, len(parameter_list))
    for i_p, p in enumerate(parameter_list):
        plot_section = axs[i_p]

        x = np.ones(len(parameter_dict[p["param"]]))
        y = parameter_dict[p["param"]]
        plot_section.scatter(x, y)

        plot_section.set_ylim(p["min"], p["max"])
        plot_section.set_xticks([])
        plot_section.set_title(p["label"])
    fig.savefig(path_save / "produce_data_parameter_distribution.png")