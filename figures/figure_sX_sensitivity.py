import copy
import pathlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import dotenv_values
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu

from analysis.personal_dirs.Roberto.model.drift_diffusion.core.ddm_stable import DDMstable
from analysis.personal_dirs.Roberto.model.utils.params import ParameterList, Parameter
from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, ResponseTimeColumn, Direction, \
    alphabet
from analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.model_service import ModelService
from analysis.personal_dirs.Roberto.utils.toolkit import count_entries_in_dict
from analysis.utils.figure_helper import Figure

# CONFIGURATIONS
# script
compute_sensitivity = False
save_result = True

# simulation
dt = 0.01
number_trial_per_model_coh = 30
time_start_stimulus = 10
time_end_stimulus = 40
time_end_trial = 50
mean_angle_bout = 50
response_time_label = ResponseTimeColumn
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = [0, 25, 50, 100]
delta_perturbation = 0.05
perturbation_epsilon = delta_perturbation/2
perturbation_array = np.arange(-0.5, 0.5+delta_perturbation, delta_perturbation)
number_of_tests = 1  # 5

# plot
style = BehavioralModelStyle()
xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start
padding = 1.5  # style.padding
padding_short = 0.75  # style.padding
plot_height = style.plot_height
plot_width = style.plot_width * 3
plot_width_short = style.plot_width * 0.9
i_plot_label = 0
palette = style.palette["default"]
style.add_palette("neutral", [Palette.color_neutral])

# env
env = dotenv_values()
# paths
try:
    path_dir = Path(env['PATH_DIR'])
    path_save = Path(env['PATH_SAVE'])
except:
    path_dir = None

parameter_list = [
    {"label": "noise_sigma",
     "label_show": "diffusion",
     "min": 0.0,
     "mean": 1.5,
     "max": 3.0},
    {"label": 'scaling_factor',
     "label_show": "drift",
     "min": -3,
     "mean": 0,
     "max": 3},
    {"label": 'leak',
     "label_show": "leak",
     "min": -3,
     "mean": 0,
     "max": 3},
    {"label": 'residual_after_bout',
     "label_show": "reset",
     "min": 0.0,
     "mean": 0.5,
     "max": 1.0},
    {"label": 'inactive_time',
     "label_show": "delay",
     "min": 0.0,
     "mean": 0.5,
     "max": 1.0},
]

if compute_sensitivity:
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=0.01))
    parameters.add_parameter("inactive_time", Parameter(min=0, max=1, value=0))
    parameters.add_parameter("residual_after_bout", Parameter(min=0, max=1, value=0))
    parameters.add_parameter("leak", Parameter(min=-3, max=3, value=0))
    parameters.add_parameter("scaling_factor", Parameter(min=-3, max=3, value=1))
    parameters.add_parameter("threshold", Parameter(min=0.001, max=2, value=1))
    parameters.add_parameter("noise_sigma", Parameter(min=0.01, max=3, value=1))

    index_model = 0
    score_list = []
    parameter_variation_list = []
    loss_dict = {p["label"]: {pert: [] for pert in perturbation_array} for p in parameter_list}
    for model_filepath in path_dir.glob('model_*.hdf5'):
        score_fit_model_list = np.zeros(number_of_tests)
        response_time_list = {}
        decision_list = {}
        time_list = {}
        time_trial_list = np.arange(0, time_end_trial, dt)

        index_model += 1
        model_filename = str(model_filepath.name)
        path_model = str(model_filepath)

        from_median_model = False
        from_best_model = True
        model_id = model_filename.split("_")[2].replace(".hdf5", "")
        df_model_target = pd.read_hdf(path_model)

        for p in parameter_list:
            best_score = np.min(df_model_target['score'])
            df_model_best = df_model_target.loc[df_model_target['score'] == best_score]
            param_best = float(df_model_best[p["label"]])

            getattr(parameters, p["label"]).value = param_best

        parameters_target = copy.deepcopy(parameters)

        for data_filepath in path_dir.glob(f'data_fish_{model_id}.hdf5'):
            path_data = str(data_filepath)
            df_data_target = pd.read_hdf(path_data)
            break

        for i_p_to_perturb, p_to_perturb in enumerate(parameter_list):
            for i_perturbation, perturbation in enumerate(perturbation_array):
                if perturbation == 0:
                    original_loss = np.min(df_model_target['score'])
                    loss_dict[p_to_perturb["label"]][perturbation].append(original_loss)
                else:
                    present_loss = 0
                    for i_test in range(number_of_tests):
                        parameters_perturbed = copy.deepcopy(parameters_target)
                        p_best = getattr(parameters_perturbed, p_to_perturb["label"]).value

                        p_perturbation = perturbation * (p_to_perturb["max"] - p_to_perturb["min"])
                        p_perturbed = p_best + p_perturbation
                        if p_perturbed > p_to_perturb["max"] or p_perturbed < p_to_perturb["min"]:
                            # p_perturbed = np.nan
                            present_loss = np.nan
                        else:
                            # p_perturbed = np.min((p_to_perturb["max"], p_perturbed))
                            # p_perturbed = np.max((p_to_perturb["min"], p_perturbed))
                            getattr(parameters_perturbed, p_to_perturb["label"]).value = p_perturbed

                            # simulate
                            model_id_label = f"{model_id}_{int(datetime.now().timestamp())}"
                            ddm_model = DDMstable(parameters_perturbed, trials_per_simulation=number_trial_per_model_coh,
                                                  time_experimental_trial=30, scaling_factor_input=1)
                            ddm_model.define_stimulus(time_start_stimulus=10, time_end_stimulus=40)

                            for index_parameter, parameter in enumerate(analysed_parameter_list):
                                response_time_list[parameter] = {}
                                decision_list[parameter] = {}
                                time_list[parameter] = {}
                                print(f"INFO | drift_diffusion naive | simulate model_id {model_id} for {analysed_parameter}={parameter}")
                                for trial in range(number_trial_per_model_coh):
                                    # constant input
                                    input_signal = np.zeros(len(time_trial_list))
                                    for i_time, time in enumerate(time_trial_list):
                                        if time >= time_start_stimulus and time <= time_end_stimulus:
                                            input_signal[i_time] += parameter / 100

                                    # label the trial
                                    trial_label = trial + index_parameter * number_trial_per_model_coh

                                    # simulate
                                    response_time_list[parameter][trial_label], decision_list[parameter][trial_label], time_list[parameter][
                                        trial_label], _ = ddm_model.simulate_trial(
                                        input_signal=input_signal)

                                df_bout_list = [np.nan for _ in range(count_entries_in_dict(time_list))]
                                index_bout = -1
                                for index_parameter, parameter in enumerate(response_time_list.keys()):
                                    for index_trial, trial in enumerate(response_time_list[parameter].keys()):
                                        for index_time, time in enumerate(time_list[parameter][trial]):
                                            index_bout += 1
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
                                                "fish_ID": model_id_label,
                                                "model_id": model_id,
                                                "trial": trial
                                            }
                                            df_bout = pd.DataFrame([bout])
                                            df_bout_list[index_bout] = df_bout
                                try:
                                    df_output_data = pd.concat(df_bout_list, ignore_index=True)
                                except ValueError:
                                    continue

                            score_dict = ModelService.compute_score_fit(df_data_target, df_output_data)
                            present_loss += score_dict["score"]

                    present_loss /= number_of_tests
                    loss_dict[p_to_perturb["label"]][perturbation].append(present_loss)

    if save_result:
        with open(path_save / "sensitivity.pkl", 'wb') as f:
            pickle.dump(loss_dict, f)
else:
    with open(path_dir / "sensitivity.pkl", 'rb') as f:
        loss_dict = pickle.load(f)


fig = Figure()

for i_p, p in enumerate(parameter_list):
    x = np.array(list(loss_dict[p["label"]].keys()))
    x_show = x * 100
    number_fish = len(loss_dict[p["label"]][-0.5])
    trajectory_array = np.zeros((number_fish, len(x)))
    for i_fish in range(number_fish):
        for i_pert in range(len(x)):
            trajectory_array[i_fish, i_pert] = loss_dict[p["label"]][x[i_pert]][i_fish]

    index_original_distribution = np.argwhere(np.abs(perturbation_array) < perturbation_epsilon)  # the closest we get to zero
    original_distribution = np.squeeze(trajectory_array[:, index_original_distribution])

    significant_list = []
    for i_pert in range(len(x)):
        if i_pert != index_original_distribution:
            pert_distribution = np.squeeze(trajectory_array[:, i_pert])
            pert_distribution = pert_distribution[~np.isnan(pert_distribution)]

            stat, p_value = mannwhitneyu(original_distribution, pert_distribution, nan_policy="omit")
            print(f"{p['label_show']} | perturbation: {x[i_pert]:.02f} | size: {len(pert_distribution)} | p_value: {p_value:.04f}")
            if p_value < 0.05:
                significant_list.append(x[i_pert])
    significant_list = np.array(significant_list)

    plot_p = fig.create_plot(plot_title=p['label_show'].capitalize(), plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=plot_width, errorbar_area=True,
                             xl=f"Parameter perturbation (%)" if i_p == len(parameter_list)-1 else None,
                             xmin=np.min(x_show), xmax=np.max(x_show),
                             xticks=[-50, -25, 0, 25, 50] if i_p == len(parameter_list)-1 else None,
                             yl="Loss", ymin=0, ymax=1, yticks=[0, 0.5, 1], vlines=[0])
    i_plot_label += 1
    ypos = ypos - plot_height - padding
    # xpos += plot_width + padding

    for i_fish in range(number_fish):
        plot_p.draw_line(x_show, trajectory_array[i_fish], lc=palette[i_p], lw=0.1, alpha=0.5)

    for i_k, k in enumerate(loss_dict[p["label"]].keys()):
        y = loss_dict[p["label"]][k]
        plot_p.draw_scatter(np.ones_like(y)*x_show[i_k], y, pc=palette[i_p], elw=0, alpha=0.8)

    # plot_p.draw_line(x_show, np.nanmedian(trajectory_array, axis=0), lc="k")

    x_positive_significant = np.min(significant_list[significant_list > perturbation_epsilon])
    xlim_positive_significant = np.array([x_positive_significant, np.max(perturbation_array)/1.5]) * 100
    plot_p.draw_line(xlim_positive_significant, np.ones(2)*0.85, lc="k")
    plot_p.draw_text(np.mean(xlim_positive_significant), 1.2, "*")
    print(fr"{p['label_show']} | significant for an perturbation of $\pm${x_positive_significant}%")

    x_negative_significant = np.max(significant_list[significant_list < perturbation_epsilon])
    xlim_negative_significant = np.array([np.min(perturbation_array)/1.5, x_negative_significant]) * 100
    plot_p.draw_line(xlim_negative_significant, np.ones(2)*0.95, lc="k")
    plot_p.draw_text(np.mean(xlim_negative_significant), 1.2, "*")

if save_result:
    fig.save(pathlib.Path.home() / 'Desktop' / f"figure_s5_sensitivity.pdf", open_file=True, tight=True)