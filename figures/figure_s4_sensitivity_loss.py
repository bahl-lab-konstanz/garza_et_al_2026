import copy
import itertools

import pandas as pd
import numpy as np
import pathlib

from scipy.stats import mannwhitneyu, laplace

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis.personal_dirs.Roberto.utils.constants import palette_0, \
    alphabet, StimulusParameterLabel, CorrectBoutColumn, ResponseTimeColumn
from analysis.utils.figure_helper import Figure

# plot configs
fit_distributions = False
fig = Figure()
style = BehavioralModelStyle()
plot_height = style.plot_height
plot_width = 1
xpos_start = 0.5
ypos_start = 0.5
xpos = xpos_start
ypos = ypos_start
padding = 1
padding_short = 0.75
i_plot_label = 0
palette = Palette.arlecchino
analysed_parameter_label = "Coherence (%)"
index_population = 1

# experimental configs
time_start_stimulus = 10
time_end_stimulus = 40
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = [0, 25, 50, 100]

score_config = {
    "label": "loss",
    "min": 0,
    "max": 10
}
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

loss_label = "weight"
path_data_sample = r"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/weight_nosmooth/100/data_synthetic_test_006_2024-11-19_17-27-07.hdf5"
path_real_data = r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/5_dpf"
models_in_group_list = [
    {"label_show": r"$\Sigma_{noise}$=0",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0",
     "loss_list": [],
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\5_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\5_dpf\data_synthetic_fish_all.hdf5",
     # None  #
     "dashes": None,
     "color": "k",
     "alpha": 1},
    {"label_show": r"$\Sigma_{noise}$=0.001",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_001",
     "loss_list": [],
     "dashes": None,
     "color": "k",
     "alpha": 0.5},
    {"label_show": r"$\Sigma_{noise}$=0.0025",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_0025",
     "loss_list": [],
     "dashes": None,
     "color": "k",
     "alpha": 0.5},
    {"label_show": r"$\Sigma_{noise}$=0.005",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_005",
     "loss_list": [],
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\6_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\6_dpf\data_synthetic_fish_all.hdf5",
     # None  #
     "dashes": None,
     "color": "k",
     "alpha": 0.5},
    {"label_show": r"$\Sigma_{noise}$=0.01",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_01",
     "loss_list": [],
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_synthetic_fish_all.hdf5",
     # None  #
     "dashes": (2, 4),
     "color": "k",
     "alpha": 1},
    {"label_show": r"$\Sigma_{noise}$=0.05",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_05",
     "loss_list": [],
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_synthetic_fish_all.hdf5",
     # None  #
     "dashes": (2, 4),
     "color": "k",
     "alpha": 1},
    {"label_show": r"$\Sigma_{noise}$=0.1",
     "path": fr"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/test_loss/sensitivity/{loss_label}/0_1",
     "loss_list": [],
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2\7_dpf\data_synthetic_fish_all.hdf5",
     # None  #
     "dashes": None,
     "color": "k",
     "alpha": 1},
]

df_data_sample = pd.read_hdf(path_data_sample)

# for m in models_in_group_list:
#     try:
#         m["df_data"] = pd.read_hdf(m["path_data"])
#         m["df_simulation"] = pd.read_hdf(m["path_simulation"])
#         m["df_simulation"].reset_index(inplace=True)
#
#         sequential_index = pd.Index(range(0, len(m["df_data"])), name="Seq_Index")
#         m["df_data"].index = pd.MultiIndex.from_arrays(
#             [sequential_index] + [m["df_data"].index.get_level_values(i) for i in range(m["df_data"].index.nlevels)],
#             names=['Seq_Index'] + m["df_data"].index.names)
#         m["df_data"]["same_direction_as_previous_bout"] = [1 if s else 0 for s in m["df_data"]["same_as_previous"]]
#
#         df_data_unindexed = m["df_data"].reset_index(allow_duplicates=True)
#         same_as_previous = [0]
#         same_as_previous.extend(
#             [1 if df_data_unindexed.loc[i_row, CorrectBoutColumn] == df_data_unindexed.loc[
#                 i_row - 1, CorrectBoutColumn] else 0 for
#              i_row in
#              range(1, len(df_data_unindexed))])
#         m["df_data"]["same_direction_as_previous_bout"] = same_as_previous
#
#         same_as_previous = [0]
#         same_as_previous.extend(
#             [1 if m["df_simulation"].loc[i_row, CorrectBoutColumn] == m["df_simulation"].loc[i_row - 1, CorrectBoutColumn] else 0 for
#              i_row in
#              range(1, len(m["df_simulation"]))])
#         m["df_simulation"]["same_direction_as_previous_bout"] = same_as_previous
#     except (KeyError, NotImplementedError):
#         print(f"No data for group {m['label_show']}")
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

# fetch data
df = pd.DataFrame()
model_list = []
loss = {}
parameter_error = {p["label"]: {} for p in parameter_list}
label_fish_time =None
max_n_models = 50
number_bins_hist = 15

show_sample_ibi_distribution = False
show_objective_function_vs_iterations = True
show_distribution_parameters = False
show_loss_reduction = False
show_perturbation_vs_laplace_scale = False
show_distribution_loss_real = False
show_loss_real = False

distribution_fit_dict = {data['label_show']: {p["label"]: {"loc": [], "scale": [], "rmse": []} for p in parameter_list}
                         for data in models_in_group_list}
for i_group, models_in_group in enumerate(models_in_group_list):
    noise_scale = float(models_in_group["label_show"].split("=")[-1])

    if show_sample_ibi_distribution:
        coh_to_show = 50
        x_limits = [0, 2]
        y_limits = [0, 0.15]
        plot_height_small = plot_height / 2
        padding_plot = 0.5
        padding_vertical = plot_height_small
        
        title = f"{models_in_group["label_show"]}"
        plot_dist = fig.create_plot(plot_title=title,
                                    plot_label=alphabet[i_plot_label] if i_group == 0 else None,
                                    xpos=xpos, ypos=ypos,
                                    plot_height=plot_height,
                                    plot_width=plot_width,
                                    xmin=x_limits[0], xmax=x_limits[-1],
                                    xticks=None, yticks=None,
                                    ymin=-y_limits[-1], ymax=y_limits[-1],
                                    hlines=[0])

        # scalebar
        if i_group == len(models_in_group_list)-2:
            y_location_scalebar = y_limits[-1] / 6
            x_location_scalebar = x_limits[-1] / 6
            plot_dist.draw_line((1.7, 1.7), (y_location_scalebar, y_location_scalebar + 0.1), lc="k")
            plot_dist.draw_text(2, y_location_scalebar, "0.1 events/s",
                                                    textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

            plot_dist.draw_line((x_location_scalebar, x_location_scalebar + 0.5), (-y_location_scalebar, -y_location_scalebar), lc="k")
            plot_dist.draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                    textlabel_rotation='horizontal', textlabel_ha='left', textlabel_va="bottom")

        df_filtered = df_data_sample[df_data_sample[analysed_parameter] == coh_to_show]
        df_filtered = df_filtered.query(query_time)

        duration = np.sum(
            BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=time_end_stimulus - time_start_stimulus)
        )

        # plot distribution of data over coherence levels
        data_corr = df_filtered[df_filtered[CorrectBoutColumn] == 1][ResponseTimeColumn]
        data_err = df_filtered[df_filtered[CorrectBoutColumn] == 0][ResponseTimeColumn]

        data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                               # bins=100,
                                                                               bins=np.arange(x_limits[0],
                                                                                              x_limits[-1], (
                                                                                                      x_limits[-1] -
                                                                                                      x_limits[
                                                                                                          0]) / 50),
                                                                               duration=duration,
                                                                               center_bin=True)
        index_in_limits = np.argwhere(
            np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
        data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
        data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

        data_hist_value_corr += np.random.normal(scale=noise_scale, size=len(data_hist_value_corr))
        data_hist_value_corr[data_hist_value_corr <= 0] = np.finfo(float).eps

        data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                             # bins=100,
                                                                             bins=np.arange(x_limits[0],
                                                                                            x_limits[-1], (
                                                                                                    x_limits[-1] -
                                                                                                    x_limits[
                                                                                                        0]) / 50),
                                                                             duration=duration,
                                                                             center_bin=True)
        index_in_limits = np.argwhere(
            np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
        data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
        data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

        data_hist_value_err += np.random.normal(scale=noise_scale, size=len(data_hist_value_err))
        data_hist_value_err[data_hist_value_err <= 0] = np.finfo(float).eps

        plot_dist.draw_line(data_hist_time_corr, data_hist_value_corr, lc=style.palette["correct_incorrect"][0], lw=0.75)
        plot_dist.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=style.palette["correct_incorrect"][1], lw=0.75)

        ypos -= (padding + plot_height)
        # xpos += (padding + plot_width)

        if i_group == 0:
            i_plot_label += 1

    if show_objective_function_vs_iterations:
        ypos_start_here = ypos
        plot_width_a = 1
        plot_width_bc = 0.5
        padding_plot = 0.5
        df = pd.DataFrame()
        model_list = []
        loss = []
        parameter_error = {p["label"]: {} for p in parameter_list}
        parameter_error_trajectory_list_dict = {p["label"]: [] for p in parameter_list}
        present_fitting_run = 0
        for i_error, error_path in enumerate(pathlib.Path(models_in_group["path"]).glob("error_test_*_fit.hdf5")):
            if error_path.is_dir():
                continue

            df_error = pd.read_hdf(str(error_path))
            if len(df_error) < 3:
                continue

            for index_parameter, parameter in enumerate(parameter_list):
                parameter_error_trajectory_list_dict[parameter['label']].append(np.array(df_error[f"{parameter['label']}_error"]))

            loss.append(np.array(df_error["score"]))

        parameter_error_trajectory_list_dict["score"] = np.array(loss)
        for p in parameter_list:
            parameter_error_trajectory_list_dict[p['label']] = np.array(parameter_error_trajectory_list_dict[p['label']])

        loss_trajectory_list = parameter_error_trajectory_list_dict["score"]
        loss_mean = np.array([np.mean(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
        loss_std = np.array([np.std(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
        iteration_list = np.arange(len(loss_mean))

        parameter_error_mean = {}
        parameter_error_std = {}

        parameter_error_mean["score"] = loss
        for index_parameter, parameter in enumerate(parameter_list):
            parameter_error_mean[parameter["label"]] = np.array([np.mean(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])
            parameter_error_std[parameter["label"]] = np.array([np.std(parameter_error_trajectory_list_dict[parameter["label"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["label"]].shape[1])])

        error_array_end = parameter_error_trajectory_list_dict["score"][:, -1]
        # data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(0, 3, 0.1), density=True, center_bin=True)
        data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(0, 3, 0.05), density=True, center_bin=True)

        # title = fr"{np.mean(error_array_end):.02f}$\pm${np.std(error_array_end):.02f}"
        # if noise_scale == 0.001:
        #     title += "\n~5dpf group"
        # elif noise_scale == 0.005:
        #     title += "\n~9dpf group"
        # else:
        #     title += "\n"

        plot_0c = fig.create_plot(plot_label=alphabet[i_plot_label] if i_group == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                  errorbar_area=False, ymin=0, ymax=3,
                                  yticks=[0, 1.5, 3] if i_group == 0 else None,
                                  yl=f"Loss at\niteration 1500" if i_group == 0 else None,
                                  xl=None, xmin=0, xmax=50, xticks=None)
        plot_0c.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=Palette.color_neutral, elw=1)
        print(fr"{models_in_group['label_show']} | loss | {np.mean(error_array_end):.04f}$\pm${np.std(error_array_end):.04f}")

        # xpos += padding + plot_width
        # ypos += padding + plot_height
        ypos = ypos - padding - plot_height
        if i_group == 0:
            i_plot_label += 1

        df_noise_dict = {p["label"]: {} for p in parameter_list}
        p_final_pdf = {p["label"]: None for p in parameter_list}

        # plotting
        for index_parameter, parameter in enumerate(parameter_list):
            error_array_end = parameter_error_trajectory_list_dict[parameter["label"]][:, -1]
            data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)

            plot_nc = fig.create_plot(plot_label=alphabet[i_plot_label] if i_group == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                      errorbar_area=False, ymin=-1, ymax=1,
                                      yticks=[-1, 0, 1] if i_group == 0 else None,
                                      yl=f"Estimation error at iteration 1500" if (index_parameter == 0 and i_group == 0) else None,
                                      xl="Percentage models (%)" if (index_parameter == len(parameter_list)-1 and i_group == 0) else None,
                                      xmin=0, xmax=50, xticks=[0, 25, 50] if index_parameter == len(parameter_list)-1 else None, hlines=[0])
            plot_nc.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=palette[index_parameter], elw=1)


            q = np.quantile(parameter_error_trajectory_list_dict[parameter["label"]][:, -1], q=[0.25, 0.75])
            q = q / (parameter["max"] - parameter["min"])
            print(f"{models_in_group['label_show']} | {parameter['label_show']} | 90% of the error is between {100*q[0]:.04f} and {100*q[1]:.04f}%")

            # threshold_low_error_estimation = 0.2
            # all_error_estimations = copy.deepcopy(parameter_error_trajectory_list_dict[parameter["label"]][:, -1])
            # low_error_estimations = np.array(all_error_estimations)
            # low_error_estimations = low_error_estimations[np.abs(low_error_estimations) < threshold_low_error_estimation]
            # print(f"{models_in_group['label_show']} | {parameter['label_show']} | {100*(len(low_error_estimations) / len(all_error_estimations)):.04f}% absolute error is below {100*threshold_low_error_estimation}:.01f%")

            if fit_distributions:
                mu, b = laplace.fit(error_array_end)
                y_dist = laplace.pdf(data_hist_time_end, mu, b)
                # y_dist = normpdf(data_hist_time_end, mu, std)
                y_dist /= np.sum(y_dist)

                rmse_laplace = np.sqrt(np.mean((data_hist_value_end - y_dist) ** 2))
                print(f"RMSE Laplace = {rmse_laplace}")
                distribution_fit_dict[models_in_group['label_show']][parameter["label"]]["loc"].append(mu)
                distribution_fit_dict[models_in_group['label_show']][parameter["label"]]["scale"].append(b)
                distribution_fit_dict[models_in_group['label_show']][parameter["label"]]["rmse"].append(rmse_laplace)

                plot_nc.draw_line(y_dist * 100, data_hist_time_end, lc="k", elw=0.3, line_dashes=(2, 4), alpha=1)
            ypos = ypos - padding - plot_height

            if i_group == 0:
                i_plot_label += 1

        if fit_distributions:
            df_noise = pd.DataFrame(df_noise_dict)
            df_noise.to_hdf(str(pathlib.Path.home() / 'Desktop' / f"df_noise_sensitivity_{loss_label}_{models_in_group['path'].split('\\')[-1]}"), key="estimation_error")


        if i_group == len(models_in_group_list)-1:
            xpos = xpos_start
            ypos -= plot_height
        else:
            xpos = xpos + plot_width_bc + padding
            ypos = ypos_start

    if i_group == len(models_in_group_list)-1:
        xpos = xpos_start
        ypos -= (padding + plot_height)

if show_distribution_parameters:
    from_best_model = True
    plot_height_here = 0.25
    padding_here = 0.05

    distribution_trajectory_dict = {p["label"]: np.zeros((number_bins_hist, len(models_in_group_list))) for p in parameter_list}
    raw_data_dict_per_fish = {p["label"]: {i_age: {} for i_age in range(len(models_in_group_list))} for p in parameter_list}
    raw_data_dict = {p["label"]: {i_age: [] for i_age in range(len(models_in_group_list))} for p in parameter_list}

    for i_age, models_in_age in enumerate(models_in_group_list):
        model_dict = {}
        n_models = 0
        path_dir = pathlib.Path(models_in_age["path"])
        for model_filepath in path_dir.glob('model_*_fit.hdf5'):
            # if n_models > max_n_models:
            #     break
            # n_models += 1
            model_filename = str(model_filepath.name)
            if label_fish_time is not None:
                if model_filename.split("_")[2].endswith(label_fish_time):
                    model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}
            else:
                model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

        fish_list = np.arange(len(model_dict.keys()))

        model_parameter_median_dict = {p["label"]: {} for p in parameter_list}
        model_parameter_dict = {p["label"]: {} for p in parameter_list}
        model_parameter_median_dict["score"] = {}
        model_parameter_dict["score"] = {}
        model_parameter_median_array = np.zeros((len(parameter_list)+1, len(fish_list)))
        for i_model, id_model in enumerate(model_dict.keys()):
            df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
            # id_fish = id_model[:2]

            id_fish = i_model
            if from_best_model:
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit_list = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

            model_parameter_median_dict["score"][id_fish] = np.median(df_model_fit_list["score"])
            model_parameter_dict["score"][id_fish] = np.array(df_model_fit_list["score"])
            model_parameter_median_array[0, i_model] = np.median(df_model_fit_list["score"])

            for i_p, p in enumerate(parameter_list):
                p_median = np.median(df_model_fit_list[p["label"]])
                model_parameter_median_dict[p["label"]][id_fish] = p_median  # (p_median - p["min"]) / (p["max"] - p["min"])
                model_parameter_dict[p["label"]][id_fish] = np.array(df_model_fit_list[p["label"]])  # (np.array(df_model_fit_list[p["label"]]) - p["min"]) / (p["max"] - p["min"])
                model_parameter_median_array[i_p+1, i_model] = p_median

                if id_model not in raw_data_dict_per_fish[p["label"]][i_age].keys():
                    raw_data_dict_per_fish[p["label"]][i_age][id_model] = [p_median]
                else:
                    raw_data_dict_per_fish[p["label"]][i_age][id_model].append(p_median)

                raw_data_dict[p["label"]][i_age].append(p_median)

        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[score_config["label"]], bin_model_parameter_median_dict[score_config["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[0, :], center_bin=True,  hist_range=[score_config["min"], score_config["max"]],
                bins=number_bins_hist,  # int((score_config["max"] - score_config["min"])/0.1),
                density=True
            )
        for i_p, p in enumerate(parameter_list):
            hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[p["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :], center_bin=True,  hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,  # int((p["max"]-p["min"])/0.1),
                density=True
            )

            distribution_trajectory_dict[p["label"]][:, i_age] = hist_model_parameter_median_dict[p["label"]]

    for i_age in range(len(models_in_group_list)):
        for i_p, p in enumerate(parameter_list):
            plot_n = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 and i_age == 0 else None, xpos=xpos, ypos=ypos,
                                     plot_height=plot_height_here, plot_width=plot_width,
                                     yl="Percentage fish (%)" if i_p == 0 and i_age == 0 else None, ymin=0, ymax=50, yticks=[0, 50] if i_p == 0 and i_age == 0 else None,
                                     xl=p['label_show'].capitalize() if i_age == len(models_in_group_list)-1 else None, xmin=p["min"], xmax=p["max"],
                                     xticks=[p["min"], p["mean"], p["max"]] if i_age == len(models_in_group_list)-1 else None, vlines=[p["mean"]] if p["mean"] == 0 else [])

            plot_n.draw_line(bin_model_parameter_median_dict[p["label"]], distribution_trajectory_dict[p["label"]][:, i_age] * 100,
                             line_dashes=models_in_group_list[i_age]["dashes"], lc=palette[i_p], alpha=models_in_group_list[i_age]["alpha"], label=models_in_group_list[i_age]["label_show"] if i_p == len(parameter_list)-1 else None)

            # i_plot_label += 1
            xpos = xpos + padding_short + plot_width

        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here
    i_plot_label += 1

    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_group_list)), 2))
    for pairs_to_compare in pairs_to_compare_list:
        for i_p, p in enumerate(parameter_list):
            data_0 = raw_data_dict[p["label"]][pairs_to_compare[0]]
            data_1 = raw_data_dict[p["label"]][pairs_to_compare[1]]

            U1, p_value = mannwhitneyu(data_0, data_1, method="exact")

            print(f"{models_in_group_list[pairs_to_compare[0]]['label_show']} vs {models_in_group_list[pairs_to_compare[1]]['label_show']} | {p['label_show']} | parameter p value {p_value}")

    xpos = xpos_start
    ypos = ypos - (padding-padding_here)
    # i_plot_label = len(parameter_list)

if show_loss_reduction:
    ypos = ypos - padding_short
    for i_age, models_in_age in enumerate(models_in_group_list):
        plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label] if i_age == 0 else None,
                                    plot_title=models_in_age["label_show"],
                                    xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                 ymin=0, ymax=20, yticks=[0, 10, 20] if i_age == 0 else None, yl=f"loss" if i_age == 0 else None,
                                 xl=None, xmin=0.5, xmax=2.5, xticks=[1, 2],
                                 xticklabels=["iteration 0", "iteration 1500"], xticklabels_rotation=45)
        ypos = ypos
        xpos += plot_width + padding_short

        loss_list = []

        for path_error in pathlib.Path(models_in_age["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_start = df_error["score"][0]
            loss_end = df_error["score"][len(df_error)-1]
            loss_list.append(loss_end)
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

        plot_loss.draw_text(2.5, 10, fr"{np.mean(loss_list):0.02f}$\pm${np.std(loss_list):0.02f}")
    ypos = ypos - padding - plot_height
    xpos = xpos_start
    i_plot_label += 1

if show_perturbation_vs_laplace_scale:
    ypos = ypos - padding_short
    plot_height_here = 1
    padding_here = 0.5

    perturbation_value_list = list([data["label_show"].split("=")[-1] for data in models_in_group_list])
    perturbation_x = np.arange(len(perturbation_value_list)) + 1
    for i_p, p in enumerate(parameter_list):
        loc_array = np.abs([distribution_fit_dict[data["label_show"]][p["label"]]["loc"] for data in models_in_group_list])
        scale_array = np.array([distribution_fit_dict[data["label_show"]][p["label"]]["scale"] for data in models_in_group_list])
        rmse_array = np.array([distribution_fit_dict[data["label_show"]][p["label"]]["rmse"] for data in models_in_group_list])

        plot_p = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 else None,
                                 plot_title=p["label_show"].capitalize(),
                                 xpos=xpos, ypos=ypos,
                                 plot_height=plot_height_here, plot_width=plot_width,
                                 ymin=0, ymax=0.6,
                                 yticks=[0, 0.3, 0.6] if i_p == 0 else None,
                                 xl=r"$\Sigma_{noise}$",  # p['label'] if i_age == len(models_in_group_list) - 1 else None,
                                 # xmin=p["min"], xmax=p["max"],
                                 xticks=perturbation_x,
                                 xticklabels=perturbation_value_list, xticklabels_rotation=45)

        plot_p.draw_line(perturbation_x, loc_array, lc=palette[i_p], alpha=0.8, line_dashes=(0.1, 3), label="Loc" if i_p == len(parameter_list)-1 else None)
        plot_p.draw_line(perturbation_x, scale_array, lc=palette[i_p], alpha=0.8, line_dashes=(2, 4), label="Scale" if i_p == len(parameter_list)-1 else None)
        plot_p.draw_line(perturbation_x, rmse_array, lc=palette[i_p], alpha=0.8, line_dashes=None, label="RMSE" if i_p == len(parameter_list)-1 else None)

        xpos = xpos + plot_width_bc + padding

    i_plot_label += 1
    xpos = xpos_start
    ypos = ypos - plot_height_here - padding

if show_distribution_loss_real:
    loss_list = []
    for model_filepath in pathlib.Path(path_real_data).glob('model_*_fit.hdf5'):
        df_model = pd.read_hdf(model_filepath)
        loss_list.append(df_model["score"].iloc[0])
    loss_list = np.array(loss_list)

    loss_hist, loss_bins = StatisticsService.get_hist(loss_list,
                                                       bins=np.arange(0, 3, 0.05),
                                                       density=True,
                                                       center_bin=True)

    plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label],
                                 xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width,
                                 ymin=0, ymax=3,
                                 yticks=[0, 1.5, 3], yl="Loss at\niteration 1500",
                                 xl=r"Percentage models (%)",
                                 xmin=0, xmax=50, xticks=[0, 25, 50])
    i_plot_label += 1
    ypos -= padding + plot_height
    plot_loss.draw_line(loss_hist * 100, loss_bins, lc=Palette.color_neutral, elw=1)

if show_loss_real:
    models_real_in_group_list = [
        {"label_show": "5dpf",
         "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf",
         "loss_list": []},
        {"label_show": "6dpf",
         "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\6_dpf",
         "loss_list": []},
        {"label_show": "7dpf",
         "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\7_dpf",
         "loss_list": []},
        {"label_show": "8dpf",
         "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\8_dpf",
         "loss_list": []},
        {"label_show": "9dpf",
         "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\9_dpf",
         "loss_list": []},
        {"label_show": "scn1lab +/+",
         "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\wt",
         "loss_list": []},
        {"label_show": "scn1lab +/-",
         "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\het",
         "loss_list": []},
        {"label_show": "disc +/+",
         "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\wt",
         "loss_list": []},
        {"label_show": "disc +/-",
         "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\het",
         "loss_list": []},
        {"label_show": "disc -/-",
         "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\hom",
         "loss_list": []},
    ]

    ypos -= plot_height * 4

    x_ticks = np.arange(len(models_real_in_group_list)+len(models_in_group_list))
    x_tick_labels = [m["label_show"] for m in models_in_group_list] + [m["label_show"] for m in models_real_in_group_list]
    plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label],
                                xpos=xpos, ypos=ypos,
                                plot_height=plot_height * 4, plot_width=plot_width * 8,
                                ymin=0, ymax=3, errorbar_area=False,
                                yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], yl="Loss at\niteration 1500",
                                xl=None, xmin=-0.5, xmax=len(x_ticks) - 0.5,
                                xticks=x_ticks, xticklabels=x_tick_labels, xticklabels_rotation=45,)

    for i_m, m in enumerate(models_in_group_list):
        path = m["path"]

        for i_model, model_path in enumerate(pathlib.Path(path).glob("model_*_fit.hdf5")):
            if model_path.is_dir():
                continue

            df_m = pd.read_hdf(str(model_path))
            m["loss_list"].append(np.median(df_m["score"]))

        plot_loss.draw_scatter(x_ticks[i_m], np.mean(m["loss_list"]), yerr=np.std(m["loss_list"]))

    x_offset = len(models_in_group_list)
    for i_m_real, m_real in enumerate(models_real_in_group_list):
        path_real = m_real["path"]

        for i_model, model_path in enumerate(pathlib.Path(path_real).glob("model_*_fit.hdf5")):
            if model_path.is_dir():
                continue

            df_m_real = pd.read_hdf(str(model_path))
            m_real["loss_list"].append(np.median(df_m_real["score"]))

        plot_loss.draw_scatter(x_offset + x_ticks[i_m_real], np.mean(m_real["loss_list"]), yerr=np.std(m_real["loss_list"]))

fig.save(pathlib.Path.home() / 'Desktop' / f"figure_s4_sensitivity_loss_{loss_label}.pdf", open_file=False, tight=True)


