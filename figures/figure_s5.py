import itertools

import matplotlib
# matplotlib.use("macosx")
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import scipy
import pathlib

from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

from analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import palette_0, color_list, palette_long, \
    alphabet, StimulusParameterLabel, ResponseTimeColumn, CorrectBoutColumn
from analysis_helpers.analysis.utils.figure_helper import Figure

# plot configs
fig = Figure()
plot_height = 1
plot_width = 1
xpos_start = 0.5
ypos_start = 0.5
xpos = xpos_start
ypos = ypos_start
padding = 1.8
padding_short = 0.75
i_plot_label = 0
palette = Palette.arlecchino
analysed_parameter_label = "coherence [%]"
index_population = 1

# experimental configs
time_start_stimulus = 10
time_end_stimulus = 20
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = [0, 25, 50, 100]

score_config = {
    "label": "loss",
    "min": 0,
    "max": 10
}
parameter_list = [
    {"param": "noise_sigma",
     "label": "diffusion",
     "min": 0.0,
     "mean": 1.5,
     "max": 3.0},
    {"param": 'scaling_factor',
     "label": "drift",
     "min": -3,
     "mean": 0,
     "max": 3},
    {"param": 'leak',
     "label": "leak",
     "min": -3,
     "mean": 0,
     "max": 3},
    {"param": 'residual_after_bout',
     "label": "reset",
     "min": 0.0,
     "mean": 0.5,
     "max": 1.0},
    {"param": 'inactive_time',
     "label": "delay",
     "min": 0.0,
     "mean": 0.5,
     "max": 1.0},
]

models_in_age_list = [
    {"label_fish_time": "-wt",
     "label_show": "wt",
     "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\wt",
     "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\wt\data_fish_all-wt.hdf5",
     "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\wt\data_synthetic_fish_wt_merged.hdf5",
     "dashes": None,
     "color": "k",
     "alpha": 1},
    {"label_fish_time": "-hom",
     "label_show": "disc -/-",
     "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\hom",
     "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\hom\data_fish_all-hom.hdf5",
     "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_1\hom\data_synthetic_fish_hom_merged.hdf5",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
]

for m in models_in_age_list:
    try:
        m["df_data"] = pd.read_hdf(m["path_data"])
        m["df_simulation"] = pd.read_hdf(m["path_simulation"])
        m["df_simulation"].reset_index(inplace=True)

        sequential_index = pd.Index(range(0, len(m["df_data"])), name="Seq_Index")
        m["df_data"].index = pd.MultiIndex.from_arrays(
            [sequential_index] + [m["df_data"].index.get_level_values(i) for i in range(m["df_data"].index.nlevels)],
            names=['Seq_Index'] + m["df_data"].index.names)
        m["df_data"]["same_direction_as_previous_bout"] = [1 if s else 0 for s in m["df_data"]["same_as_previous"]]

        df_data_unindexed = m["df_data"].reset_index(allow_duplicates=True)
        same_as_previous = [0]
        same_as_previous.extend(
            [1 if df_data_unindexed.loc[i_row, CorrectBoutColumn] == df_data_unindexed.loc[
                i_row - 1, CorrectBoutColumn] else 0 for
             i_row in
             range(1, len(df_data_unindexed))])
        m["df_data"]["same_direction_as_previous_bout"] = same_as_previous

        same_as_previous = [0]
        same_as_previous.extend(
            [1 if m["df_simulation"].loc[i_row, CorrectBoutColumn] == m["df_simulation"].loc[i_row - 1, CorrectBoutColumn] else 0 for
             i_row in
             range(1, len(m["df_simulation"]))])
        m["df_simulation"]["same_direction_as_previous_bout"] = same_as_previous
    except (KeyError, NotImplementedError):
        print(f"No data for group {m['label_show']}")
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

# fetch data
df = pd.DataFrame()
model_list = []
loss = {}
parameter_error = {p["param"]: {} for p in parameter_list}
label_fish_time =None
max_n_models = 50
number_bins_hist = 15

show_loss_reduction = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_distribution_parameters = True
show_same_dir_coh_0 = False


if show_loss_reduction:
    plot_width_here = 1.3
    ypos = ypos - padding_short
    for i_age, models_in_age in enumerate(models_in_age_list):
        plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label] if i_age == 0 else None,
                                    plot_title=models_in_age["label_show"],
                                    xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_here,
                                 ymin=0, ymax=20, yticks=[0, 10, 20] if i_age == 0 else None, yl=f"loss" if i_age == 0 else None,
                                 xl=None, xmin=0.5, xmax=2.5, xticks=[1, 2],
                                 xticklabels=["iteration 0", "iteration 1500"], xticklabels_rotation=45)
        ypos = ypos
        if i_age == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width_here

        loss_list = []

        for path_error in pathlib.Path(models_in_age["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_start = df_error["score"][0]
            loss_end = df_error["score"][len(df_error)-1]
            loss_list.append(loss_end)
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

        plot_loss.draw_text(2.5, 10, fr"{np.mean(loss_list):0.02f}$\pm${np.std(loss_list):0.02f}")
    # ypos = ypos
    # xpos += plot_width + padding_short
    i_plot_label += 1

if show_psychometric_curve:
    # plot_height = plot_height
    plot_width_here = 1.3

    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[analysed_parameter].isin(analysed_parameter_list)]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, correct_bout_list_data, std_correct_bout_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=analysed_parameter)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, correct_bout_list_sim, std_correct_bout_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(df_simulation_filtered, analysed_parameter=analysed_parameter)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width_here,
                                 errorbar_area=True,
                                 xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list), xticks=[int(p) for p in analysed_parameter_list],
                                 yl="accuracy [-]" if i_m == 0 else None, ymin=0, ymax=1, yticks=[0, 0.5, 1] if i_m == 0 else None, hlines=[0.5])

        # draw
        plot_0.draw_line(x=parameter_list_data, y=correct_bout_list_data,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data),  # / np.sqrt(number_individuals),
                         lc=m["color"], lw=1, alpha=m["alpha"])
        plot_0.draw_line(x=parameter_list_sim, y=correct_bout_list_sim,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim),  # / np.sqrt(number_models),
                         lc=m["color"], lw=1, alpha=m["alpha"], line_dashes=(1, 2))

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width_here

    i_plot_label += 1
    # ypos = ypos - padding - plot_height
    # xpos = xpos_start

if show_coherence_vs_interbout_interval:
    # plot_height = 1
    # plot_width = 1
    plot_width_here = 1.3

    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[analysed_parameter].isin(analysed_parameter_list)]
        df_data_filtered = df_data_filtered[df_data_filtered[CorrectBoutColumn] != -1]
        ibi_quantiles = np.quantile(df_data_filtered[ResponseTimeColumn], [0.05, 0.95])
        df_data_filtered = df_data_filtered[np.logical_and(df_data_filtered[ResponseTimeColumn] > ibi_quantiles[0], df_data_filtered[ResponseTimeColumn] < ibi_quantiles[1])]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, interbout_interval_list_data, std_interbout_interval_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, interbout_interval_list_sim, std_interbout_interval_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
                df_simulation_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # plot
        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width_here,
                                 errorbar_area=True,
                                 xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list), xticks=[int(p) for p in analysed_parameter_list],
                                 yl="interbout interval [s]" if i_m == 0 else None, ymin=0, ymax=3, yticks=[0, 1.50, 3] if i_m == 0 else None)
        # i_plot_label += 1

        # draw
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),  # / np.sqrt(number_individuals),
                         lc=m["color"], lw=1, alpha=m["alpha"], label=f"data" if i_m == 0 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),  # / np.sqrt(number_models),
                         lc=m["color"], lw=1, alpha=m["alpha"], line_dashes=(1, 2), label=f"simulation" if i_m == 0 else None)

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width_here

    i_plot_label += 1
    ypos = ypos - padding - plot_height
    xpos = xpos_start

if show_distribution_parameters:
    from_best_model = True
    plot_height_here = 0.25
    plot_width_here = 2
    padding_here = 0.05
    path_noise_statistics = r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth_2coh\results\df_noise"
    noise_statistics = pd.read_hdf(path_noise_statistics)
    number_resampling = 10000

    distribution_trajectory_dict = {p["param"]: np.zeros((number_bins_hist, len(models_in_age_list))) for p in parameter_list}
    raw_data_dict_per_fish = {p["param"]: {i_age: {} for i_age in range(len(models_in_age_list))} for p in parameter_list}
    raw_data_dict = {p["param"]: {i_age: [] for i_age in range(len(models_in_age_list))} for p in parameter_list}
    quantiles_groups = {p["param"]: np.zeros((len(models_in_age_list), 3)) for p in parameter_list}
    for i_age, models_in_age in enumerate(models_in_age_list):
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

        model_parameter_median_dict = {p["param"]: {} for p in parameter_list}
        model_parameter_dict = {p["param"]: {} for p in parameter_list}
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
                p_median = np.median(df_model_fit_list[p["param"]])
                model_parameter_median_dict[p["param"]][id_fish] = p_median  # (p_median - p["min"]) / (p["max"] - p["min"])
                model_parameter_dict[p["param"]][id_fish] = np.array(df_model_fit_list[p["param"]])  # (np.array(df_model_fit_list[p["param"]]) - p["min"]) / (p["max"] - p["min"])
                model_parameter_median_array[i_p+1, i_model] = p_median

                if id_model not in raw_data_dict_per_fish[p["param"]][i_age].keys():
                    raw_data_dict_per_fish[p["param"]][i_age][id_model] = [p_median]
                else:
                    raw_data_dict_per_fish[p["param"]][i_age][id_model].append(p_median)

                raw_data_dict[p["param"]][i_age].append(p_median)

        for i_p, p in enumerate(parameter_list):
            noise_statistics_p = dict(noise_statistics[p["param"]])
            noise_statistics_p["mu"] = noise_statistics_p["mu"] * (
                    p["max"] - p["min"])  # rescale mean to actual search space width (it was normalized before)
            noise_statistics_p["b"] = noise_statistics_p["b"] * (
                    p["max"] - p["min"])  # rescale std to actual search space width (it was normalized before)
            noise_statistics_p["size"] = len(model_parameter_median_array[i_p + 1, :])
            # model_parameter_sampling_list = StatisticsService.sample_random(array=model_parameter_median_array[i_p + 1, :], sample_number=number_resampling, sample_percentage_size=1, with_replacement=True, add_noise=noise_statistics_p)
            model_parameter_sampling_list = StatisticsService.sample_random(
                array=model_parameter_median_array[i_p + 1, :], sample_number=number_resampling,
                sample_percentage_size=1, with_replacement=True, add_noise=None)

            median_list = np.array([np.median(m) for m in model_parameter_sampling_list])

            quantiles_groups[p["param"]][i_age, :] = np.quantile(median_list, [0.05, 0.5, 0.95])
            print(
                f"group {models_in_age['label_show']} | {p['label']}: {quantiles_groups[p['param']][i_age, 1]:.05f} [{quantiles_groups[p['param']][i_age, 0]:.05f}, {quantiles_groups[p['param']][i_age, 2]:.05f}]")

        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[score_config["label"]], bin_model_parameter_median_dict[score_config["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[0, :], center_bin=True,  hist_range=[score_config["min"], score_config["max"]],
                bins=number_bins_hist,  # int((score_config["max"] - score_config["min"])/0.1),
                density=True
            )
        for i_p, p in enumerate(parameter_list):
            hist_model_parameter_median_dict[p["param"]], bin_model_parameter_median_dict[p["param"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :], center_bin=True,  hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,  # int((p["max"]-p["min"])/0.1),
                density=True
            )
            distribution_trajectory_dict[p["param"]][:, i_age] = hist_model_parameter_median_dict[p["param"]]

    for i_age in range(len(models_in_age_list)):
        for i_p, p in enumerate(parameter_list):
            plot_n = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 and i_age == 0 else None, xpos=xpos, ypos=ypos,
                                     plot_height=plot_height_here, plot_width=plot_width_here,
                                     yl="percentage fish [%]" if i_p == 0 and i_age == 0 else None, ymin=0, ymax=50, yticks=[0, 50] if i_p == 0 and i_age == 0 else None,
                                     xl=p['label'] if i_age == len(models_in_age_list)-1 else None, xmin=p["min"], xmax=p["max"],
                                     xticks=[p["min"], p["mean"], p["max"]] if i_age == len(models_in_age_list)-1 else None, vlines=[p["mean"]] if p["mean"] == 0 else [])

            plot_n.draw_line(bin_model_parameter_median_dict[p["param"]], distribution_trajectory_dict[p["param"]][:, i_age] * 100,
                             line_dashes=models_in_age_list[i_age]["dashes"], lc=palette[i_p], alpha=models_in_age_list[i_age]["alpha"], label=models_in_age_list[i_age]["label_show"] if i_p == len(parameter_list)-1 else None)

            # i_plot_label += 1
            xpos = xpos + padding_short + plot_width_here

        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here
    i_plot_label += 1

    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_age_list)), 2))
    for pairs_to_compare in pairs_to_compare_list:
        for i_p, p in enumerate(parameter_list):
            data_0 = raw_data_dict[p["param"]][pairs_to_compare[0]]
            data_1 = raw_data_dict[p["param"]][pairs_to_compare[1]]

            U1, p_value = mannwhitneyu(data_0, data_1, method="exact")

            print(f"{models_in_age_list[pairs_to_compare[0]]['label_show']} vs {models_in_age_list[pairs_to_compare[1]]['label_show']} | {p['label']} | parameter p value {p_value}")

    xpos = xpos_start
    ypos = ypos - (padding-padding_here)
    # i_plot_label = len(parameter_list)

    # BOOTSTRAP TEST PLOT
    plot_list = []
    for i_p, p in enumerate(parameter_list):
        plot_n = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width_here, errorbar_area=False,
                                 ymin=0, ymax=len(models_in_age_list) + 1,
                                 yticks=np.arange(len(models_in_age_list), 0, -1) if i_p == 0 else None,
                                 yticklabels=[g["label_show"] for g in models_in_age_list] if i_p == 0 else None,
                                 xl=p['label'], xmin=p["min"], xmax=p["max"],
                                 xticks=[p["min"], p["mean"], p["max"]], vlines=[p["mean"]] if p["mean"] == 0 else [])
        xpos = xpos + padding_short + plot_width_here
        for i_group, models_in_group in enumerate(models_in_age_list):
            plot_n.draw_line(quantiles_groups[p["param"]][i_group, :],
                             np.ones(quantiles_groups[p["param"]].shape[1]) * (len(models_in_age_list) - i_group),
                             lc=palette[i_p])
        plot_list.append(plot_n)

    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_age_list)), 2))
    for i_p, p in enumerate(parameter_list):
        parameter_range = p["max"] - p["min"]
        x_sig = p["max"] - parameter_range / 2 + parameter_range / 10
        is_sig = False
        for pairs_to_compare in pairs_to_compare_list:
            data_0 = quantiles_groups[p["param"]][pairs_to_compare[0], :]
            data_1 = quantiles_groups[p["param"]][pairs_to_compare[1], :]

            max_start = np.max([data_0[0], data_1[0]])
            min_end = np.min([data_0[-1], data_1[-1]])

            overlap = min_end - max_start

            print(
                f"{models_in_age_list[pairs_to_compare[0]]['label_show']} vs {models_in_age_list[pairs_to_compare[1]]['label_show']} | {p['label']} | overlap: {overlap}")

            plot_n = plot_list[i_p]
            if overlap < 0:
                is_sig = True
                x_sig_array = np.ones(2) * x_sig
                y_sig_array = np.array(
                    (len(models_in_age_list) - pairs_to_compare[0], len(models_in_age_list) - pairs_to_compare[1]))
                plot_n.draw_line(x_sig_array, y_sig_array, lc="k")

                x_sig += parameter_range / 10

        if is_sig:
            plot_n.draw_text(x=x_sig + parameter_range / 6, y=np.ceil(len(models_in_age_list) / 2), text="*",
                             textlabel_rotation=90)
    i_plot_label += 1
    xpos = xpos_start
    ypos = ypos - (plot_height + padding)
    # i_plot_label = len(parameter_list)


if show_same_dir_coh_0:
    # plot_height = 1
    # plot_width = 1
    analysed_parameter_list = [0]

    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[analysed_parameter].isin(analysed_parameter_list)]
        df_data_filtered = df_data_filtered[df_data_filtered[CorrectBoutColumn] != -1]
        # ibi_quantiles = np.quantile(df_data_filtered[ResponseTimeColumn], [0.05, 0.95])
        # df_data_filtered = df_data_filtered[np.logical_and(df_data_filtered[ResponseTimeColumn] > ibi_quantiles[0],
        #                                                    df_data_filtered[ResponseTimeColumn] < ibi_quantiles[1])]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        number_individuals = 16  # len(df_data_filtered["experiment_ID"].unique())

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[
            df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        # ibi_quantiles = np.quantile(df_simulation_filtered[ResponseTimeColumn], [0.05, 0.95])
        # df_simulation_filtered = df_simulation_filtered[np.logical_and(df_simulation_filtered[ResponseTimeColumn] > ibi_quantiles[0],
        #                                                    df_simulation_filtered[ResponseTimeColumn] < ibi_quantiles[1])]

        windowed_data, time_stamp_list_data, sem_list_data = BehavioralProcessing.windowing_column(
            df_data_filtered[df_data_filtered[analysed_parameter] == 0],
            'same_direction_as_previous_bout',
            window_step_size=0.5,
            window_operation="mean_multiple_fish",
            time_column=ResponseTimeColumn)

        windowed_sim, time_stamp_list_sim, sem_list_sim = BehavioralProcessing.windowing_column(
            df_simulation_filtered[df_simulation_filtered[analysed_parameter] == 0],
            'same_direction_as_previous_bout',
            window_step_size=0.5,
            window_operation="mean_multiple_fish",
            time_column=ResponseTimeColumn)

        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos,
                                 plot_height=plot_height,plot_width=plot_width,
                                 errorbar_area=True,
                                 xl="interbout interval [s]", xmin=0, xmax=1.5,
                                 xticks=[0, 0.75, 1.5], yl='same direction as previous bout [%]' if i_m == 0 else None,
                                 # ymin=0.5, ymax=1.5, yticks=[0.5, 1, 1.5])
                                 ymin=40, ymax=70, yticks=[40, 55, 70] if i_m == 0 else [])
        i_plot_label += 1

        # plot
        plot_0.draw_line(x=time_stamp_list_data, y=windowed_data * 100,
                         errorbar_area=True, yerr=sem_list_data,
                         lc=m["color"], lw=1, alpha=m["alpha"])
        plot_0.draw_line(x=time_stamp_list_sim, y=windowed_sim * 100,
                         errorbar_area=True, yerr=sem_list_sim / number_individuals,
                         lc=m["color"], lw=1, alpha=m["alpha"], line_dashes=(1, 2))

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width

    ypos = ypos - padding - plot_height
    xpos = xpos_start

fig.save(pathlib.Path.home() / 'Desktop' / f"figure_s5_mutation_disc.pdf", open_file=True, tight=True)


