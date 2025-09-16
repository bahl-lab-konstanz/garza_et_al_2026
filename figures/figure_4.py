import itertools
import pickle

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.service.behavioral_processing import BehavioralProcessing
from rg_behavior_model.service.statistics_service import StatisticsService
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import ConfigurationExperiment
from rg_behavior_model.utils.constants import StimulusParameterLabel

# env
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_data = path_dir / "age_analysis"
path_save = Path(env['PATH_SAVE'])

# parameters plot
style = BehavioralModelStyle(plot_label_i=1)

plot_height = style.plot_height
plot_width = style.plot_width_small

xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

padding = style.padding
padding_small = style.padding_small

palette = style.palette["arlecchino"]
number_bins_hist = 15  # number of bins used in histogram plots

# define age-specific datasets
models_in_age_list = [
    {"label_show": "5dpf",
     "path": f"{path_data}/5_dpf",
     "path_fish": f"{path_data}/5_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/5_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "6dpf",
     "path": f"{path_data}/6_dpf",
     "path_fish": f"{path_data}/6_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/6_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "7dpf",
     "path": f"{path_data}/7_dpf",
     "path_fish": f"{path_data}/7_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/7_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "8dpf",
     "path": f"{path_data}/8_dpf",
     "path_fish": f"{path_data}/8_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/8_dpf/data_synthetic_fish_all.hdf5"},
    {"label_show": "9dpf",
     "path": f"{path_data}/9_dpf",
     "path_fish": f"{path_data}/9_dpf/data_fish_all.hdf5",
     "path_simulation": f"{path_data}/9_dpf/data_synthetic_fish_all.hdf5"},
]

# fetch data
for m in models_in_age_list:
    try:
        m["df_data"] = pd.read_hdf(m["path_fish"])
        m["df_simulation"] = pd.read_hdf(m["path_simulation"])
        m["df_simulation"].reset_index(inplace=True)
    except (KeyError, NotImplementedError):
        print(f"No data for group {m['label_show']}")
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

# show plots
show_loss_reduction = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True
show_distribution_parameters = True

# Make a standard figure
fig = Figure()

if show_loss_reduction:
    ypos = ypos - padding_small
    for i_age, models_in_age in enumerate(models_in_age_list):
        loss_list = []
        loss_start_list = []
        loss_end_list = []
        for path_error in Path(models_in_age["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_start_list.append(df_error["score"][0])

            loss_end = df_error["score"][len(df_error)-1]
            loss_end_list.append(loss_end)
            loss_list.append(loss_end)

        plot_title = f"{models_in_age['label_show']}\n" fr"n={len(loss_list)}"
        plot_loss = fig.create_plot(plot_label=style.get_plot_label() if i_age == 0 else None,
                                    plot_title=plot_title,
                                    xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                    ymin=0, ymax=20, yticks=[0, 10, 20] if i_age == 0 else None,
                                    yl="Loss" if i_age == 0 else None,
                                    xl="Iteration", xmin=0.5, xmax=2.5, xticks=[1, 2],
                                    xticklabels=["0", "1500"])
        xpos += plot_width + padding_small
        ypos = ypos

        for i_loss in range(len(loss_list)):
            loss_start = loss_start_list[i_loss]
            loss_end = loss_end_list[i_loss]
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

    xpos = xpos_start
    ypos = ypos - padding - plot_height

if show_psychometric_curve:
    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[StimulusParameterLabel.COHERENCE].isin(ConfigurationExperiment.coherence_list)]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, correct_bout_list_data, std_correct_bout_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[StimulusParameterLabel.COHERENCE].isin(ConfigurationExperiment.coherence_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, correct_bout_list_sim, std_correct_bout_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(df_simulation_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        plot_0 = fig.create_plot(plot_label=style.get_plot_label() if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                                 xticks=None,
                                 yl="Percentage\ncorrect swims (%)" if i_m == 0 else None, ymin=45, ymax=100, yticks=[50, 100] if i_m == 0 else None, hlines=[50])

        # draw
        plot_0.draw_line(x=parameter_list_data, y=correct_bout_list_data*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data)*100,  # / np.sqrt(number_individuals),
                         lc="k", lw=1)
        plot_0.draw_line(x=parameter_list_sim, y=correct_bout_list_sim*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim)*100,  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2))

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_small
        xpos = xpos + pad + plot_width

    xpos = xpos_start
    ypos = ypos - padding - plot_height

if show_coherence_vs_interbout_interval:
    for i_m, m in enumerate(models_in_age_list):
        df_data = m["df_data"]
        df_data_filtered = df_data.query(query_time)
        df_data_filtered = df_data_filtered[df_data_filtered[StimulusParameterLabel.COHERENCE].isin(ConfigurationExperiment.coherence_list)]

        original_len = len(df_data_filtered)
        df_data_filtered = df_data_filtered[df_data_filtered[ConfigurationExperiment.CorrectBoutColumn] != -1]
        filtered_len = len(df_data_filtered)
        print(f"EXCLUDE STRAIGHT BOUT |at {m['label_show']} {(original_len - filtered_len)/original_len*100:.03f}% dropped")

        ibi_quantiles = np.quantile(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn], [0.05, 0.95])
        df_data_filtered = df_data_filtered[np.logical_and(df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] > ibi_quantiles[0],
                                                           df_data_filtered[ConfigurationExperiment.ResponseTimeColumn] < ibi_quantiles[1])]
        try:
            df_data_filtered["experiment_ID"] = df_data_filtered["fish_ID"]
            df_data_filtered.drop("fish_ID", inplace=True)
        except KeyError:
            pass
        parameter_list_data, interbout_interval_list_data, std_interbout_interval_list_data = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_data_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE, column_name=ConfigurationExperiment.ResponseTimeColumn)
        try:
            number_individuals = len(df_data_filtered.index.unique("experiment_ID"))
        except KeyError:
            number_individuals = len(df_data_filtered["experiment_ID"].unique())

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[StimulusParameterLabel.COHERENCE].isin(ConfigurationExperiment.coherence_list)]

        number_models = len(df_simulation_filtered["fish_ID"].unique())

        parameter_list_sim, interbout_interval_list_sim, std_interbout_interval_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
                df_simulation_filtered, analysed_parameter=StimulusParameterLabel.COHERENCE, column_name=ConfigurationExperiment.ResponseTimeColumn)
        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # define plot
        plot_0 = fig.create_plot(plot_label=style.get_plot_label() if i_m == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width,
                                 errorbar_area=True,
                                 xl=ConfigurationExperiment.coherence_label,
                                 xmin=min(ConfigurationExperiment.coherence_list),
                                 xmax=max(ConfigurationExperiment.coherence_list),
                                 xticks=[int(p) for p in ConfigurationExperiment.coherence_list],
                                 yl="Interbout\ninterval (s)" if i_m == 0 else None,
                                 ymin=0, ymax=3, yticks=[0, 1.50, 3] if i_m == 0 else None)

        # draw
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),  # / np.sqrt(number_individuals),
                         lc="k", lw=1, label=f"data" if i_m == 0 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2), label=f"simulation" if i_m == 0 else None)

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_small
        xpos = xpos + pad + plot_width

    xpos = xpos_start
    ypos = ypos - padding - plot_height

if show_distribution_parameters:
    # configuration plot
    plot_height_here = style.plot_height_small
    padding_here = style.padding_in_plot_small

    from_best_model = True

    number_resampling_bootstrapping = StatisticsService.number_resampling_bootstrapping
    threshold_p_value_significant = StatisticsService.threshold_p_value_significant

    # initialization data structures to store results
    distribution_trajectory_dict = {p["label"]: np.zeros((number_bins_hist, len(models_in_age_list))) for p in ConfigurationDDM.parameter_list}
    raw_data_dict_per_fish = {p["label"]: {i_age: {} for i_age in range(len(models_in_age_list))} for p in ConfigurationDDM.parameter_list}
    raw_data_dict = {p["label"]: {i_age: [] for i_age in range(len(models_in_age_list))} for p in ConfigurationDDM.parameter_list}
    median_groups = {p["label"]: np.zeros(len(models_in_age_list)) for p in ConfigurationDDM.parameter_list}
    quantiles_groups = {p["label"]: np.zeros((len(models_in_age_list), 3)) for p in ConfigurationDDM.parameter_list}
    # compute
    for i_age, models_in_age in enumerate(models_in_age_list):
        model_dict = {}
        path_dir = Path(models_in_age["path"])
        for model_filepath in path_dir.glob('model_*_fit.hdf5'):
            model_filename = str(model_filepath.name)
            model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

        fish_list = np.arange(len(model_dict.keys()))

        model_parameter_median_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_dict = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
        model_parameter_median_dict["score"] = {}
        model_parameter_dict["score"] = {}
        model_parameter_median_array = np.zeros((len(ConfigurationDDM.parameter_list)+1, len(fish_list)))
        for i_model, id_model in enumerate(model_dict.keys()):
            df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])

            id_fish = i_model
            if from_best_model:
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit_list = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]

            model_parameter_median_dict["score"][id_fish] = np.median(df_model_fit_list["score"])
            model_parameter_dict["score"][id_fish] = np.array(df_model_fit_list["score"])
            model_parameter_median_array[0, i_model] = np.median(df_model_fit_list["score"])

            for i_p, p in enumerate(ConfigurationDDM.parameter_list):
                p_median = np.median(df_model_fit_list[p["label"]])
                model_parameter_median_dict[p["label"]][id_fish] = p_median  # (p_median - p["min"]) / (p["max"] - p["min"])
                model_parameter_dict[p["label"]][id_fish] = np.array(df_model_fit_list[p["label"]])  # (np.array(df_model_fit_list[p["label"]]) - p["min"]) / (p["max"] - p["min"])
                model_parameter_median_array[i_p+1, i_model] = p_median

                if id_model not in raw_data_dict_per_fish[p["label"]][i_age].keys():
                    raw_data_dict_per_fish[p["label"]][i_age][id_model] = [p_median]
                else:
                    raw_data_dict_per_fish[p["label"]][i_age][id_model].append(p_median)

                raw_data_dict[p["label"]][i_age].append(p_median)

        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            median_groups[p["label"]][i_age] = np.median(raw_data_dict[p["label"]][i_age])

            model_parameter_sampling_list = StatisticsService.sample_random(
                array=model_parameter_median_array[i_p + 1, :], sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)

            median_list = np.array([np.median(m) for m in model_parameter_sampling_list])
            quantiles_groups[p["label"]][i_age, :] = np.quantile(median_list, [0.05, 0.5, 0.95])
            print(f"group {models_in_age['label_show']} | {p['label_show']}: {quantiles_groups[p['label']][i_age, 1]:.05f} [{quantiles_groups[p['label']][i_age, 0]:.05f}, {quantiles_groups[p['label']][i_age, 2]:.05f}]")

        hist_model_parameter_median_dict = {}
        bin_model_parameter_median_dict = {}
        hist_model_parameter_median_dict[ConfigurationDDM.score_config["label"]], bin_model_parameter_median_dict[ConfigurationDDM.score_config["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[0, :], center_bin=True,  hist_range=[ConfigurationDDM.score_config["min"], ConfigurationDDM.score_config["max"]],
                bins=number_bins_hist,
                density=True
            )
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[p["label"]] = StatisticsService.get_hist(
                model_parameter_median_array[i_p + 1, :], center_bin=True,  hist_range=[p["min"], p["max"]],
                bins=number_bins_hist,
                density=True
            )
            distribution_trajectory_dict[p["label"]][:, i_age] = hist_model_parameter_median_dict[p["label"]]

    # plot distribution parameters estimations in each age population
    for i_age in range(len(models_in_age_list)):
        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            plot_n = fig.create_plot(plot_label=style.get_plot_label() if i_p == 0 and i_age == 0 else None, xpos=xpos, ypos=ypos,
                                     plot_height=plot_height_here, plot_width=plot_width,
                                     yl="Percentage fish (%)" if i_p == 0 and i_age == 0 else None, ymin=0, ymax=50, yticks=[0, 50] if i_p == 0 and i_age == 0 else None,
                                     xmin=p["min"], xmax=p["max"],
                                     vlines=[p["mean"]] if p["mean"] == 0 else [])

            plot_n.draw_line(bin_model_parameter_median_dict[p["label"]], distribution_trajectory_dict[p["label"]][:, i_age] * 100,
                             line_dashes=models_in_age_list[i_age]["dashes"], lc=palette[i_p], alpha=models_in_age_list[i_age]["alpha"], label=models_in_age_list[i_age]["label_show"] if i_p == len(ConfigurationDDM.parameter_list)-1 else None)

            xpos = xpos + padding_small + plot_width

        xpos = xpos_start
        ypos = ypos - plot_height_here - padding_here

    xpos = xpos_start
    ypos = ypos - (padding-padding_here)

    # plot highlighting populations with significantly different parameter estimations
    plot_list = []
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        plot_n = fig.create_plot(plot_label=style.get_plot_label() if i_p==0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height, plot_width=plot_width, errorbar_area=False,
                                 ymin=0, ymax=len(models_in_age_list) + 1,
                                 yticks=np.arange(len(models_in_age_list), 0, -1) if i_p == 0 else None,
                                 yticklabels=[g["label_show"] for g in models_in_age_list] if i_p == 0 else None,
                                 xl=p['label_show'].capitalize(), xmin=p["min"], xmax=p["max"],
                                 xticks=[p["min"], p["mean"], p["max"]], vlines=[p["mean"]] if p["mean"] == 0 else [])
        xpos = xpos + padding_small + plot_width
        for i_group, models_in_group in enumerate(models_in_age_list):
            plot_n.draw_line(quantiles_groups[p["label"]][i_group, :],
                             np.ones(quantiles_groups[p["label"]].shape[1]) * (len(models_in_age_list) - i_group),
                             lc=palette[i_p])
        plot_list.append(plot_n)

    pairs_to_compare_list = list(itertools.combinations(range(len(models_in_age_list)), 2))
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        parameter_range = p["max"] - p["min"]
        x_sig = p["max"] - parameter_range / 2 + parameter_range / 10
        is_sig = False
        print(f"{p['label_show']}")
        for pairs_to_compare in pairs_to_compare_list:
            median_delta = np.abs(median_groups[p["label"]][pairs_to_compare[0]] - median_groups[p["label"]][pairs_to_compare[1]])
            combined_array = np.concatenate((np.array(raw_data_dict[p["label"]][pairs_to_compare[0]]),
                                             np.array(raw_data_dict[p["label"]][pairs_to_compare[1]])))
            model_parameter_sampling_control_0 = StatisticsService.sample_random(
                array=combined_array, sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)
            model_parameter_sampling_control_1 = StatisticsService.sample_random(
                array=combined_array, sample_number=number_resampling_bootstrapping,
                sample_percentage_size=1, with_replacement=True, add_noise=None)

            median_control_0 = np.array([np.median(m) for m in model_parameter_sampling_control_0])
            median_control_1 = np.array([np.median(m) for m in model_parameter_sampling_control_1])
            median_control_delta = np.abs(median_control_0 - median_control_1)

            p_value = np.mean(median_control_delta >= median_delta)
            print(f"{models_in_age_list[pairs_to_compare[0]]['label_show']} vs {models_in_age_list[pairs_to_compare[1]]['label_show']} | p = {p_value}")

            plot_n = plot_list[i_p]
            if p_value < threshold_p_value_significant:
                is_sig = True
                x_sig_array = np.ones(2) * x_sig
                y_sig_array = np.array(
                    (len(models_in_age_list) - pairs_to_compare[0], len(models_in_age_list) - pairs_to_compare[1]))
                plot_n.draw_line(x_sig_array, y_sig_array, lc="k")

                x_sig += parameter_range / 10

        if is_sig:
            plot_n.draw_text(x=x_sig + parameter_range / 6, y=np.ceil(len(models_in_age_list) / 2), text="*",
                             textlabel_rotation=-90)
    xpos = xpos_start
    ypos = ypos - (plot_height + padding)

fig.save(path_save / f"figure_4.pdf", open_file=False, tight=style.page_tight)


