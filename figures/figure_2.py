import pathlib

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.utils.figure_helper import Figure
from rg_behavior_model.figures.style import BehavioralModelStyle
from rg_behavior_model.model.core.params import Parameter, ParameterList
from rg_behavior_model.model.ddm import DDMstable
from rg_behavior_model.service.behavioral_processing import BehavioralProcessing
from rg_behavior_model.service.statistics_service import StatisticsService
from rg_behavior_model.utils.configuration_ddm import ConfigurationDDM
from rg_behavior_model.utils.configuration_experiment import StimulusParameterLabel, ConfigurationExperiment
from rg_behavior_model.utils.constants import ResponseTimeColumn

# env
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])
path_save = Path(env['PATH_SAVE'])
path_data = path_dir / 'base_dataset'

# configuration plot
style = BehavioralModelStyle(plot_label_i=1)

xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start

plot_height = style.plot_height
plot_height_small = plot_height / 2.5

plot_width = style.plot_width
plot_width_small = style.plot_width_small

padding = style.padding
padding_plot = style.padding_in_plot
padding_short = style.padding / 2
padding_vertical = plot_height_small

lw = style.linewidth
number_bins_hist = 15

# show plots
show_trajectory_decision_variable = False
show_loss_reduction = False
show_psychometric_curve = False
show_coherence_vs_interbout_interval = False
show_rt_distributions = False
show_individual_estimations = True
show_distribution_parameters = True

# configurations experiment
time_start_simulation = 0  # (s)
time_end_simulation = 30  # (s)
time_duration_simulation = time_end_simulation - time_start_simulation
analysed_parameter = StimulusParameterLabel.COHERENCE.value
query_time = f'start_time > {ConfigurationExperiment.time_start_stimulus} and end_time < {ConfigurationExperiment.time_end_stimulus}'

df_dict = {}
fish_to_include_list = ConfigurationExperiment.example_fish_list
config_list = [{"label": "data", "line_dashes": None, "alpha": 0.5, "color": None},
               {"label": "fit", "line_dashes": (2, 4), "alpha": 1, "color": "k"}]
for i_fish, fish in enumerate(fish_to_include_list):
    df_data = pd.read_hdf(path_data / f"data_fish_{fish}.hdf5")
    for path_fit in path_data.glob(f"data_synthetic_fish_{fish}_*.hdf5"):
        df_fit = pd.read_hdf(path_fit)
        break
    df_dict[fish] = {"fit": df_fit, "data": df_data, "color": style.palette["fish_code"][i_fish]}  # BehavioralProcessing.remove_fast_straight_bout(df, threshold_response_time=100)

# Make a standard figure
fig = Figure()

plot_height_row = plot_height_small * 2 + padding_vertical

if show_trajectory_decision_variable:
    x_show_start = 0
    x_show_end = 100
    plot_width_here = style.plot_width * 8
    plot_height_here = style.plot_height * 0.7
    index_parameter_to_simulate = [1]
    simulated_parameters = [ConfigurationExperiment.coherence_list[i] for i in index_parameter_to_simulate]

    # model parameters
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=ConfigurationDDM.dt))
    parameters.add_parameter("noise_sigma", Parameter(value=1))
    parameters.add_parameter("scaling_factor", Parameter(value=0.6))
    parameters.add_parameter("leak", Parameter(value=-1))
    parameters.add_parameter("residual_after_bout", Parameter(value=0.03))
    parameters.add_parameter("inactive_time", Parameter(value=0.1))
    parameters.add_parameter("threshold", Parameter(value=ConfigurationDDM.threshold))

    # instantiate model
    ddm_model = DDMstable(parameters, trials_per_simulation=200, time_experimental_trial=30, scaling_factor_input=1)
    ddm_model.define_stimulus(time_start_stimulus=ConfigurationExperiment.time_start_stimulus, time_end_stimulus=ConfigurationExperiment.time_end_stimulus)

    time_trial_list = np.arange(x_show_start, x_show_end + ConfigurationDDM.dt, ConfigurationDDM.dt)
    for index_parameter, parameter in enumerate(simulated_parameters):
        # plot frame
        plot_n = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos,
                                 ypos=ypos,
                                 plot_height=plot_height_here, plot_width=plot_width_here,
                                 xl="Simulation time (s)",
                                 xmin=x_show_start, xmax=x_show_end, xticks=None,
                                 ymin=-1 * ConfigurationDDM.threshold - 0.5, ymax=ConfigurationDDM.threshold + 0.5,
                                 yticks=[-ConfigurationDDM.threshold, ConfigurationDDM.threshold], yticklabels=["-B", "B"], hlines=[0])
        plot_n.draw_line(time_trial_list, np.zeros_like(time_trial_list) - ConfigurationDDM.threshold, lc="k")
        plot_n.draw_line(time_trial_list, np.zeros_like(time_trial_list) + ConfigurationDDM.threshold, lc="k")

        scalebar_time = 5
        scalebar = time_trial_list[-int(scalebar_time/ConfigurationDDM.dt):]
        plot_n.draw_line(scalebar, np.zeros_like(scalebar) - ConfigurationDDM.threshold - 0.3, lc="k")
        plot_n.draw_text(x_show_end - scalebar_time, -ConfigurationDDM.threshold - 0.4, f"{scalebar_time}s", textlabel_ha="left", textlabel_va="top")

        plot_n.draw_line(np.linspace(x_show_start, x_show_end, len(time_trial_list)), np.zeros_like(time_trial_list) + ConfigurationDDM.threshold + 0.3, lc=style.palette["stimulus"][-1-index_parameter_to_simulate[index_parameter]])
        plot_n.draw_text(x_show_end + 1, ConfigurationDDM.threshold + 0.2, f"Coh={parameter}%", textlabel_ha="left")

        # constant input
        input_signal = np.zeros(len(time_trial_list)) + parameter / 100

        # simulate
        response_time_list, bout_decision_list, decision_time_list, internal_state_trajectory = ddm_model.simulate_trial(
            input_signal=input_signal)

        # plot
        start_time_decision = 0
        for i_d, decision_time in enumerate(decision_time_list):
            if decision_time > x_show_end:
                break
            index_time_decision = np.argwhere(time_trial_list <= x_show_end).flatten()
            plot_n.draw_line(time_trial_list[index_time_decision],
                             internal_state_trajectory[index_time_decision],
                             lc=style.palette["neutral"][0], lw=0.1)
            if bout_decision_list[i_d] > 0:
                y = ConfigurationDDM.threshold
                color_dot = style.palette["correct_incorrect"][0]
            else:
                y = -ConfigurationDDM.threshold
                color_dot = style.palette["correct_incorrect"][1]
            plot_n.draw_scatter(decision_time, y, pc=color_dot, elw=0)

        ypos -= padding + plot_height_here

if show_loss_reduction:
    ypos = ypos - padding_short

    plot_loss = fig.create_plot(plot_label=style.get_plot_label(),
                                xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                ymin=0, ymax=20, yticks=[0, 10, 20],
                                yl="Loss",
                                xl="Iteration", xmin=0.5, xmax=2.5, xticks=[1, 2],
                                xticklabels=["0", "1500"])
    for i_fish_id, fish_id in enumerate(fish_to_include_list):
        loss_array = np.zeros(2)
        for path_error in path_data.glob(f"error_fish_{fish_id}_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_array[0] = df_error["score"][0]

            loss_array[1] = df_error["score"][len(df_error)-1]

            plot_loss.draw_line((1, 2), loss_array, lc=df_dict[fish_id]["color"])
            plot_loss.draw_scatter((1, 2), loss_array, ec=df_dict[fish_id]["color"], pc=df_dict[fish_id]["color"], label=f"fish {i_fish_id}")
    ypos = ypos - padding - plot_height
    xpos = xpos_start

if show_psychometric_curve:
    plot_height_here = plot_height_row
    plot_width_here = plot_width
    color_line = "gray"
    show_label = True

    plot_0 = fig.create_plot(plot_label=style.get_plot_label(), xpos=xpos, ypos=ypos,
                             plot_height=plot_height_here, plot_width=plot_width_here,
                             xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                             xticks=None,
                             yl="Percentage\ncorrect swims (%)", ymin=0, ymax=100,
                             yticks=[0, 50, 100], hlines=[0.5])

    i_fish = 1
    for k_fish, df_fish_dict in df_dict.items():
        for config in config_list:
            # filter
            df = df_fish_dict[config["label"]]
            df_filtered = df.query(query_time)
            df_filtered = df_filtered[df_filtered[analysed_parameter].isin(ConfigurationExperiment.coherence_list)]

            # compute
            p_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(df_filtered, analysed_parameter=analysed_parameter)
            correct_bout_list *= 100
            p_list = np.array([int(p) for p in p_list])

            # plot
            plot_0.draw_line(x=p_list, y=correct_bout_list, lc=df_fish_dict["color"], lw=lw, alpha=config["alpha"], line_dashes=config["line_dashes"])
            if show_label and config["label"] == "data":
                plot_0.draw_text(max(p_list) + 0.1, correct_bout_list[-1], f"fish {i_fish}",
                                 textlabel_rotation='horizontal', textlabel_ha='left', textcolor=df_fish_dict["color"])
                i_fish += 1

    ypos = ypos
    xpos = xpos + padding + plot_width

xpos = xpos_start
ypos = ypos - padding - plot_height

if show_coherence_vs_interbout_interval:
    plot_height_here = plot_height_row
    plot_width_here = plot_width
    color_line = "gray"
    show_label = True

    plot_0 = fig.create_plot(xpos=xpos, ypos=ypos,
                             plot_height=plot_height_here, plot_width=plot_width_here,
                             errorbar_area=True,
                             xl=ConfigurationExperiment.coherence_label, xmin=min(ConfigurationExperiment.coherence_list), xmax=max(ConfigurationExperiment.coherence_list),
                             xticks=[int(p) for p in ConfigurationExperiment.coherence_list],
                             yl="Interbout interval (s)", ymin=0, ymax=2, yticks=[0, 1, 2])

    i_fish = 1
    for k_fish, df_fish_dict in df_dict.items():
        for config in config_list:
            # filter
            df = df_fish_dict[config["label"]]
            df_filtered = df.query(query_time)
            df_filtered = df_filtered[df_filtered[analysed_parameter].isin(ConfigurationExperiment.coherence_list)]

            # computation
            p_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(df_filtered,
                                                                                                                      analysed_parameter=analysed_parameter,
                                                                                                                      column_name=ResponseTimeColumn)
            p_list = np.array([int(p) for p in p_list])

            # plot
            plot_0.draw_line(x=p_list, y=correct_bout_list, lc=df_fish_dict["color"], lw=lw, alpha=config["alpha"], line_dashes=config["line_dashes"])
            if show_label and config["label"] == "data":
                plot_0.draw_text(max(p_list) + 0.1, correct_bout_list[-1], f"fish {i_fish}",
                                 textlabel_rotation='horizontal', textlabel_ha='left', textcolor=df_fish_dict["color"])
                i_fish += 1

    ypos = ypos
    xpos = xpos + padding + plot_width

xpos = xpos_start
ypos = ypos - padding - plot_height * 2

if show_rt_distributions:
    # parameters
    padding_here = plot_height_small * 2
    palette = style.palette["stimulus"]
    x_limits = [0, 2]
    y_limits = [0, 0.2]

    plot_height_here = plot_height_small * 3
    plot_width_here = 0.7

    plot_section = {}

    df_dict.pop(ConfigurationExperiment.all_fish_label, None)  # get rid of the df with all data merged

    for i_k, k in enumerate(df_dict.keys()):
        for i_param, parameter in enumerate(ConfigurationExperiment.coherence_list):
            # plot
            line_dashes = None
            plot_title = None
            if i_k == 0:
                if i_param == 0:
                    plot_title = f"Coh={parameter}%"
                else:
                    plot_title = f"{parameter}%"
            plot_section[i_param] = fig.create_plot(plot_label=style.get_plot_label() if i_k == 0 and i_param == 0 else None,
                                                            plot_title=plot_title,
                                                            xpos=xpos + i_param * (plot_width_here + padding_plot),
                                                            ypos=ypos + plot_height_small + padding_vertical,
                                                            plot_height=plot_height_here,
                                                            plot_width=plot_width_here,
                                                            xmin=x_limits[0], xmax=x_limits[-1],
                                                            xticks=None, yticks=None,
                                                            ymin=-y_limits[-1], ymax=y_limits[-1],
                                                            hlines=[0])

            if i_param == len(ConfigurationExperiment.coherence_list) - 1 and i_k == len(df_dict)-1:
                y_location_scalebar = y_limits[-1] / 6
                x_location_scalebar = x_limits[-1] / 6
                plot_section[i_param].draw_line((1.7, 1.7), (y_location_scalebar, y_location_scalebar + 0.1), lc="k")
                plot_section[i_param].draw_text(2, y_location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left', textlabel_va="bottom")

                plot_section[i_param].draw_line((x_location_scalebar, x_location_scalebar + 0.5), (-y_location_scalebar, -y_location_scalebar), lc="k")
                plot_section[i_param].draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                        textlabel_rotation='horizontal', textlabel_ha='left', textlabel_va="bottom")

            for config in config_list:
                df = df_dict[k][config["label"]]
                plot_section_corr = plot_section[i_param]
                plot_section_err = plot_section[i_param]

                df_filtered = df[df[analysed_parameter] == parameter]
                df_filtered = df_filtered.query(query_time)

                duration = np.sum(
                    BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=ConfigurationExperiment.time_end_stimulus-ConfigurationExperiment.time_start_stimulus)
                )

                # plot distribution of data over coherence levels
                data_corr = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 1][ResponseTimeColumn]
                data_err = df_filtered[df_filtered[ConfigurationExperiment.CorrectBoutColumn] == 0][ResponseTimeColumn]

                data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                       bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                       duration=duration,
                                                                                       center_bin=True)
                index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
                data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
                data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

                data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                     bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                     duration=duration,
                                                                                     center_bin=True)
                index_in_limits = np.argwhere(
                        np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
                data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
                data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

                if config["color"] is None:
                    lc_correct = style.palette["correct_incorrect"][0]
                    lc_incorrect = style.palette["correct_incorrect"][1]
                    alpha_correct = 1
                    alpha_incorrect = 1
                else:
                    lc_correct = config["color"]
                    lc_incorrect = config["color"]
                    alpha_correct = 0.7
                    alpha_incorrect = 0.3

                plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=lc_correct,
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_correct)
                plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=lc_incorrect,
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_incorrect)

        ypos = ypos - (padding_here + plot_height_small)

    ypos = ypos
    xpos = xpos + i_param * (plot_width_here + padding_plot) + plot_width_here + padding
xpos = xpos_start

if show_distribution_parameters:
    from_best_model = True
    plot_height_here = style.plot_height
    plot_width_here = plot_width_small
    padding_here = style.padding
    palette = style.palette["default"]

    distribution_trajectory_dict = {p["label"]: np.zeros(number_bins_hist) for p in ConfigurationDDM.parameter_list}
    raw_data_dict_per_fish = {p["label"]: {} for p in ConfigurationDDM.parameter_list}
    raw_data_dict = {p["label"]: [] for p in ConfigurationDDM.parameter_list}
    model_dict = {}
    n_models = 0
    for model_filepath in path_data.glob('model_*_fit.hdf5'):
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
        # id_fish = id_model[:2]

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

            if id_model not in raw_data_dict_per_fish[p["label"]].keys():
                raw_data_dict_per_fish[p["label"]][id_model] = [p_median]
            else:
                raw_data_dict_per_fish[p["label"]][id_model].append(p_median)

            raw_data_dict[p["label"]].append(p_median)

    if show_individual_estimations:
        plot_height_here = plot_height * 3
        number_fish = model_parameter_median_array.shape[1]
        fish_id_array = np.flip(np.arange(number_fish))

        for i_p, p in enumerate(ConfigurationDDM.parameter_list):
            vlines = [p["mean"]] if p["mean"] == 0 else []
            if p["relevant_values"] is not None:
                vlines.extend(p["relevant_values"])
            plot_individual = fig.create_plot(plot_label=style.get_plot_label() if i_p == 0 else None, xpos=xpos, ypos=ypos,
                                             plot_height=plot_height_here, plot_width=plot_width_here,
                                             yl="Fish ID" if i_p == 0 else None, ymin=0-0.5, ymax=number_fish-0.5, yticks=None,
                                             xmin=p["min"], xmax=p["max"], xticks=[p["min"], p["mean"], p["max"]],
                                             xl=p['label_show'].capitalize(),
                                             vlines=vlines)
            xpos += plot_width_here + padding_short

            plot_individual.draw_scatter(model_parameter_median_array[i_p + 1, :], fish_id_array, pc=palette[i_p], elw=0)
        xpos = xpos_start
        ypos -= plot_height_here + padding


    plot_height_here = style.plot_height / 4

    hist_model_parameter_median_dict = {}
    bin_model_parameter_median_dict = {}
    hist_model_parameter_median_dict[ConfigurationDDM.score_config["label"]], bin_model_parameter_median_dict[ConfigurationDDM.score_config["label"]] = StatisticsService.get_hist(
            model_parameter_median_array[0, :], center_bin=True,  hist_range=[ConfigurationDDM.score_config["min"], ConfigurationDDM.score_config["max"]],
            bins=number_bins_hist,  # int((score_config["max"] - score_config["min"])/0.1),
            density=True
        )
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        hist_model_parameter_median_dict[p["label"]], bin_model_parameter_median_dict[p["label"]] = StatisticsService.get_hist(
            model_parameter_median_array[i_p + 1, :], center_bin=True,  hist_range=[p["min"], p["max"]],
            bins=number_bins_hist,
            density=True
        )
        distribution_trajectory_dict[p["label"]] = hist_model_parameter_median_dict[p["label"]]
        print(f"{p['label_show']} spans {(np.max(model_parameter_median_array[i_p + 1, :])-np.min(model_parameter_median_array[i_p + 1, :]))/(p['max']-p['min'])*100}% of the parameter space")
        if p["label"] == "leak":
            around_optimal_integration = hist_model_parameter_median_dict[p["label"]][9:18]
            print(f"LEAK | {np.sum(around_optimal_integration)*100}% is around optimal")

            optimal_integration = hist_model_parameter_median_dict[p["label"]][11:16]
            print(f"LEAK | {np.sum(optimal_integration)*100}% is optimal")

    # #####
    for i_p, p in enumerate(ConfigurationDDM.parameter_list):
        plot_n = fig.create_plot(plot_label=style.get_plo if i_p == 0 else None, xpos=xpos, ypos=ypos,
                                 plot_height=plot_height_here, plot_width=plot_width_here,
                                 yl="Percentage fish (%)" if i_p == 0 else None, ymin=0, ymax=50, yticks=[0, 50] if i_p == 0 else None,
                                 xmin=p["min"], xmax=p["max"], xticks=[p["min"], p["mean"], p["max"]], xl=p['label_show'].capitalize(),
                                 vlines=[p["mean"]] if p["mean"] == 0 else [])

        plot_n.draw_line(bin_model_parameter_median_dict[p["label"]], distribution_trajectory_dict[p["label"]] * 100,
                         lc=palette[i_p])

        xpos = xpos + padding_short + plot_width_here

    xpos = xpos_start
    ypos = ypos - plot_height_here - padding_here

fig.save(path_save / "figure_2.pdf", open_file=False, tight=style.page_tight)
