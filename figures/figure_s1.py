import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, \
    ResponseTimeColumn, alphabet, Keyword
from analysis_helpers.analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis_helpers.analysis.utils.figure_helper import Figure

# env
path_dir_example = Path(r"C:\Users\Roberto\Academics\data\benchmarking\example_datasets")
path_dir_100 = Path(r"C:\Users\Roberto\Academics\data\benchmarking\weight_nosmooth\_30")

# parameters plot
plot_height = 1
plot_height_small = plot_height / 2.5
padding_plot = 0.5
padding_vertical = plot_height_small
i_plot_label = 0  # 6  # 0  #
plot_label_list = alphabet
color_neutral = "#bfbfbfff"

style = BehavioralModelStyle()
xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start
padding = style.padding
plot_height = style.plot_height
plot_width = style.plot_width * 3/2
style.add_palette("green", Palette.green_short)
style.add_palette("neutral", [Palette.color_neutral])
palette = style.palette["default"]

# single-fish analysis is necessary
show_rt_distributions = True
show_psychometric_curve = True
show_coherence_vs_interbout_interval = True


# parameters_experiment
analysed_parameter_list = [0, 25, 50, 100]
time_start_stimulus = 10  # 10  # seconds
time_end_stimulus = 40  # seconds
time_experimental_trial = 50  # seconds
number_individuals = 16
analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
analysed_parameter_label = "coherence [%]"  # "period [s]"  #
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

df_dict = {}
# all_label = "all"
fish_0_label = "fish_200"
fish_1_label = "synthetic_test_ok"
fish_2_label = "synthetic_test_low"
fish_3_label = "synthetic_test_high"
# color_list = Palette.cool
fish_to_include_list = [fish_0_label, fish_1_label, fish_2_label, fish_3_label]
label_list = ["real", "accepted", "not accepted low", "not accepted high"]
# color_list = ["#B37DE4", "#FF03C8", "#66F7FF", "#85FF0A"]
color_list = ["#000000", "#000000", "#000000", "#000000"]
dash_list = [None, None, None, None]
# dash_list = [(1, 2), (2, 2), (1, 1, 2, 1), None]
for fish in fish_to_include_list:
    df = pd.read_hdf(path_dir_example / f"data_{fish}.hdf5")
    df_dict[fish] = df  # BehavioralProcessing.remove_fast_straight_bout(df, threshold_response_time=100)


# Make a standard figure
fig = Figure()

plot_height_row = plot_height_small * 2 + padding_vertical

for i_fish, id_fish in enumerate(fish_to_include_list):
    if show_psychometric_curve:
        plot_height = plot_height_row
        plot_width = 1

        # plot
        line_dashes = None
        plot_0 = fig.create_plot(plot_label=plot_label_list[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list), xticks=[int(p) for p in analysed_parameter_list], yl="accuracy [-]",
                                 ymin=0, ymax=1,
                                 yticks=[0, 0.5, 1], hlines=[0.5])


        # for i_id, id in enumerate(fish_to_include_list):
        df_fish = df_dict[id_fish]
        df_fish_filtered = df_fish[df_fish[analysed_parameter].isin(analysed_parameter_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=analysed_parameter)

        # plot
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_list[i_fish], line_dashes=dash_list[i_fish], alpha=0.8)

        # ypos = ypos - padding - plot_height
        # xpos = xpos_start
        ypos = ypos
        xpos = xpos + padding + plot_width

    if show_coherence_vs_interbout_interval:
        plot_height = plot_height_row
        plot_width = 1

        # plot
        line_dashes = None
        ymax = 30 if "low" in id_fish else 2
        plot_0 = fig.create_plot(xpos=xpos, ypos=ypos,
                                 plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xl=analysed_parameter_label, xmin=min(parameter_list), xmax=max(parameter_list),
                                 xticks=[int(p) for p in parameter_list], yl="interbout interval [s]",
                                 ymin=0, ymax=ymax,
                                 yticks=[0, int(ymax/2), ymax])
        # i_plot_label += 1

        # for i_id, id in enumerate(fish_to_include_list):
        df_fish = df_dict[id_fish]
        df_fish_filtered = df_fish[df_fish[analysed_parameter].isin(analysed_parameter_list)]
        # computation
        parameter_list, correct_bout_list, std_correct_bout_list = BehavioralProcessing.compute_quantities_per_parameters(
            df_fish_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)

        # plot
        plot_0.draw_line(x=parameter_list, y=correct_bout_list, lc=color_list[i_fish], line_dashes=dash_list[i_fish], alpha=0.8)

        ypos = ypos
        xpos = xpos + padding + plot_width

    if show_rt_distributions:
        # parameters
        palette = style.palette["stimulus"]
        x_limits = [0, 2]  # None  #
        if "high" in id_fish:
            y_limits = [0, 0.5]
        else:
            y_limits = [0, 0.15]

        plot_width = 0.7

        plot_section = {"corr": {}, "err": {}}

        # for i_fish, id_fish in enumerate(df_dict.keys()):
        number_individuals = 1

        df = df_dict[id_fish]
        # id_list = np.sort(np.array(df.index.unique(level='experiment_ID')))
        # number_individuals = len(id_list)
        for i_param, parameter in enumerate(analysed_parameter_list):
            # plot
            # if i_fish == 0:
            if i_param == 0:
                plot_section["corr"][i_param] = fig.create_plot(xpos=xpos, ypos=ypos + plot_height_small + padding_vertical,
                                                    plot_height=plot_height_small,
                                                    plot_width=plot_width,
                                                    xmin=x_limits[0], xmax=x_limits[-1], xticks=[],
                                                    yl="correct",
                                                    ymin=y_limits[0], ymax=y_limits[-1], yticks=y_limits)
                                 # ,legend_xpos=xpos_start-1.5 * plot_width, legend_ypos=ypos_start)
                plot_section["err"][i_param] = fig.create_plot(xpos=xpos, ypos=ypos,
                                                   plot_height=plot_height_small,
                                                   plot_width=plot_width,
                                                   xl="interbout interval [s]",
                                                   xmin=x_limits[0], xmax=x_limits[-1], xticks=[0, 1, 2],
                                                   yl="incorrect",
                                                   ymin=y_limits[0], ymax=y_limits[-1], yticks=y_limits)
                plot_section["err"][i_param].draw_text(-3, 0.2, "activity [events/s]", textlabel_rotation='vertical', textlabel_ha='center')
            else:
                plot_section["corr"][i_param] = fig.create_plot(xpos=xpos + i_param * (plot_width + padding_plot), ypos=ypos + plot_height_small + padding_vertical,
                                                    plot_height=plot_height_small,
                                                    plot_width=plot_width,
                                                    xmin=x_limits[0], xmax=x_limits[-1], xticks=[],
                                                    ymin=y_limits[0], ymax=y_limits[-1])
                plot_section["err"][i_param] = fig.create_plot(xpos=xpos + i_param * (plot_width + padding_plot), ypos=ypos,
                                                   plot_height=plot_height_small,
                                                   plot_width=plot_width,
                                                   xmin=x_limits[0], xmax=x_limits[-1], xticks=[0, 1, 2],
                                                   ymin=y_limits[0], ymax=y_limits[-1])
            plot_section_corr = plot_section["corr"][i_param]
            plot_section_err = plot_section["err"][i_param]

            df_filtered = df[df[analysed_parameter] == parameter]
            df_filtered = df_filtered.query(query_time)

            duration = np.sum(
                BehavioralProcessing.get_duration_trials_in_df(df_filtered, fixed_time_trial=time_end_stimulus-time_start_stimulus)
            ) * number_individuals

            # plot distribution of data over coherence levels
            data_corr = df_filtered[df_filtered[CorrectBoutColumn] == 1][ResponseTimeColumn]
            data_err = df_filtered[df_filtered[CorrectBoutColumn] == 0][ResponseTimeColumn]

            data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                                   # bins=100,
                                                                                   bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                   duration=duration,
                                                                                   center_bin=True)
            index_in_limits = np.argwhere(np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
            data_hist_time_corr = data_hist_time_corr[index_in_limits].flatten()
            data_hist_value_corr = data_hist_value_corr[index_in_limits].flatten()

            data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                                 # bins=100,
                                                                                 bins=np.arange(x_limits[0], x_limits[-1], (x_limits[-1]-x_limits[0])/50),
                                                                                 duration=duration,
                                                                                 center_bin=True)
            index_in_limits = np.argwhere(
                    np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
            data_hist_time_err = data_hist_time_err[index_in_limits].flatten()
            data_hist_value_err = data_hist_value_err[index_in_limits].flatten()

            plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=color_list[i_fish], line_dashes=dash_list[i_fish], alpha=0.8, label=label_list[i_fish] if i_param == len(analysed_parameter_list)-1 else None)
            plot_section_err.draw_line(data_hist_time_err, data_hist_value_err, lc=color_list[i_fish], line_dashes=dash_list[i_fish], alpha=0.8)

        # i_plot_label += 1
        plot_height = 1
        ypos = ypos
        xpos = xpos + i_param * (plot_width + padding_plot) + plot_width + padding

    xpos = xpos_start
    ypos -= padding + plot_height
    i_plot_label += 1

parameter_list = [
    {"param": "noise_sigma",
     "label": "diffusion",
     "min": 0.0,
     "mean": 1.5,
     "max": 3.0},
    {"param": "scaling_factor",
     "label": "drift",
     "min": -3,
     "mean": 0,
     "max": 3},
    {"param": "leak",
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
parameter_dict = {p["param"]: [] for p in parameter_list}
for path_model in path_dir_100.glob("model_test_*.hdf5"):
    if "_fit" in path_model.name:
        continue
    df_model = pd.read_hdf(path_model)
    for p in parameter_list:
        parameter_dict[p["param"]].append(df_model[p["param"]])

parameter_array = np.squeeze(np.stack([(np.array(parameter_dict[p["param"]]) - p["min"]) / (p["max"] - p["min"]) for p in parameter_list]))

palette = style.palette["default"]
plot_height_here = plot_height * 2
plot_width_here = plot_width * 2
padding_here = padding
ypos -= 1
plot_height_here_ = plot_height_here
plot_width_here_ = plot_width_here * len(parameter_list) + padding_here * (len(parameter_list) - 1)

xticks = np.arange(0, 15, 3) + 1
plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos,
                             plot_height=plot_height_here_,
                             plot_width=plot_width_here_,
                             errorbar_area=False,
                             xmin=0, xmax=15, # xticks=xticks, xticklabels=[p["label"] for p in parameter_list], xticklabels_rotation=45,
                             ymin=0, ymax=1)
xpos += xpos_start
ypos -= plot_height_here_ + padding

for i_link in range(parameter_array.shape[1]):
    plot_0.draw_line(xticks, parameter_array[:, i_link], pc=Palette.color_neutral, alpha=0.1, lw=0.1)

for i_p, p in enumerate(parameter_list):
    x = np.ones(len(parameter_dict[p["param"]])) + i_p * 3
    y = (np.array(parameter_dict[p["param"]]) - p["min"]) / (p["max"] - p["min"])
    plot_0.draw_scatter(x, y, pc=palette[i_p], ec=palette[i_p])

    plot_0.draw_line(x[:2]-0.5, [0, 1], lc="k")
    plot_0.draw_text(x[0]-1, 0, int(p["min"]))
    plot_0.draw_text(x[0]-1, 1, int(p["max"]))
    plot_0.draw_text(x[0]-1.5, 0.5, p["label"], textlabel_rotation="vertical")

# fig.save(Path.home() / 'Academics' / 'graphics' / 'pictures' / 'figures_for_papers' / 'behavior_model' / "figure_1_experiment_coh_temp.pdf", open_file=True, tight=True)
fig.save(Path.home() / 'Desktop' / "figure_s1_acceptability_criterion.pdf", open_file=True, tight=style.page_tight)
