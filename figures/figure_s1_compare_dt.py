import pathlib

import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import dotenv_values

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, \
    ResponseTimeColumn, alphabet, Keyword
from analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis.utils.figure_helper import Figure

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
padding = 1.5  # style.padding
padding_short = 0.75  # style.padding
plot_height = style.plot_height
plot_width = style.plot_width
plot_width_short = style.plot_width * 0.9
letter_counter = 1
style.add_palette("neutral", [Palette.color_neutral])
palette = style.palette["default"]
style.add_palette("fish_code", ["#73489C", "#753B51", "#103882", "#7F0C0C"])

# single-fish analysis is necessary
show_distribution_error_across_repeats = True
show_distribution_parameters = True

# configurations experiment
analysed_parameter_list = [0, 25, 50, 100]
time_start_stimulus = 10  # 10  # seconds
time_end_stimulus = 40  # seconds
time_experimental_trial = 30  # seconds
analysed_parameter = StimulusParameterLabel.COHERENCE.value  # StimulusParameterLabel.PERIOD.value  #
analysed_parameter_label = "Coh (%)"  # "period [s]"  #
query_time = f'start_time > {time_start_stimulus} and end_time < {time_end_stimulus}'

# configuraitons structure
models_in_group_list = [
    {"label_fish_group": "",
     "label_show": "dt=0.0001",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_0_0001_sim",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
    {"label_fish_group": "",
     "label_show": "dt=0.001",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_0_001_sim",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
    {"label_fish_group": "",
     "label_show": "dt=0.01",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_0_01_fit",
     "dashes": None,
     "color": "k",
     "alpha": 1},
    {"label_fish_group": "",
     "label_show": "dt=0.1",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_0_1_sim",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
    {"label_fish_group": "",
     "label_show": "dt=0.5",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_0_5_sim",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
    {"label_fish_group": "",
     "label_show": "dt=1",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\dt_analysis\5_dpf_1_sim",
     "dashes": None,
     "color": "k",  # "#004AAA",
     "alpha": 1,  # 0.25
     },
]
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

# configurations plots
number_bins_hist = 15

# Make a standard figure
fig = Figure()

plot_height_row = plot_height_small * 2 + padding_vertical

df_dict = {}
all_label = "all"
fish_0_label = "205"
fish_1_label = "506"
fish_2_label = "201"
# fish_3_label = "504"
fish_to_include_list = [fish_1_label]
config_list = [{"label": "data", "line_dashes": None, "alpha": 0.5, "color": None, "time_start_stimulus": 10, "time_end_stimulus": 40},
               {"label": "fit", "line_dashes": (2, 4), "alpha": 1, "color": "k", "time_start_stimulus":10, "time_end_stimulus": 40}]

# parameters
padding_here = plot_height_small * 2
palette = style.palette["stimulus"]
x_limits = [0, 2]  # None  #
y_limits = [0, 0.2]

plot_height_here = plot_height_small * 3
plot_width_here = 0.7

def compute_distributions(df_filtered, duration):
    # plot distribution of data over coherence levels
    data_corr = df_filtered[df_filtered[CorrectBoutColumn] == 1][ResponseTimeColumn]
    data_err = df_filtered[df_filtered[CorrectBoutColumn] == 0][ResponseTimeColumn]

    data_hist_value_corr, data_hist_time_corr = StatisticsService.get_hist(data_corr,
                                                                           bins=int((x_limits[1] - x_limits[0]) / (0.05)),
                                                                           hist_range=x_limits,
                                                                           duration=duration,
                                                                           center_bin=True)
    # index_in_limits = np.argwhere(
    #     np.logical_and(data_hist_time_corr > x_limits[0], data_hist_time_corr < x_limits[1]))
    data_hist_time_corr = data_hist_time_corr.flatten()
    data_hist_value_corr = data_hist_value_corr.flatten()

    data_hist_value_err, data_hist_time_err = StatisticsService.get_hist(data_err,
                                                                         bins=int((x_limits[1] - x_limits[0]) / (0.05)),
                                                                         hist_range=x_limits,
                                                                         duration=duration,
                                                                         center_bin=True)
    # index_in_limits = np.argwhere(
    #     np.logical_and(data_hist_time_err > x_limits[0], data_hist_time_err < x_limits[1]))
    data_hist_time_err = data_hist_time_err.flatten()
    data_hist_value_err = data_hist_value_err.flatten()

    return data_hist_value_corr, data_corr, data_hist_value_err, data_err, data_hist_time_corr

def extract_rt_hist_from_df(df, analysed_parameter_list, time_start_stimulus=10, time_end_stimulus=40, duration_dict=None):
    distributions_dict = {}
    for i_param, parameter in enumerate(analysed_parameter_list):
        df_filtered = df[df[analysed_parameter] == parameter]
        df_filtered = df_filtered.query(query_time)

        if duration_dict is None:
            duration = np.sum(
                BehavioralProcessing.get_duration_trials_in_df(df_filtered,
                                                               fixed_time_trial=time_end_stimulus - time_start_stimulus)
            )
        else:
            duration = duration_dict[parameter]

        data_hist_value_corr, data_corr, data_hist_value_err, data_err, data_hist_time_corr = compute_distributions(df_filtered, duration)

        distributions_dict[parameter] = {"corr": data_hist_value_corr, "data_corr": data_corr, "err": data_hist_value_err, "data_err": data_err, "bins": data_hist_time_corr, "duration": duration}

    return distributions_dict

dt_array = np.zeros(len(models_in_group_list))
loss_dict = {}
for i_m, m in enumerate(models_in_group_list):
    dt_array[i_m] = float(m["label_show"].split("=")[-1])
    path_dir = Path(m["path"])

    for i_fish, fish in enumerate(fish_to_include_list):
        df_data = pd.read_hdf(path_dir / f"data_fish_{fish}.hdf5")
        for path_fit in path_dir.glob(f"data_synthetic_fish_{fish}_*.hdf5"):
            df_fit = pd.read_hdf(path_fit)
            break
        df_dict[fish] = {"fit": df_fit, "data": df_data, "color": style.palette["fish_code"][i_fish]}  # BehavioralProcessing.remove_fast_straight_bout(df, threshold_response_time=100)

    plot_section = {"corr": {}, "err": {}}

    df_dict.pop(all_label, None)  # get rid of the df with all data merged

    for i_k, k in enumerate(df_dict.keys()):
        for i_param, parameter in enumerate(analysed_parameter_list):
            # plot
            line_dashes = None
            plot_title = None
            if i_k == 0 and i_m == 0:
                if i_param == 0:
                    plot_title = f"Coh={parameter}%"
                else:
                    plot_title = f"{parameter}%"
            plot_section["corr"][i_param] = fig.create_plot(
                plot_label=plot_label_list[i_plot_label] if i_k == 0 and i_param == 0 and i_m == 0 else None,
                plot_title=plot_title,
                xpos=xpos + i_param * (plot_width_here + padding_plot),
                ypos=ypos + plot_height_small + padding_vertical,
                plot_height=plot_height_here,
                plot_width=plot_width_here,
                xmin=x_limits[0], xmax=x_limits[-1],
                xticks=None, yticks=None,
                yl=m["label_show"] if i_param == 0 else None,
                ymin=-y_limits[-1], ymax=y_limits[-1],
                hlines=[0])
            plot_section["err"][i_param] = plot_section["corr"][i_param]

            if i_param == len(analysed_parameter_list) - 1 and i_k == len(df_dict) - 1 and i_m == len(models_in_group_list) - 1:
                y_location_scalebar = y_limits[-1] / 6
                x_location_scalebar = x_limits[-1] / 6
                plot_section["corr"][i_param].draw_line((1.7, 1.7),
                                                        (y_location_scalebar, y_location_scalebar + 0.1),
                                                        lc="k")
                plot_section["corr"][i_param].draw_text(2, y_location_scalebar, "0.1 events/s",
                                                        textlabel_rotation='vertical', textlabel_ha='left',
                                                        textlabel_va="bottom")

                plot_section["corr"][i_param].draw_line((x_location_scalebar, x_location_scalebar + 0.5),
                                                        (-y_location_scalebar, -y_location_scalebar), lc="k")
                plot_section["corr"][i_param].draw_text(x_location_scalebar, -4 * y_location_scalebar, "0.5 s",
                                                        textlabel_rotation='horizontal', textlabel_ha='left',
                                                        textlabel_va="bottom")

            for config in config_list:
                df = df_dict[k][config["label"]]
                target_distributions_dict = extract_rt_hist_from_df(df, analysed_parameter_list, time_start_stimulus=config["time_start_stimulus"], time_end_stimulus=config["time_end_stimulus"])
                data_hist_time_corr = data_hist_time_err = target_distributions_dict[parameter]["bins"]
                data_hist_value_corr = target_distributions_dict[parameter]["corr"]
                data_hist_value_err = target_distributions_dict[parameter]["err"]

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

                plot_section_corr = plot_section["corr"][i_param]
                plot_section_err = plot_section["err"][i_param]
                plot_section_corr.draw_line(data_hist_time_corr, data_hist_value_corr, lc=lc_correct,
                                            # lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                            lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_correct)
                plot_section_err.draw_line(data_hist_time_err, -1 * data_hist_value_err, lc=lc_incorrect,
                                           # lc=style.palette["fish_code"][i_k], # palette[-1-i_param],
                                           lw=0.75, line_dashes=config["line_dashes"], alpha=alpha_incorrect)

        ypos = ypos - (padding_here + plot_height_small)
xpos = xpos_start
i_plot_label += 1

for i_m, m in enumerate(models_in_group_list):
    path_dir = Path(m["path"])
    for path_fish_experiment in path_dir.glob(f"data_fish_*.hdf5"):
        label_fish = path_fish_experiment.name.split("_")[2].replace(".hdf5", "")

        if label_fish == "all":
            continue
        df_experiment = pd.read_hdf(path_fish_experiment)
        distribution_experiment_dict = extract_rt_hist_from_df(df_experiment, analysed_parameter_list, time_start_stimulus=time_start_stimulus, time_end_stimulus=time_end_stimulus)

        if label_fish not in loss_dict.keys():
            loss_dict[label_fish] = {}

        for path_fish_fit in path_dir.glob(f"data_synthetic_fish_{label_fish}*.hdf5"):
            df_fit = pd.read_hdf(path_fish_fit)
        # it is safer to force in the duration of the simulation, as with higher dt we may have empty trials, not identified
        duration_dict_fit = {p: 30 * 30 for p in analysed_parameter_list}  # duration stimulus in trial * number of trials per coh
        distribution_fit_dict = extract_rt_hist_from_df(df_fit, analysed_parameter_list, time_start_stimulus=time_start_stimulus, time_end_stimulus=time_end_stimulus, duration_dict=duration_dict_fit)

        if label_fish == fish_1_label:
            pass

        loss_dict[label_fish][m["label_show"]] = 0
        for p in analysed_parameter_list:
            loss_dict[label_fish][m["label_show"]] += BehavioralProcessing.kl_divergence_rt_distribution_weight(distribution_experiment_dict[p]["data_corr"],
                                                                                                            distribution_fit_dict[p]["data_corr"],
                                                                                                            resolution=int((x_limits[1] - x_limits[0]) / (0.05)),
                                                                                                            focus_scope=x_limits,
                                                                                                            duration_0=distribution_experiment_dict[p]["duration"],
                                                                                                            duration_1=distribution_fit_dict[p]["duration"],
                                                                                                            order_max_result=True,
                                                                                                            correct_by_area=False)
            loss_dict[label_fish][m["label_show"]] += BehavioralProcessing.kl_divergence_rt_distribution_weight(distribution_experiment_dict[p]["data_err"],
                                                                                                            distribution_fit_dict[p]["data_err"],
                                                                                                            resolution=int((x_limits[1] - x_limits[0]) / (0.05)),
                                                                                                            focus_scope=x_limits,
                                                                                                            duration_0=distribution_experiment_dict[p]["duration"],
                                                                                                            duration_1=distribution_fit_dict[p]["duration"],
                                                                                                            order_max_result=True,
                                                                                                            correct_by_area=False)

loss_array = np.zeros((len(loss_dict), len(models_in_group_list)))
for i_fish, label_fish in enumerate(loss_dict.keys()):
    for i_m, m in enumerate(models_in_group_list):
        loss_array[i_fish, i_m] = loss_dict[label_fish][m["label_show"]] if m["label_show"] in loss_dict[label_fish].keys() else np.nan
loss_mean_array = np.nanmean(loss_array, axis=0)

plot_dt_compare = fig.create_plot(plot_label=plot_label_list[i_plot_label],
                                    xpos=xpos, ypos=ypos,
                                    plot_height=plot_height, plot_width=len(analysed_parameter_list)-1 * (plot_width_here + padding_plot),
                                    xlog=True, xmin=dt_array[0], xmax=dt_array[-1], xl="dt (s)",
                                    yticks=[0, 5, 10], xticks=dt_array,
                                    ymin=0, ymax=10, yl="Loss",
                                  vlines=dt_array
                                  )
i_plot_label += 1
for i_fish in range(len(loss_dict)):
    plot_dt_compare.draw_line(dt_array, loss_array[i_fish, :], lc=style.palette["neutral"][0], lw=0.05)
plot_dt_compare.draw_line(dt_array, loss_mean_array, lc="k", lw=1)


# fig.save(Path.home() / 'Academics' / 'graphics' / 'pictures' / 'figures_for_papers' / 'behavior_model' / "figure_1_experiment_coh_temp.pdf", open_file=True, tight=True)
fig.save(Path.home() / 'Desktop' / "figure_s1_compare_dt.pdf", open_file=True, tight=style.page_tight)
