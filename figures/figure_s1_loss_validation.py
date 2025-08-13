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
    "label": "Loss",
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

models_in_group_list = [
    {"label_show": r"$D_{KL}^*$",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf\data_fish_all.hdf5",
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf\data_synthetic_fish_all.hdf5",
     "ylim": [0, 20]},
    {"label_show": r"$D_{KL}$",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf_dkl",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf_dkl\data_fish_all.hdf5",
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf_dkl\data_synthetic_fish_all.hdf5",
     "ylim": [0, 500]},
]

for m in models_in_group_list:
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

if show_loss_reduction:
    ypos = ypos - padding_short
    for i_group, models_in_group in enumerate(models_in_group_list):
        loss_list = []
        loss_start_list = []
        loss_end_list = []
        ylim = models_in_group["ylim"]
        for path_error in pathlib.Path(models_in_group["path"]).glob("error_fish_*.hdf5"):
            df_error = pd.read_hdf(path_error)
            loss_start_list.append(df_error["score"][0])
            # loss_start = df_error["score"][0]

            loss_end = df_error["score"][len(df_error)-1]
            loss_end_list.append(loss_end)
            loss_list.append(loss_end)

        # plot_title = f"{models_in_group['label_show']}\n" fr"{np.mean(loss_list):0.02f}$\pm${np.std(loss_list):0.02f}"
        plot_title = models_in_group['label_show']
        plot_loss = fig.create_plot(plot_label=alphabet[i_plot_label] if i_group == 0 else None, plot_title=plot_title,
                                    xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                    ymin=ylim[0], ymax=ylim[-1], yticks=[ylim[0], int(np.mean(ylim)), ylim[-1]],
                                    yl=f"Loss" if i_group == 0 else None,
                                    xl="Iteration", xmin=0.5, xmax=2.5, xticks=[1, 2], xticklabels=["0", "1500"])
        ypos = ypos
        xpos += plot_width + padding_short

        for i_loss in range(len(loss_list)):
            loss_start = loss_start_list[i_loss]
            loss_end = loss_end_list[i_loss]
            plot_loss.draw_line((1, 2), (loss_start, loss_end), lc="k", lw=0.05, alpha=0.5)
            plot_loss.draw_scatter((1, 2), (loss_start, loss_end), ec="k", pc="k", alpha=0.5)

        # plot_loss.draw_text(1.5, 20, fr"{np.mean(loss_list):0.02f}$\pm${np.std(loss_list):0.02f}")
    ypos = ypos - padding - plot_height
    xpos = xpos_start
    i_plot_label += 1

if show_psychometric_curve:
    # plot_height = plot_height

    for i_m, m in enumerate(models_in_group_list):
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

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, correct_bout_list_sim, std_correct_bout_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(df_simulation_filtered, analysed_parameter=analysed_parameter)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 # xl=analysed_parameter_label,
                                 xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
                                 xticks=None,  # xticks=[int(p) for p in analysed_parameter_list],
                                 yl="Percentage\ncorrect swims (%)" if i_m == 0 else None, ymin=45, ymax=100, yticks=[50, 75, 100] if i_m == 0 else None, hlines=[50])

        # draw
        plot_0.draw_line(x=parameter_list_data, y=np.array(correct_bout_list_data)*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data)*100,  # / np.sqrt(number_individuals),
                         lc="k", lw=1)
        plot_0.draw_line(x=parameter_list_sim, y=np.array(correct_bout_list_sim)*100,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim)*100,  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2))

        if i_m == len(models_in_group_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width

    i_plot_label += 1
    ypos = ypos - padding - plot_height*2
    xpos = xpos_start

if show_coherence_vs_interbout_interval:
    plot_height_here = plot_height * 2
    # plot_width = 1

    for i_m, m in enumerate(models_in_group_list):
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

        df_simulation = m["df_simulation"]
        df_simulation_filtered = df_simulation.query(query_time)
        df_simulation_filtered = df_simulation_filtered[df_simulation_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_models = len(df_simulation_filtered["fish_ID"].unique())
        parameter_list_sim, interbout_interval_list_sim, std_interbout_interval_list_sim = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
                df_simulation_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])

        # plot
        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height_here,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list), xticks=[int(p) for p in analysed_parameter_list],
                                 yl="Interbout interval (s)" if i_m == 0 else None, ymin=0, ymax=6, yticks=[0, 1.5, 3, 4.5, 6] if i_m == 0 else None)

        # draw
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),  # / np.sqrt(number_individuals),
                         lc="k", lw=1, label=f"data" if i_m == 0 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),  # / np.sqrt(number_models),
                         lc="k", lw=1, line_dashes=(1, 2), label=f"simulation" if i_m == 0 else None)

        if i_m == len(models_in_group_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width

    i_plot_label += 1
    ypos = ypos - padding - plot_height_here
    xpos = xpos_start


fig.save(pathlib.Path.home() / 'Desktop' / f"figure_s1_loss_validation.pdf", open_file=True, tight=True)


