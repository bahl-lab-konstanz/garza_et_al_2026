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
plot_width = 0.9
xpos_start = 0.5
ypos_start = 0.5
xpos = xpos_start
ypos = ypos_start
padding=1.5
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
    # {"label_fish_time": "-wt",
    #  "label_show": "wt",
    #  "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\wt",
    #  "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\wt\data_fish_all-wt.hdf5",
    #  "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\wt\data_synthetic_fish_wt_merged.hdf5",
    #  "dashes": None,
    #  "color": "k",
    #  "alpha": 1},
    # {"label_fish_time": "-het",
    #  "label_show": "scn1lab +/-",
    #  "path": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\het",
    #  "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\het\data_fish_all-het.hdf5",
    #  "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_2\het\data_synthetic_fish_het_merged.hdf5",
    #  "dashes": (2, 4),
    #  "color": "#004AAA",
    #  "alpha": 1,  # 0.25
    #  },
    # {"label_fish_time": "",
    #  "label_show": "WT",
    #  "path": r"C:\Users\Roberto\Academics\data\dots_constant\mutant_analysis\from_fiona\WT\attempt_1",
    #  "path_data": None,  # r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_0\wt\data_fish_all-wt.hdf5",
    #  "path_simulation": None,  # r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_0\wt\data_synthetic_fish_all-wt_fit.hdf5",
    #  "dashes": None,
    #  "color": "k",
    #  "alpha": 1},
    # {"label_fish_time": "",
    #  "label_show": "KO",
    #  "path": r"C:\Users\Roberto\Academics\data\dots_constant\mutant_analysis\from_fiona\KO\attempt_1",
    #  "path_data": None,  # r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_0\het\data_fish_all-het.hdf5",
    #  "path_simulation": None,  # r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_0\het\data_synthetic_fish_all-het_fit.hdf5",
    #  "dashes": (2, 4),
    #  "color": "#004AAA",
    #  "alpha": 1},
    {"label_show": "5dpf",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\5_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\5_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\5_dpf\data_synthetic_fish_all.hdf5",
     "path_control": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\5_dpf\synthetic_control_nofit\data_synthetic_test_all.hdf5",
     # None  #
     "dashes": None,
     "color": "k",
     "alpha": 1},
    {"label_show": "6dpf",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\6_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\6_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\6_dpf\data_synthetic_fish_all.hdf5",
     "path_control": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\6_dpf\synthetic_control_nofit\data_synthetic_test_all.hdf5",
     # None  #
     "dashes": None,
     "color": "k",
     "alpha": 0.5},
    {"label_show": "7dpf",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\7_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\7_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\7_dpf\data_synthetic_fish_all.hdf5",
     "path_control": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\7_dpf\synthetic_control_nofit\data_synthetic_test_all.hdf5",
     # None  #
     "dashes": (2, 4),
     "color": "k",
     "alpha": 1},
    {"label_show": "8dpf",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\8_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\8_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\8_dpf\data_synthetic_fish_all.hdf5",
     "path_control": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\8_dpf\synthetic_control_nofit\data_synthetic_test_all.hdf5",
     # None  #
     "dashes": (2, 4),
     "color": "k",
     "alpha": 0.5},
    {"label_show": "9dpf",
     "path": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\9_dpf",
     "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\9_dpf\data_fish_all.hdf5",
     # None  #
     "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3\9_dpf\data_synthetic_fish_all.hdf5",
     "path_control": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\9_dpf\synthetic_control_nofit\data_synthetic_test_all.hdf5",
     # None  #
     "dashes": (0.1, 3),
     "color": "k",
     "alpha": 1},
]

for m in models_in_age_list:
    try:
        m["df_data"] = pd.read_hdf(m["path_data"])

        m["df_simulation"] = pd.read_hdf(m["path_simulation"])
        # m["df_simulation"].reset_index(inplace=True)

        m["df_control"] = pd.read_hdf(m["path_control"])
        # m["df_control"].reset_index(inplace=True)

        # sequential_index = pd.Index(range(0, len(m["df_data"])), name="Seq_Index")
        # m["df_data"].index = pd.MultiIndex.from_arrays(
        #     [sequential_index] + [m["df_data"].index.get_level_values(i) for i in range(m["df_data"].index.nlevels)],
        #     names=['Seq_Index'] + m["df_data"].index.names)
        # m["df_data"]["same_direction_as_previous_bout"] = [1 if s else 0 for s in m["df_data"]["same_as_previous"]]
        #
        # df_data_unindexed = m["df_data"].reset_index(allow_duplicates=True)
        # same_as_previous = [0]
        # same_as_previous.extend(
        #     [1 if df_data_unindexed.loc[i_row, CorrectBoutColumn] == df_data_unindexed.loc[
        #         i_row - 1, CorrectBoutColumn] else 0 for
        #      i_row in
        #      range(1, len(df_data_unindexed))])
        # m["df_data"]["same_direction_as_previous_bout"] = same_as_previous
        #
        # same_as_previous = [0]
        # same_as_previous.extend(
        #     [1 if m["df_simulation"].loc[i_row, CorrectBoutColumn] == m["df_simulation"].loc[i_row - 1, CorrectBoutColumn] else 0 for
        #      i_row in
        #      range(1, len(m["df_simulation"]))])
        # m["df_simulation"]["same_direction_as_previous_bout"] = same_as_previous
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

show_psychometric_curve = True
show_coherence_vs_interbout_interval = True

if show_psychometric_curve:
    # plot_height = plot_height

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

        df_control = m["df_control"]
        df_control_filtered = df_control.query(query_time)
        df_control_filtered = df_control_filtered[df_control_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_controls = len(df_control_filtered["fish_ID"].unique())
        parameter_list_control, correct_bout_list_control, std_correct_bout_list_control = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_control_filtered, analysed_parameter=analysed_parameter)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])
        parameter_list_control = np.array([int(p) for p in parameter_list_control])

        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None,
                                 plot_title=m["label_show"],
                                 xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list),
                                 # xl=analysed_parameter_label,
                                 xticks=None,  # xticks=[int(p) for p in analysed_parameter_list],
                                 yl="accuracy [-]" if i_m == 0 else None, ymin=0, ymax=1, yticks=[0, 0.5, 1] if i_m == 0 else None, hlines=[0.5])

        # draw
        plot_0.draw_line(x=parameter_list_data, y=correct_bout_list_data,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_data),  # / np.sqrt(number_individuals),
                         lc=m["color"], lw=1, alpha=m["alpha"])
        plot_0.draw_line(x=parameter_list_sim, y=correct_bout_list_sim,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_sim),  # / np.sqrt(number_models),
                         lc=m["color"], lw=1, alpha=m["alpha"], line_dashes=(1, 2))
        plot_0.draw_line(x=parameter_list_control, y=correct_bout_list_control,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_control),  # / np.sqrt(number_models),
                         lc="r", lw=1, alpha=m["alpha"], line_dashes=(0.1, 3))

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width

    i_plot_label += 1
    ypos = ypos - padding - plot_height
    xpos = xpos_start

if show_coherence_vs_interbout_interval:
    # plot_height = 1
    # plot_width = 1

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

        df_control = m["df_control"]
        df_control_filtered = df_control.query(query_time)
        df_control_filtered = df_control_filtered[df_control_filtered[analysed_parameter].isin(analysed_parameter_list)]
        number_controls = len(df_control_filtered["fish_ID"].unique())
        parameter_list_control, correct_bout_list_control, std_correct_bout_list_control = BehavioralProcessing.compute_quantities_per_parameters_multiple_fish(
            df_control_filtered, analysed_parameter=analysed_parameter, column_name=ResponseTimeColumn)

        parameter_list_data = np.array([int(p) for p in parameter_list_data])
        parameter_list_sim = np.array([int(p) for p in parameter_list_sim])
        parameter_list_control = np.array([int(p) for p in parameter_list_control])

        # plot
        plot_0 = fig.create_plot(plot_label=alphabet[i_plot_label] if i_m == 0 else None, xpos=xpos, ypos=ypos, plot_height=plot_height,
                                 plot_width=plot_width,
                                 errorbar_area=True,
                                 xl=analysed_parameter_label, xmin=min(analysed_parameter_list), xmax=max(analysed_parameter_list), xticks=[int(p) for p in analysed_parameter_list],
                                 yl="interbout interval [s]" if i_m == 0 else None, ymin=0, ymax=5, yticks=[0, 2.50, 5] if i_m == 0 else None)
        i_plot_label += 1

        # draw
        plot_0.draw_line(x=parameter_list_data, y=interbout_interval_list_data,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_data),  # / np.sqrt(number_individuals),
                         lc=m["color"], lw=1, alpha=m["alpha"], label=f"data" if i_m == len(models_in_age_list)-1 else None)
        plot_0.draw_line(x=parameter_list_sim, y=interbout_interval_list_sim,
                         errorbar_area=True, yerr=np.array(std_interbout_interval_list_sim),  # / np.sqrt(number_models),
                         lc=m["color"], lw=1, alpha=m["alpha"], line_dashes=(1, 2), label=f"simulation" if i_m == len(models_in_age_list)-1 else None)
        plot_0.draw_line(x=parameter_list_control, y=correct_bout_list_control,
                         errorbar_area=True, yerr=np.array(std_correct_bout_list_control),  # / np.sqrt(number_models),
                         lc="r", lw=1, alpha=m["alpha"], line_dashes=(0.1, 3), label="control" if i_m == len(models_in_age_list)-1 else None)

        if i_m == len(models_in_age_list) - 1:
            pad = padding
        else:
            pad = padding_short
        xpos = xpos + pad + plot_width

    i_plot_label += 1
    ypos = ypos - padding - plot_height
    xpos = xpos_start

fig.save(pathlib.Path.home() / 'Desktop' / f"figure_s4_control_fit_fish.pdf", open_file=True, tight=True)