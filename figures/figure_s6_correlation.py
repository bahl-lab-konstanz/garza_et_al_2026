import itertools
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.constants import alphabet, bcolors
from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis.utils.figure_helper import Figure
from analysis.personal_dirs.Roberto.utils.palette import Palette

# paths
path_dir_control = Path(r"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/weight_nosmooth/_30")
path_dir = Path(r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/5_dpf")
path_corr_tensor_fit = Path(r"/media/roberto/TOSHIBA EXT/Academics/data/benchmarking/weight_nosmooth/_30/results/corr_tensor_fit.pkl")

# configuration analysis
parameter_list = [
        {"label": "noise_sigma",
         "label_show": "diffusion",
         "min": 0,
         "max": 3},
        {"label": 'scaling_factor',
         "label_show": "drift",
         "min": -3,
         "max": 3},
        {"label": 'leak',
         "label_show": "leak",
         "min": -3,
         "max": 3},
        {"label": 'residual_after_bout',
         "label_show": "reset",
         "min": 0,
         "max": 1},
        {"label": 'inactive_time',
         "label_show": "delay",
         "min": 0,
         "max": 1},
    ]
number_bootstraps = int(1e4)  # int(5e3)  #
sample_percentage_size = 1
high_corr_threshold = 0.5
corr_threshold = 0.6
no_corr_threshold = 0.3
p_value_threshold = 0.01

# configurations figure
style = BehavioralModelStyle()
xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start
padding = 0.8  # style.padding
plot_height = style.plot_height
plot_width = style.plot_width
plot_width_short = style.plot_width * 0.8
plot_label_list = alphabet
i_plot_label = 0
style.add_palette("green", Palette.green_short)
style.add_palette("neutral", [Palette.color_neutral])
palette = style.palette["default"]
style.add_palette("fish_code", ["#73489C", "#753B51", "#103882", "#7F0C0C"])

do_basic_computation = True  # keep always True
show_parameter_space = False
show_correlation_matrices = True
show_trajectory_correlation = True

# Make a standard figure
fig = Figure()

if do_basic_computation:
    ##### correlation matrices
    # control synthetic datasets
    model_dict = {}
    model_dict_nofit = {}
    error_list = np.array([])
    for model_filepath in path_dir_control.glob('model_test_*.hdf5'):
        model_filename = str(model_filepath.name)
        if model_filename.endswith("fit.hdf5"):
            model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}
        else:
            model_dict_nofit[model_filename.split("_")[2]] = {"data": model_filepath}

    # sampling
    model_array_sampling = np.zeros((len(parameter_list), len(model_dict_nofit.keys())))
    model_dict_sampling = {p["label"]: [] for p in parameter_list}
    model_dict_sampling["id"] = []
    i_m = 0
    for i_model, id_model in enumerate(model_dict_nofit.keys()):
        df_model_fit_list = pd.read_hdf(model_dict_nofit[id_model]["data"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(parameter_list):
            model_array_sampling[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_sampling[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_sampling["id"].append(id_model)
        i_m += 1

    # fit synthetic fish
    model_array_control = np.zeros((len(parameter_list), len(model_dict.keys())))
    model_dict_control = {p["label"]: [] for p in parameter_list}
    model_dict_control["id"] = []
    i_m = 0
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(parameter_list):
            model_array_control[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_control[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_control["id"].append(id_model)
        i_m += 1
    theta_fish_control_list = list(model_array_control.T)

    # real fish
    model_dict = {}
    error_list = np.array([])
    for model_filepath in path_dir.glob('model_*_fit.hdf5'):
        model_filename = str(model_filepath.name)

        df_temp = pd.read_hdf(model_filepath)
        model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

    model_array = np.zeros((len(parameter_list), len(model_dict.keys())))
    model_dict_fish = {p["label"]: [] for p in parameter_list}
    model_dict_fish["id"] = []
    i_m = 0
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(parameter_list):
            model_array[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
            model_dict_fish[p["label"]].append(df_model_fit[p["label"]][0])
        model_dict_fish["id"].append(id_model)
        i_m += 1

    df_model_sampling_original = pd.DataFrame(model_dict_sampling)
    df_model_sampling_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_sampling_list = BehavioralProcessing.randomly_sample_df(df=df_model_sampling_original,
                                                                     sample_number=number_bootstraps,
                                                                     sample_percentage_size=sample_percentage_size,
                                                                     with_replacement=True)
    relation_tensor_sampling = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

    df_model_control_original = pd.DataFrame(model_dict_control)
    df_model_control_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_control_list = BehavioralProcessing.randomly_sample_df(df=df_model_control_original,
                                                                    sample_number=number_bootstraps,
                                                                    sample_percentage_size=sample_percentage_size,
                                                                    with_replacement=True)
    relation_tensor_control = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

    df_model_fish_original = pd.DataFrame(model_dict_fish)
    df_model_fish_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
    df_model_fish_list = BehavioralProcessing.randomly_sample_df(df=df_model_fish_original,
                                                                 sample_number=number_bootstraps,
                                                                 sample_percentage_size=sample_percentage_size,
                                                                 with_replacement=True)
    relation_tensor_fish = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

    delta_corr_synthetic = np.abs(
        np.array(df_model_control_original.corr()) - np.array(df_model_sampling_original.corr()))
    df_model_synthetic_combined_original = pd.concat((df_model_sampling_original, df_model_control_original))
    df_model_synthetic_combined_list_0 = BehavioralProcessing.randomly_sample_df(
        df=df_model_synthetic_combined_original,
        sample_number=number_bootstraps,
        sample_percentage_size=sample_percentage_size,
        with_replacement=True)
    df_model_synthetic_combined_list_1 = BehavioralProcessing.randomly_sample_df(
        df=df_model_synthetic_combined_original,
        sample_number=number_bootstraps,
        sample_percentage_size=sample_percentage_size,
        with_replacement=True)
    relation_tensor_synthetic_combined_delta = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

    relation_control_original = np.array(df_model_control_original.corr())
    delta_corr_fish = np.abs(np.array(df_model_fish_original.corr()) - relation_control_original)
    df_model_fish_combined_original = pd.concat((df_model_fish_original, df_model_control_original))
    df_model_fish_combined_list_0 = BehavioralProcessing.randomly_sample_df(df=df_model_fish_combined_original,
                                                                            sample_number=number_bootstraps,
                                                                            sample_percentage_size=sample_percentage_size,
                                                                            with_replacement=True)
    df_model_fish_combined_list_1 = BehavioralProcessing.randomly_sample_df(df=df_model_fish_combined_original,
                                                                            sample_number=number_bootstraps,
                                                                            sample_percentage_size=sample_percentage_size,
                                                                            with_replacement=True)
    relation_tensor_fish_combined_delta = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

    for i_df in range(number_bootstraps):
        df_model_sampling = df_model_sampling_list[i_df]
        # df_model_control.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_sampling[i_df] = np.array(df_model_sampling.corr())

        df_model_control = df_model_control_list[i_df]
        # df_model_control.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_control[i_df] = np.array(df_model_control.corr())

        df_model_fish = df_model_fish_list[i_df]
        # df_model_fish.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
        relation_tensor_fish[i_df] = np.array(df_model_fish.corr())

        df_model_synthetic_combined_0 = df_model_synthetic_combined_list_0[i_df]
        corr_synthetic_combined_0 = np.array(df_model_synthetic_combined_0.corr())
        df_model_synthetic_combined_1 = df_model_synthetic_combined_list_1[i_df]
        corr_synthetic_combined_1 = np.array(df_model_synthetic_combined_1.corr())
        relation_tensor_synthetic_combined_delta[i_df] = np.abs(corr_synthetic_combined_0 - corr_synthetic_combined_1)

        df_model_fish_combined_0 = df_model_fish_combined_list_0[i_df]
        corr_fish_combined_0 = np.array(df_model_fish_combined_0.corr())
        df_model_fish_combined_1 = df_model_fish_combined_list_1[i_df]
        corr_fish_combined_1 = np.array(df_model_fish_combined_1.corr())
        relation_tensor_fish_combined_delta[i_df] = np.abs(corr_fish_combined_0 - corr_fish_combined_1)

    # check statistical difference of biology from baseline of the model
    p_value_synthetic = np.mean(relation_tensor_synthetic_combined_delta >= delta_corr_synthetic, axis=0)
    p_value_corr_acceptable = p_value_synthetic > p_value_threshold
    p_value_synthetic[np.triu_indices(len(parameter_list))] = 1

    p_value_fish = np.mean(relation_tensor_fish_combined_delta >= delta_corr_fish, axis=0)
    p_value_fish[np.triu_indices(len(parameter_list))] = 1

    relation_matrix_sampling = np.mean(relation_tensor_sampling, axis=0)
    relation_matrix_sampling[np.triu_indices(len(parameter_list))] = 0
    relation_matrix_sampling_std = np.std(relation_tensor_control, axis=0)

    relation_matrix_control = np.mean(relation_tensor_control, axis=0)
    relation_matrix_control[np.triu_indices(len(parameter_list))] = 0
    relation_matrix_control_std = np.std(relation_tensor_control, axis=0)

    relation_matrix_fish = np.mean(relation_tensor_fish, axis=0)
    relation_matrix_fish[np.triu_indices(len(parameter_list))] = 0
    relation_matrix_fish_std = np.std(relation_tensor_fish, axis=0)

if show_parameter_space:
    ##### scatterplots in parameter space
    model_dict = {}
    error_list = np.array([])
    for model_filepath in path_dir.glob('model_*_fit.hdf5'):
        model_filename = str(model_filepath.name)
        # if label_fish_time is not None:
        #     if model_filename.split("_")[2].endswith(label_fish_time):
        #         model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}
        # else:
        model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}
    # for model_filepath in path_dir.glob(r'model_*.hdf5'):
    #     if not (str(model_filepath).endswith('fit.hdf5') or str(model_filepath).endswith('fit_all.hdf5')):
    #         model_filename = str(model_filepath.name)
    #         model_id = model_filename.split("_")[2]
    #         try:
    #             model_dict[model_id]["target"] = model_filepath
    #         except:
    #             print(f"No fit for model {model_id}")

    model_array = np.zeros((len(parameter_list), len(model_dict.keys())))
    i_m = 0
    for i_model, id_model in enumerate(model_dict.keys()):
        df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
        best_score = np.min(df_model_fit_list['score'])
        df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
        for i_p, p in enumerate(parameter_list):
            model_array[i_p, i_m] = df_model_fit[p["label"]].iloc[0]
        i_m += 1

    for i_y, parameter_y in enumerate(parameter_list):
        for i_x, parameter_x in enumerate(parameter_list):
            y_p = model_array[i_y, :]
            x_p = model_array[i_x, :]

            # process and plot
            plot_xy = fig.create_plot(plot_label=plot_label_list[i_plot_label] if i_x == 0 and i_y == 0 else None,
                                      xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width,
                                      xmin=parameter_x["min"], xmax=parameter_x["max"], xticks=[parameter_x["min"], parameter_x["max"]],
                                      xl=parameter_x["label_show"] if i_y == 0 else None, yl=parameter_y["label_show"] if i_x == 0 else None,
                                      ymin=parameter_y["min"], ymax=parameter_y["max"], yticks=[parameter_y["min"], parameter_y["max"]],
                                      hlines=[0], vlines=[0])
            xpos += padding + plot_width

            plot_xy.draw_scatter(y_p, x_p, elw=0, alpha=0.5)

        xpos = xpos_start
        ypos -= padding + plot_height
    ypos -= padding + plot_height

if show_correlation_matrices:
    # PLOT
    plot_width_matrix = plot_width * 1.6
    plot_size_matrix = plot_width * 1.7
    
    # correlation matrices
    plot_sampling = fig.create_plot(plot_label=alphabet[i_plot_label], plot_title="Correlation matrix\nsampling synthetic",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(parameter_list)-0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(parameter_list)), xticklabels=[p["label_show"].capitalize() for p in parameter_list],
                                   ymin=-0.5, ymax=len(parameter_list)-0.5,
                                   yticks=np.arange(len(parameter_list)), yticklabels=[p["label_show"].capitalize() for p in reversed(parameter_list)],)
    i_plot_label += 1
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(parameter_list))
    x = np.tile(x_, (len(parameter_list), 1))
    y = x.T
    im = plot_sampling.draw_image(relation_matrix_sampling, (-0.5, len(parameter_list)-0.5, len(parameter_list)-0.5, -0.5),
                            colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    plot_control = fig.create_plot(plot_title="Correlation matrix\nfit synthetic",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(parameter_list)-0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(parameter_list)), xticklabels=[p["label_show"].capitalize() for p in parameter_list],
                                   ymin=-0.5, ymax=len(parameter_list)-0.5,
                                   yticks=np.arange(len(parameter_list)), yticklabels=[p["label_show"].capitalize() for p in reversed(parameter_list)],)
    i_plot_label += 1
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(parameter_list))
    x = np.tile(x_, (len(parameter_list), 1))
    y = x.T
    im = plot_control.draw_image(relation_matrix_control, (-0.5, len(parameter_list)-0.5, len(parameter_list)-0.5, -0.5),
                            colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    plot_fish = fig.create_plot(plot_title="Correlation matrix\nexperiment",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(parameter_list)-0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(parameter_list)), xticklabels=[p["label_show"].capitalize() for p in parameter_list],
                                   ymin=-0.5, ymax=len(parameter_list)-0.5)
                                   # yticks=np.arange(len(parameter_list)), yticklabels=[p["label_show"] for p in reversed(parameter_list)],)
    i_plot_label += 1
    xpos += padding*2 + plot_size_matrix

    x_ = np.arange(len(parameter_list))
    x = np.tile(x_, (len(parameter_list), 1))
    y = x.T
    im = plot_fish.draw_image(relation_matrix_fish, (-0.5, len(parameter_list)-0.5, len(parameter_list)-0.5, -0.5),
                            colormap='seismic', zmin=-1, zmax=1, image_interpolation=None)

    divider = make_axes_locatable(plot_fish.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_fish.figure.fig.colorbar(im, cax=cax, orientation='vertical', ticks=[-1, -corr_threshold, 0, corr_threshold, 1])

    # matrices of p-values
    plot_fit = fig.create_plot(plot_title="p-value\ncorrelation fit",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(parameter_list)-0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(parameter_list)), xticklabels=[p["label_show"].capitalize() for p in parameter_list],
                                   ymin=-0.5, ymax=len(parameter_list)-0.5,
                                   yticks=np.arange(len(parameter_list)), yticklabels=[p["label_show"].capitalize() for p in reversed(parameter_list)],)
    i_plot_label += 1
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(parameter_list))
    x = np.tile(x_, (len(parameter_list), 1))
    y = x.T
    im = plot_fit.draw_image(p_value_synthetic, (-0.5, len(parameter_list)-0.5, len(parameter_list)-0.5, -0.5),
                            colormap='gray', zmin=0, zmax=0.1, image_interpolation=None)

    plot_corrected = fig.create_plot(plot_title="p-value\ncorrelation corrected",
                                   xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                   plot_width=plot_size_matrix,
                                   xmin=-0.5, xmax=len(parameter_list)-0.5, xticklabels_rotation=90,
                                   xticks=np.arange(len(parameter_list)), xticklabels=[p["label_show"].capitalize() for p in parameter_list],
                                   ymin=-0.5, ymax=len(parameter_list)-0.5)
                                   # yticks=np.arange(len(parameter_list)), yticklabels=[p["label_show"] for p in reversed(parameter_list)],)
    i_plot_label += 1
    xpos += padding + plot_size_matrix

    x_ = np.arange(len(parameter_list))
    x = np.tile(x_, (len(parameter_list), 1))
    y = x.T
    im = plot_corrected.draw_image(p_value_fish, (-0.5, len(parameter_list)-0.5, len(parameter_list)-0.5, -0.5),
                                   colormap='gray', zmin=0, zmax=0.1, image_interpolation=None)

    divider = make_axes_locatable(plot_corrected.ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plot_corrected.figure.fig.colorbar(im, cax=cax, orientation='vertical', ticks=[0, 0.01, 0.05, 0.1])

    xpos = xpos_start
    ypos -= padding*2 + plot_size_matrix

if show_trajectory_correlation:
    padding_here = 1
    xpos_start_here = xpos
    ypos_start_here = ypos
    # ##### trajecotry correlation
    models_in_age_list = [
        {"label_show": "5dpf",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/5_dpf",
         "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf\data_fish_all.hdf5",
         # None  #
         "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\5_dpf\data_synthetic_fish_all.hdf5",
         # None  #
         "dashes": None,
         "color": "k",
         "alpha": 1},
        {"label_show": "6dpf",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/6_dpf",
         "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\6_dpf\data_fish_all.hdf5",
         # None  #
         "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\6_dpf\data_synthetic_fish_all.hdf5",
         # None  #
         "dashes": None,
         "color": "k",
         "alpha": 0.5},
        {"label_show": "7dpf",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/7_dpf",
         "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\7_dpf\data_fish_all.hdf5",
         # None  #
         "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\7_dpf\data_synthetic_fish_all.hdf5",
         # None  #
         "dashes": None,
         "color": "k",
         "alpha": 1},
        {"label_show": "8dpf",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/8_dpf",
         "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\8_dpf\data_fish_all.hdf5",
         # None  #
         "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\8_dpf\data_synthetic_fish_all.hdf5",
         # None  #
         "dashes": None,
         "color": "k",
         "alpha": 0.5},
        {"label_show": "9dpf",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/age_analysis/week_1-2-3_/9_dpf",
         "path_data": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\9_dpf\data_fish_all.hdf5",
         # None  #
         "path_simulation": r"C:\Users\Roberto\Academics\data\dots_constant\age_analysis\week_1-2-3_\9_dpf\data_synthetic_fish_all.hdf5",
         # None  #
         "dashes": None,
         "color": "k",
         "alpha": 1},
    ]
    models_in_mutation_scn_list = [
        {"label_show": "scn1lab +/+",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/harpaz_2021/scn1lab_NIBR_20200708/attempt_3/wt",
         "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\wt\data_fish_all-wt.hdf5",
         "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\wt\data_synthetic_fish_all-wt.hdf5",
         "dashes": None,
         "color": "k",
         "alpha": 1},
        {"label_show": "scn1lab +/-",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/harpaz_2021/scn1lab_NIBR_20200708/attempt_3/het",
         "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\het\data_fish_all-het.hdf5",
         "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\scn1lab_NIBR_20200708\attempt_3\het\data_synthetic_fish_all-het.hdf5",
         "dashes": None,
         "color": "k",
         "alpha": 1},
    ]
    models_in_mutation_disc_list = [
        {"label_show": "disc1 +/+",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/harpaz_2021/disc1_hetnix/attempt_3/wt",
         "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\wt\data_fish_all-wt.hdf5",
         "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\wt\data_synthetic_fish_all-wt.hdf5",
         "dashes": None,
         "color": "k",
         "alpha": 1},
        {"label_show": "disc1 +/-",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/harpaz_2021/disc1_hetnix/attempt_3/het",
         "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\het\data_fish_all-het.hdf5",
         "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\het\data_synthetic_fish_all-het.hdf5",
         "dashes": None,
         "color": "k",
         "alpha": 1},
        {"label_show": "disc1 -/-",
         "path": r"/media/roberto/TOSHIBA EXT/Academics/data/harpaz_2021/disc1_hetnix/attempt_3/hom",
         "path_data": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\hom\data_fish_all-hom.hdf5",
         "path_simulation": r"C:\Users\Roberto\Academics\data\harpaz_2021\disc1_hetnix\attempt_3\hom\data_synthetic_fish_all-hom.hdf5",
         "dashes": None,
         "color": "k",
         "alpha": 1},
    ]

    def show_trajectory_correlation(models_list, fig, xpos_start_here, ypos_start_here, xl=None, xticklabels=None,
                                    plot_width_here=plot_width_short, xticklabels_rotation=None,
                                    show_title=False, show_scalebar=False, show_ticks=False, p_value_corr_acceptable=None):
        xpos = xpos_start_here
        ypos = ypos_start_here
        parameter_corr_trajectory = np.zeros((len(models_list), len(parameter_list), len(parameter_list)))
        parameter_pval_trajectory = np.zeros((len(models_list), len(parameter_list), len(parameter_list)))
        parameter_corr_trajectory_std = np.zeros((len(models_list), len(parameter_list), len(parameter_list)))
        parameter_corr_trajectory_q25 = np.zeros((len(models_list), len(parameter_list), len(parameter_list)))
        parameter_corr_trajectory_q75 = np.zeros((len(models_list), len(parameter_list), len(parameter_list)))
        combination_list = list(itertools.combinations(range(len(parameter_list)), 2))
        for i_m, m in enumerate(models_list):
            path_dir = m["path"]

            # real fish
            model_dict = {}
            error_list = np.array([])
            for model_filepath in Path(path_dir).glob('model_*_fit.hdf5'):
                model_filename = str(model_filepath.name)

                df_temp = pd.read_hdf(model_filepath)
                model_dict[model_filename.split("_")[2]] = {"fit": model_filepath}

            model_array = np.zeros((len(parameter_list), len(model_dict.keys())))
            model_dict_group = {p["label"]: [] for p in parameter_list}
            model_dict_group["id"] = []
            i_m_ = 0
            for i_model, id_model in enumerate(model_dict.keys()):
                df_model_fit_list = pd.read_hdf(model_dict[id_model]["fit"])
                best_score = np.min(df_model_fit_list['score'])
                df_model_fit = df_model_fit_list.loc[df_model_fit_list['score'] == best_score]
                for i_p, p in enumerate(parameter_list):
                    model_array[i_p, i_m_] = df_model_fit[p["label"]].iloc[0]
                    model_dict_group[p["label"]].append(df_model_fit[p["label"]][0])
                model_dict_group["id"].append(id_model)
                i_m_ += 1

            df_model_group_original = pd.DataFrame(model_dict_group)
            df_model_group_original.set_index('id', inplace=True)  # set the Animal_ID as index to "preserve identity"
            relation_group_original = np.array(df_model_group_original.corr())
            df_model_group_list = BehavioralProcessing.randomly_sample_df(df=df_model_group_original,
                                                                         sample_number=number_bootstraps,
                                                                         sample_percentage_size=sample_percentage_size,
                                                                         with_replacement=True)
            relation_tensor_group = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

            df_model_group_combined_original = pd.concat((df_model_control_original, df_model_group_original))
            df_model_group_combined_list_0 = BehavioralProcessing.randomly_sample_df(df=df_model_group_combined_original,
                                                                         sample_number=number_bootstraps,
                                                                         sample_percentage_size=sample_percentage_size,
                                                                         with_replacement=True)
            df_model_group_combined_list_1 = BehavioralProcessing.randomly_sample_df(df=df_model_group_combined_original,
                                                                         sample_number=number_bootstraps,
                                                                         sample_percentage_size=sample_percentage_size,
                                                                         with_replacement=True)
            relation_tensor_group_combined_delta = np.zeros((number_bootstraps, len(parameter_list), len(parameter_list)))

            for i_df in range(number_bootstraps):
                df_model_group = df_model_group_list[i_df]
                relation_tensor_group[i_df] = np.array(df_model_group.corr())

                df_model_combined_0 = df_model_group_combined_list_0[i_df]
                corr_combined_0 = np.array(df_model_combined_0.corr())
                df_model_combined_1 = df_model_group_combined_list_1[i_df]
                corr_combined_1 = np.array(df_model_combined_1.corr())
                relation_tensor_group_combined_delta[i_df] = np.abs(corr_combined_0 - corr_combined_1)

            # check statistical difference of biology from baseline of the model
            corr_delta_group = np.abs(relation_group_original - relation_control_original)
            # relation_tensor_group_combined_delta = np.abs(relation_tensor_group - relation_tensor_control)
            p_value_group = np.mean(relation_tensor_group_combined_delta >= corr_delta_group, axis=0)

            parameter_corr_trajectory[i_m, :, :] = np.nanmean(relation_tensor_group, axis=0)
            parameter_corr_trajectory_std[i_m, :, :] = np.nanstd(relation_tensor_group, axis=0)
            parameter_corr_trajectory_q25[i_m, :, :] = np.nanquantile(relation_tensor_group, q=0.25, axis=0)
            parameter_corr_trajectory_q75[i_m, :, :] = np.nanquantile(relation_tensor_group, q=0.75, axis=0)
            parameter_pval_trajectory[i_m, :, :] = p_value_group

        x = np.arange(len(models_list))
        i_p1_old = 0
        not_shown_scalebar_yet = True
        for i_p1, i_p2 in combination_list:
            p1 = parameter_list[i_p1]
            p2 = parameter_list[i_p2]
            if i_p1_old != i_p1:
                xpos = xpos_start_here
                ypos -= padding * 2 + plot_height
                i_p1_old = i_p1
            plot_pp = fig.create_plot(plot_title=f"{p1['label_show']}-{p2['label_show']}" if show_title else None,
                                      xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_here,
                                      xmin=-0.3, xmax=len(models_list) - 0.3, xticks=x, xl=xl, xticklabels=xticklabels,
                                      xticklabels_rotation=xticklabels_rotation,
                                      ymin=-1, ymax=1, yticks=[-1, 0, 1] if show_ticks else None,
                                      errorbar_area=False, hlines=[0])
            xpos += padding * 3 + plot_width_short

            plot_pp.draw_line(x, parameter_corr_trajectory[:, i_p1, i_p2], lc="k", yerr=parameter_corr_trajectory_std[:, i_p1, i_p2])
            for i_x_, x_ in enumerate(x):
                if p_value_corr_acceptable[i_p1, i_p2] and parameter_pval_trajectory[i_x_, i_p1, i_p2] < p_value_threshold \
                        and np.abs(parameter_corr_trajectory[i_x_, i_p1, i_p2]) - parameter_corr_trajectory_std[i_x_, i_p1, i_p2] > high_corr_threshold:
                        # and np.min([np.abs(parameter_corr_trajectory_q25[i_m, :, :]), np.abs(parameter_corr_trajectory_q75[i_m, :, :])]) > high_corr_threshold:
                    plot_pp.draw_text(x_, 1, "*")
            # if show_ticks:
            #     plot_pp.draw_text(x[0]-0.5, 0, "0")
            #     plot_pp.draw_text(x[0]-0.5, 1, "+1")
            #     plot_pp.draw_text(x[0]-0.5, -1, "-1")
            if show_scalebar and not_shown_scalebar_yet:
                plot_pp.draw_line(np.ones(2) * x[-1] - 0.2, (-1, -0.8), lc="k", lw=0.8)
                plot_pp.draw_text(x[-1], -1, "Corrected\ncoherence\n0.2", textlabel_ha="left", textlabel_rotation=90)
                not_shown_scalebar_yet = False

        return parameter_corr_trajectory, parameter_corr_trajectory_std, parameter_pval_trajectory

    xpos_start_here = xpos_start
    res_age = show_trajectory_correlation(models_in_age_list, fig, xpos_start_here, ypos_start_here, "Age (dpf)",
                                ["5", "6", "7", "8", "9"], plot_width_short,
                                0, False, False, True, p_value_corr_acceptable)

    xpos_start_here = xpos_start + padding_here * 0.3 + plot_width_short
    res_scn = show_trajectory_correlation(models_in_mutation_scn_list, fig, xpos_start_here, ypos_start_here, "scn1",
                                ["+/+", "+/-"], plot_width_short * len(models_in_mutation_scn_list) / len(models_in_age_list),
                                45, True, False, False, p_value_corr_acceptable)

    xpos_start_here =  xpos_start + padding_here + plot_width_short
    res_disc = show_trajectory_correlation(models_in_mutation_disc_list, fig, xpos_start_here, ypos_start_here, "disc1",
                                ["+/+", "+/-", "-/-"], plot_width_short * len(models_in_mutation_disc_list) / len(models_in_age_list),
                                45, False, False, False, p_value_corr_acceptable)

fig.save(Path.home() / 'Desktop' / "figure_s6_correlation.pdf", open_file=True, tight=True)