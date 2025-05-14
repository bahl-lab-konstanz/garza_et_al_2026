import matplotlib
# matplotlib.use("macosx")
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import scipy
import pathlib

from dotenv import dotenv_values
from scipy.optimize import curve_fit
from scipy.stats import norm, laplace

from analysis.personal_dirs.Roberto.plot.paper_behavior_model.behavioral_model_style import BehavioralModelStyle
from analysis.personal_dirs.Roberto.utils.fast_functions import normpdf
from analysis.personal_dirs.Roberto.utils.service.statistics_service import StatisticsService
from analysis_helpers.analysis.personal_dirs.Roberto.utils.constants import palette_0, alphabet
from analysis_helpers.analysis.personal_dirs.Roberto.utils.palette import Palette
from analysis_helpers.analysis.utils.figure_helper import Figure

# parameters
show_objective_function_vs_iterations = True
show_repeatability = True

# env
env = dotenv_values()
path_dir = pathlib.Path(env['PATH_DIR'])

# Make a standard figure
fig = Figure()

style = BehavioralModelStyle
xpos_start = style.xpos_start
ypos_start =style.ypos_start
xpos = xpos_start
ypos = ypos_start
padding = 1.5  # style.padding
plot_height = style.plot_height
plot_width = style.plot_width * 3/2
i_plot_label = 3
show_best_model = True
palette = style.palette["default"]

# parameters
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

if show_objective_function_vs_iterations:
    plot_width_a = 1
    plot_width_bc = 1
    df = pd.DataFrame()
    model_list = []
    loss = []
    parameter_error = {p["param"]: {} for p in parameter_list}
    parameter_error_trajectory_list_dict = {p["param"]: [] for p in parameter_list}

    present_fitting_run = 0
    for i_error, error_path in enumerate(path_dir.glob("error_test_*_fit.hdf5")):
        if error_path.is_dir():
            continue

        df_error = pd.read_hdf(str(error_path))
        if len(df_error) < 3:
            continue

        for index_parameter, parameter in enumerate(parameter_list):
            parameter_error_trajectory_list_dict[parameter['param']].append(np.array(df_error[f"{parameter['param']}_error"]))

        loss.append(np.array(df_error["score"]))

    parameter_error_trajectory_list_dict["score"] = np.array(loss)
    for p in parameter_list:
        parameter_error_trajectory_list_dict[p['param']] = np.array(parameter_error_trajectory_list_dict[p['param']])

    loss_trajectory_list = parameter_error_trajectory_list_dict["score"]
    loss_mean = np.array([np.mean(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
    loss_std = np.array([np.std(parameter_error_trajectory_list_dict["score"][:, i]) for i in range(parameter_error_trajectory_list_dict["score"].shape[1])])
    iteration_list = np.arange(len(loss_mean))

    parameter_error_mean = {}
    parameter_error_std = {}

    parameter_error_mean["score"] = loss
    for index_parameter, parameter in enumerate(parameter_list):
        parameter_error_mean[parameter["param"]] = np.array([np.mean(parameter_error_trajectory_list_dict[parameter["param"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["param"]].shape[1])])
        parameter_error_std[parameter["param"]] = np.array([np.std(parameter_error_trajectory_list_dict[parameter["param"]][:, i]) for i in range(parameter_error_trajectory_list_dict[parameter["param"]].shape[1])])

    xpos = xpos_start

    plot_0a = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height,
                             plot_width=plot_width_a, errorbar_area=True,
                             xl=None, xmin=-0.5, xmax=1500.5, xticks=[0, 1500],  # xticks=[0, 500, 1000, 1500],
                             yl="loss", ymin=0, ymax=3, yticks=[0, 1.5, 3])

    for loss_trajectory in loss_trajectory_list:
        plot_0a.draw_line(x=iteration_list[:len(loss_trajectory)], y=loss_trajectory, lc=Palette.color_neutral, lw=0.05,
                         alpha=0.5)

    plot_0a.draw_line(x=iteration_list, y=loss_mean, lc=palette_0[-1], lw=0.75)

    xpos = xpos + plot_width_a + padding
    ypos = ypos

    error_array_start = parameter_error_trajectory_list_dict["score"][:, 0]
    error_array_end = parameter_error_trajectory_list_dict["score"][:, -1]
    data_hist_value_start, data_hist_time_start = StatisticsService.get_hist(error_array_start, bins=np.arange(0, 3, 0.1), density=True, center_bin=True)
    data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(0, 3, 0.1), density=True, center_bin=True)

    plot_0b = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_bc,
                              errorbar_area=False, ymin=0, ymax=3,
                              yticks=[0, 1.5, 3],
                              yl=f"loss\nat iteration 0",
                              xl=None, xmin=0, xmax=50, xticks=[0, 25, 50], hlines=[0])
    plot_0b.draw_line(data_hist_value_start * 100, data_hist_time_start, lc=Palette.color_neutral, elw=1)

    xpos = xpos + plot_width_bc + padding
    ypos = ypos

    plot_0c = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_bc,
                              errorbar_area=False, ymin=0, ymax=3,
                              yticks=[0, 1.5, 3],
                              yl=f"loss\nat iteration 1500",
                              xl=None, xmin=0, xmax=50, xticks=[0, 25, 50], hlines=[0])
    plot_0c.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=Palette.color_neutral, elw=1)

    # xpos += padding + plot_width
    # ypos += padding + plot_height
    i_plot_label += 1
    xpos = xpos_start
    ypos = ypos - padding - plot_height

    df_noise_dict = {p["param"]: {} for p in parameter_list}
    p_final_pdf = {p["param"]: None for p in parameter_list}

    # plotting
    for index_parameter, parameter in enumerate(parameter_list):
        plot_na = fig.create_plot(plot_label=alphabet[i_plot_label], xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_a, errorbar_area=True,
                                xl="iteration" if index_parameter==len(parameter_list)-1 else None, xmin=-0.5, xmax=1500.5, xticks=[0, 1500],  # xticks=[0, 500, 1000, 1500],
                                yl=f"error {parameter['label']}", ymin=-1, ymax=1, yticks=[-1, 0, 1])

        for parameter_error_trajectory in parameter_error_trajectory_list_dict[parameter['param']]:
            plot_na.draw_line(x=iteration_list[:len(parameter_error_trajectory)], y=parameter_error_trajectory, lc=palette[index_parameter], lw=0.05, alpha=0.5)

        plot_na.draw_line(x=iteration_list, y=parameter_error_mean[parameter["param"]], lc="k", lw=0.75)

        xpos = xpos + plot_width_a + padding
        ypos = ypos

        error_array_start = parameter_error_trajectory_list_dict[parameter["param"]][:, 0]
        error_array_end = parameter_error_trajectory_list_dict[parameter["param"]][:, -1]
        data_hist_value_start, data_hist_time_start = StatisticsService.get_hist(error_array_start, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)
        data_hist_value_end, data_hist_time_end = StatisticsService.get_hist(error_array_end, bins=np.arange(-1, 1, 0.05), density=True, center_bin=True)

        plot_nb = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_bc,
                                  errorbar_area=False, ymin=-1, ymax=1,
                                  yticks=[-1, 0, 1],
                                  yl=f"error {parameter['label']}\nat iteration 0",
                                  xl="percentage models" if index_parameter==len(parameter_list)-1 else None, xmin=0, xmax=50, xticks=[0, 25, 50], hlines=[0])
        plot_nb.draw_line(data_hist_value_start * 100, data_hist_time_start, lc=palette[index_parameter], elw=1)

        xpos = xpos + plot_width_bc + padding
        ypos = ypos

        plot_nc = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_bc,
                                  errorbar_area=False, ymin=-1, ymax=1,
                                  yticks=[-1, 0, 1],
                                  yl=f"error {parameter['label']}\nat iteration 1500",
                                  xl="percentage models" if index_parameter==len(parameter_list)-1 else None, xmin=0, xmax=50, xticks=[0, 25, 50], hlines=[0])
        plot_nc.draw_line(data_hist_value_end * 100, data_hist_time_end, lc=palette[index_parameter], elw=1)

        # res = curve_fit(normpdf, data_hist_time_end[int(len(data_hist_time_end) / 2 - 5):int(len(data_hist_time_end) / 2 + 5)], data_hist_value_end[int(len(data_hist_time_end) / 2 - 5):int(len(data_hist_time_end) / 2 + 5)], p0=[0, 0.01])
        # mu, std = res[0]
        # mu, std = norm.fit(error_array_end)
        mu, b = laplace.fit(error_array_end)
        df_noise_dict[parameter['param']]["label"] = "laplace"
        df_noise_dict[parameter['param']]["mu"] = mu
        df_noise_dict[parameter['param']]["b"] = b
        y_dist = laplace.pdf(data_hist_time_end, mu, b)
        # y_dist = normpdf(data_hist_time_end, mu, std)
        y_dist /= np.sum(y_dist)
        p_final_pdf[parameter["param"]] = y_dist

        plot_nc.draw_line(y_dist * 100, data_hist_time_end, lc="k", elw=0.3, line_dashes=(2, 4), alpha=1)

        # xpos += padding + plot_width
        # ypos += padding + plot_height
        i_plot_label += 1
        xpos = xpos_start
        ypos = ypos - padding - plot_height

    df_noise = pd.DataFrame(df_noise_dict)
    df_noise.to_hdf(str(pathlib.Path.home() / 'Desktop' / "df_noise"), key="estimation_error")

if show_repeatability:
    path_dir_repeat = pathlib.Path(r"C:\Users\Roberto\Academics\data\benchmarking\test_repeatability")
    test_id = "006"
    plot_width_here = 1.5
    for path in path_dir_repeat.glob(f"model_test_{test_id}_*.hdf5"):
        if "_fit." in path.name:
            continue

        # test_id = path.name.split("_")[2].replace(".hdf5", "")
        df_target = pd.read_hdf(path)

        for i_p, p in enumerate(parameter_list):
            p_target = df_target[p["param"]]
            plot_p = fig.create_plot(plot_label=alphabet[i_plot_label] if i_p == 0 else None,
                                     xpos=xpos, ypos=ypos, plot_height=plot_height, plot_width=plot_width_here,
                                     errorbar_area=False, ymin=p["min"], ymax=p["max"],
                                     yticks=[p["min"], p["mean"], p["max"]],
                                     yl=f"{p['label']}",
                                     xl="iteration",  # if i_p == len(parameter_list)-1 else None,
                                     xmin=0.5, xmax=2.5,
                                     xticks=[1, 2],  # if i_p == len(parameter_list)-1 else None,
                                     xticklabels=["0", "1500"],  # if i_p == len(parameter_list)-1 else None,
                                     # xticklabels_rotation=45 if i_p == len(parameter_list)-1 else None,
                                     hlines=[df_target[p["param"]][0]])
            # ypos -= (plot_height + padding)
            xpos += (plot_width_here + padding)
            if i_p == 0:
                i_plot_label += 1

            p_fit_end_list = []
            for path_fit in path_dir_repeat.glob(f"error_test_{test_id}_*_fit.hdf5"):
                df_fit = pd.read_hdf(path_fit)
                p_fit_start = df_fit[f"{p['param']}_value"][0]
                p_fit_end = df_fit[f"{p['param']}_value"][len(df_fit)-1]
                p_fit_end_list.append(p_fit_end)

                plot_p.draw_line((1, 2), (p_fit_start, p_fit_end), lc=palette[i_p], lw=0.05, alpha=0.5)
                plot_p.draw_scatter((1, 2), (p_fit_start, p_fit_end), pc=palette[i_p], ec=palette[i_p])

            plot_p.draw_scatter([2], [np.median(p_fit_end_list)], pc="k")


# fig.save(pathlib.Path.home() / 'Academics' / 'graphics' / 'pictures' / 'figures_for_papers' / 'behavior_model' / "figure_2_temp_per.pdf", open_file=True, tight=True)
fig.save(pathlib.Path.home() / 'Desktop' / "figure_2_update.pdf", open_file=True, tight=style.page_tight)
