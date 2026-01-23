"""
Overview

This script performs a **sensitivity analysis** on Drift Diffusion Model (DDM) parameters
to quantify how perturbations of model parameters affect model fit to behavioral data.

Main functionality:
1. **Configuration & Setup**:
   - Loads paths, plotting styles, and simulation parameters.
   - Defines perturbation steps for sensitivity testing.

2. **Computation (if enabled)**:
   - Iterates over fitted models, perturbs DDM parameters, and re-simulates behavior.
   - Computes model loss (fit error) for each perturbation.
   - Stores results in a dictionary and optionally saves them as a pickle file.

3. **Analysis & Visualization**:
   - Loads sensitivity results (either from computation or file).
   - For each parameter:
       - Aggregates trajectories of losses across fish/models.
       - Performs Mann–Whitney U tests to check significance of perturbations.
       - Visualizes trajectories and highlights significant perturbations.
   - Exports the sensitivity figure as a PDF.

This enables the identification of **robust parameters** vs. **sensitive parameters**
by testing how small changes affect model fit.
"""

import copy
import pickle

import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from dotenv import dotenv_values
from scipy.stats import mannwhitneyu

from garza_et_al_2026.figures.style import BehavioralModelStyle
from garza_et_al_2026.model.core.params import ParameterList, Parameter
from garza_et_al_2026.model.ddm import DDMstable
from garza_et_al_2026.service.fast_functions import count_entries_in_dict
from garza_et_al_2026.service.model_service import ModelService
from garza_et_al_2026.utils.configuration_ddm import ConfigurationDDM
from garza_et_al_2026.utils.configuration_experiment import ConfigurationExperiment
from garza_et_al_2026.utils.constants import Direction, StimulusParameterLabel
from analysis.utils.figure_helper import Figure


# ------------------------------------------------------------
# Environment paths (loaded from .env file)
# ------------------------------------------------------------
env = dotenv_values()
path_dir = Path(env['PATH_DIR'])               # Directory containing model/data files
path_save = Path(env['PATH_SAVE'])             # Output directory for saving figures
path_data = path_dir / "base_dataset_5dpfWT"          # Dataset subdirectory


# ------------------------------------------------------------
# Plotting configurations
# ------------------------------------------------------------
style = BehavioralModelStyle()

# Starting positions for plots on figure grid
xpos_start = style.xpos_start
ypos_start = style.ypos_start
xpos = xpos_start
ypos = ypos_start

# Plot dimensions
plot_height = style.plot_height
plot_height_small = plot_height / 2
plot_width = style.plot_width_large
plot_width_small = style.plot_width_small

# Padding between plots
padding = style.padding

# Color palette for plotting
palette = style.palette["default"]


# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
dt = ConfigurationDDM.dt                                  # Simulation timestep
response_time_label = ConfigurationExperiment.ResponseTimeColumn
analysed_parameter = StimulusParameterLabel.COHERENCE.value
analysed_parameter_list = ConfigurationExperiment.coherence_list

# Perturbation parameters for sensitivity analysis
delta_perturbation = 0.05
perturbation_epsilon = delta_perturbation / 2
perturbation_array = np.arange(-0.5, 0.5 + delta_perturbation, delta_perturbation)

number_of_tests = 1  # Can increase for more robust estimates (e.g., 5)


# ------------------------------------------------------------
# Toggle functionality
# ------------------------------------------------------------
compute_sensitivity = False   # If True: recompute sensitivity analysis
save_result = True            # If True: save/load results


# ------------------------------------------------------------
# Sensitivity computation
# ------------------------------------------------------------
if compute_sensitivity:
    # Define parameter set to be perturbed
    parameters = ParameterList()
    parameters.add_parameter("dt", Parameter(value=0.01))
    parameters.add_parameter("inactive_time", Parameter(min=0, max=1, value=0))
    parameters.add_parameter("residual_after_bout", Parameter(min=0, max=1, value=0))
    parameters.add_parameter("leak", Parameter(min=-3, max=3, value=0))
    parameters.add_parameter("scaling_factor", Parameter(min=-3, max=3, value=1))
    parameters.add_parameter("threshold", Parameter(min=0.001, max=2, value=1))
    parameters.add_parameter("noise_sigma", Parameter(min=0.01, max=3, value=1))

    index_model = 0
    score_list = []
    parameter_variation_list = []
    # Dictionary to store loss values per parameter & perturbation
    loss_dict = {p["label"]: {pert: [] for pert in perturbation_array}
                 for p in ConfigurationDDM.parameter_list}

    # Iterate over all fitted models
    for model_filepath in path_dir.glob('model_*.hdf5'):
        score_fit_model_list = np.zeros(number_of_tests)
        response_time_list = {}
        decision_list = {}
        time_list = {}
        time_trial_list = np.arange(0, ConfigurationExperiment.time_end_trial, dt)

        index_model += 1
        model_filename = str(model_filepath.name)
        path_model = str(model_filepath)

        # Extract model ID from filename
        model_id = model_filename.split("_")[2].replace(".hdf5", "")
        df_model_target = pd.read_hdf(path_model)

        # Get best-fit parameter values from stored model fits
        for p in ConfigurationDDM.parameter_list:
            best_score = np.min(df_model_target['score'])
            df_model_best = df_model_target.loc[df_model_target['score'] == best_score]
            param_best = float(df_model_best[p["label"]])
            getattr(parameters, p["label"]).value = param_best

        parameters_target = copy.deepcopy(parameters)

        # Load corresponding behavioral data
        for data_filepath in path_dir.glob(f'data_fish_{model_id}.hdf5'):
            path_data = str(data_filepath)
            df_data_target = pd.read_hdf(path_data)
            break

        # Iterate over parameters to perturb
        for i_p_to_perturb, p_to_perturb in enumerate(ConfigurationDDM.parameter_list):
            for i_perturbation, perturbation in enumerate(perturbation_array):
                if perturbation == 0:
                    # Store original (unperturbed) model loss
                    original_loss = np.min(df_model_target['score'])
                    loss_dict[p_to_perturb["label"]][perturbation].append(original_loss)
                else:
                    # Perturb parameter and simulate model
                    present_loss = 0
                    for i_test in range(number_of_tests):
                        parameters_perturbed = copy.deepcopy(parameters_target)
                        p_best = getattr(parameters_perturbed, p_to_perturb["label"]).value

                        # Compute perturbation in parameter space
                        p_perturbation = perturbation * (p_to_perturb["max"] - p_to_perturb["min"])
                        p_perturbed = p_best + p_perturbation

                        # Skip if perturbation exceeds parameter bounds
                        if p_perturbed > p_to_perturb["max"] or p_perturbed < p_to_perturb["min"]:
                            present_loss = np.nan
                        else:
                            getattr(parameters_perturbed, p_to_perturb["label"]).value = p_perturbed

                            # Simulate perturbed DDM model
                            model_id_label = f"{model_id}_{int(datetime.now().timestamp())}"
                            ddm_model = DDMstable(
                                parameters_perturbed,
                                trials_per_simulation=ConfigurationDDM.number_trial_per_model_coh,
                                time_experimental_trial=30,
                                scaling_factor_input=1
                            )
                            ddm_model.define_stimulus(time_start_stimulus=10, time_end_stimulus=40)

                            # Run simulations across different coherence values
                            for index_parameter, parameter in enumerate(analysed_parameter_list):
                                response_time_list[parameter] = {}
                                decision_list[parameter] = {}
                                time_list[parameter] = {}
                                print(f"INFO | drift_diffusion naive | simulate model_id {model_id} for {analysed_parameter}={parameter}")
                                for trial in range(ConfigurationDDM.number_trial_per_model_coh):
                                    # Construct constant input signal
                                    input_signal = np.zeros(len(time_trial_list))
                                    for i_time, time in enumerate(time_trial_list):
                                        if ConfigurationExperiment.time_start_stimulus <= time <= ConfigurationExperiment.time_end_stimulus:
                                            input_signal[i_time] += parameter / 100

                                    # Unique trial label
                                    trial_label = trial + index_parameter * ConfigurationDDM.number_trial_per_model_coh

                                    # Run trial simulation
                                    response_time_list[parameter][trial_label], \
                                    decision_list[parameter][trial_label], \
                                    time_list[parameter][trial_label], _ = ddm_model.simulate_trial(
                                        input_signal=input_signal
                                    )

                                # Convert simulation outputs into bout DataFrames
                                df_bout_list = [np.nan for _ in range(count_entries_in_dict(time_list))]
                                index_bout = -1
                                for index_parameter, parameter in enumerate(response_time_list.keys()):
                                    for index_trial, trial in enumerate(response_time_list[parameter].keys()):
                                        for index_time, time in enumerate(time_list[parameter][trial]):
                                            index_bout += 1
                                            flipped_bout_angle = ConfigurationDDM.mean_angle_bout + np.random.normal(scale=22.25)
                                            time_adjusted = time - ConfigurationExperiment.time_end_trial * trial
                                            bout = {
                                                "estimated_orientation_change": flipped_bout_angle,
                                                'start_time': time,
                                                'end_time': time,
                                                "correct_bout": decision_list[parameter][trial][index_time],
                                                StimulusParameterLabel.DIRECTION.value: Direction.LEFT.value,
                                                analysed_parameter: parameter,
                                                response_time_label: response_time_list[parameter][trial][index_time],
                                                "fish_ID": model_id_label,
                                                "model_id": model_id,
                                                "trial": trial
                                            }
                                            df_bout = pd.DataFrame([bout])
                                            df_bout_list[index_bout] = df_bout
                                try:
                                    df_output_data = pd.concat(df_bout_list, ignore_index=True)
                                except ValueError:
                                    continue

                            # Compute fit score vs. real data
                            score_dict = ModelService.compute_score_fit(df_data_target, df_output_data)
                            present_loss += score_dict["score"]

                    present_loss /= number_of_tests
                    loss_dict[p_to_perturb["label"]][perturbation].append(present_loss)

    # Save computed sensitivity results
    if save_result:
        with open(path_data / "sensitivity.pkl", 'wb') as f:
            pickle.dump(loss_dict, f)
else:
    # Load precomputed sensitivity results
    with open(path_data / "sensitivity.pkl", 'rb') as f:
        loss_dict = pickle.load(f)


# ------------------------------------------------------------
# Visualization of sensitivity results
# ------------------------------------------------------------
fig = Figure()

for i_p, p in enumerate(ConfigurationDDM.parameter_list):
    # Extract perturbation values for parameter p
    x = np.array(list(loss_dict[p["label"]].keys()))
    x_show = x * 100  # Convert to percentages
    number_fish = len(loss_dict[p["label"]][-0.5])

    # Collect trajectories across fish/models
    trajectory_array = np.zeros((number_fish, len(x)))
    for i_fish in range(number_fish):
        for i_pert in range(len(x)):
            trajectory_array[i_fish, i_pert] = loss_dict[p["label"]][x[i_pert]][i_fish]

    # Baseline (closest to zero perturbation)
    index_original_distribution = np.argwhere(np.abs(perturbation_array) < perturbation_epsilon)
    original_distribution = np.squeeze(trajectory_array[:, index_original_distribution])

    # Statistical test for each perturbation vs. baseline
    significant_list = []
    for i_pert in range(len(x)):
        if i_pert != index_original_distribution:
            pert_distribution = np.squeeze(trajectory_array[:, i_pert])
            pert_distribution = pert_distribution[~np.isnan(pert_distribution)]

            stat, p_value = mannwhitneyu(original_distribution, pert_distribution, nan_policy="omit")
            print(f"{p['label_show']} | perturbation: {x[i_pert]:.02f} | size: {len(pert_distribution)} | p_value: {p_value:.04f}")
            if p_value < 0.05:
                significant_list.append(x[i_pert])
    significant_list = np.array(significant_list)

    # Create sensitivity plot for parameter p
    plot_p = fig.create_plot(
        plot_title=p['label_show'].capitalize(),
        plot_label=style.get_plot_label(),
        xpos=xpos,
        ypos=ypos,
        plot_height=plot_height,
        plot_width=plot_width,
        errorbar_area=True,
        xl=f"Parameter perturbation (%)" if i_p == len(ConfigurationDDM.parameter_list) - 1 else None,
        xmin=np.min(x_show), xmax=np.max(x_show),
        xticks=[-50, -25, 0, 25, 50] if i_p == len(ConfigurationDDM.parameter_list) - 1 else None,
        yl="Loss", ymin=0, ymax=1, yticks=[0, 0.5, 1], vlines=[0]
    )
    ypos = ypos - plot_height - padding

    # Draw trajectories for all fish
    for i_fish in range(number_fish):
        plot_p.draw_line(x_show, trajectory_array[i_fish], lc=palette[i_p], lw=0.1, alpha=0.5)

    # Scatter points for perturbations
    for i_k, k in enumerate(loss_dict[p["label"]].keys()):
        y = loss_dict[p["label"]][k]
        plot_p.draw_scatter(np.ones_like(y) * x_show[i_k], y, pc=palette[i_p], elw=0, alpha=0.8)

    # Mark significant perturbations (positive side)
    x_positive_significant = np.min(significant_list[significant_list > perturbation_epsilon])
    xlim_positive_significant = np.array([x_positive_significant, np.max(perturbation_array) / 1.5]) * 100
    plot_p.draw_line(xlim_positive_significant, np.ones(2) * 0.85, lc="k")
    plot_p.draw_text(np.mean(xlim_positive_significant), 1.2, "*")
    print(fr"{p['label_show']} | significant for an perturbation of $\pm${x_positive_significant}%")

    # Mark significant perturbations (negative side)
    x_negative_significant = np.max(significant_list[significant_list < perturbation_epsilon])
    xlim_negative_significant = np.array([np.min(perturbation_array) / 1.5, x_negative_significant]) * 100
    plot_p.draw_line(xlim_negative_significant, np.ones(2) * 0.95, lc="k")
    plot_p.draw_text(np.mean(xlim_negative_significant), 1.2, "*")

# -----------------------------------------------------------------------------
# Save final figure
# -----------------------------------------------------------------------------
if save_result:
    fig.save(path_save / "figure_s2_sensitivity.pdf", open_file=False, tight=style.page_tight)
