import numpy as np

from analysis.personal_dirs.Roberto.utils.service.behavioral_processing import BehavioralProcessing
from analysis.personal_dirs.Roberto.utils.constants import StimulusParameterLabel, CorrectBoutColumn, ResponseTimeColumn


class ModelService():
    @ staticmethod
    # test if I want to check simulation against data coming from a specific time window within the trial
    def is_window(key: str):
        try:
            return key.startswith("window")
        except AttributeError:
            return False

    @classmethod
    def compute_score_fit(cls, df_0, df_1, analysed_parameter=StimulusParameterLabel.COHERENCE.value, resolution=40, focus_scope=(0, 2), time_selection=None,
                          missing_data_score=100, time_experimental_trial=30, plot_distributions=False):
        score_dict = {}
        score = 0
        try:
            if time_selection is not None:
                if "start" in time_selection.keys():
                    df_0 = df_0.query(f"start_time > {time_selection['start']}")
                    df_1 = df_1.query(f"start_time > {time_selection['start']}")
                if "end" in time_selection.keys():
                    df_0 = df_0.query(f"end_time < {time_selection['end']}")
                    df_1 = df_1.query(f"end_time < {time_selection['end']}")
            analysed_parameter_list = set(df_0[analysed_parameter].unique()) & set(df_1[analysed_parameter].unique())
            for key in analysed_parameter_list:
                score_dict[key] = {}
                df_0_key = df_0[df_0[analysed_parameter] == key]
                df_1_key = df_1[df_1[analysed_parameter] == key]

                df_0_correct = df_0_key[df_0_key[CorrectBoutColumn] == 1]
                df_1_correct = df_1_key[df_1_key[CorrectBoutColumn] == 1]
                df_0_error = df_0_key[df_0_key[CorrectBoutColumn] == 0]
                df_1_error = df_1_key[df_1_key[CorrectBoutColumn] == 0]

                df_0_correct_response_time_list = np.array(df_0_correct[ResponseTimeColumn])
                df_1_correct_response_time_list = np.array(df_1_correct[ResponseTimeColumn])
                df_0_error_response_time_list = np.array(df_0_error[ResponseTimeColumn])
                df_1_error_response_time_list = np.array(df_1_error[ResponseTimeColumn])

                try:
                    duration_0 = np.sum(
                        BehavioralProcessing.get_duration_trials_in_df(df_0_key,
                                                                       fixed_time_trial=time_experimental_trial))  # seconds
                except KeyError:
                    duration_0 = len(df_0_key['trial'].unique()) * time_experimental_trial # seconds

                try:
                    duration_1 = np.sum(
                        BehavioralProcessing.get_duration_trials_in_df(df_1_key,
                                                                       fixed_time_trial=time_experimental_trial))  # seconds
                except KeyError:
                    duration_1 = len(df_1_key['trial'].unique()) * time_experimental_trial # seconds

                KL_divergence_response_correct = BehavioralProcessing.kl_divergence_rt_distribution_weight(df_0_correct_response_time_list,
                                                                                                    df_1_correct_response_time_list,
                                                                                                    resolution=resolution,
                                                                                                    focus_scope=focus_scope,
                                                                                                    duration_0=duration_0,
                                                                                                    duration_1=duration_1,
                                                                                                    order_max_result=True,
                                                                                                    correct_by_area=False,
                                                                                                    plot_distributions=plot_distributions)
                score_dict[key]["corr"] = KL_divergence_response_correct
                KL_divergence_response_error = BehavioralProcessing.kl_divergence_rt_distribution_weight(df_0_error_response_time_list,
                                                                                                  df_1_error_response_time_list,
                                                                                                  resolution=resolution,
                                                                                                  focus_scope=focus_scope,
                                                                                                  duration_0=duration_0,
                                                                                                  duration_1=duration_1,
                                                                                                  order_max_result=True,
                                                                                                  correct_by_area=False,
                                                                                                  plot_distributions=plot_distributions)
                score_dict[key]["err"] = KL_divergence_response_error

                score = score + KL_divergence_response_correct + KL_divergence_response_error  # + np.abs(self.parameters.leak.value)**2  # + abs(np.log(area_target/area_simulation))  # + 10*(1-float(self.input_signal.label)/100) * self.parameters.scaling_factor.value  # + abs(area_simulation - area_target) + abs(median_rt_target - median_rt_simulation)

        except ValueError:
            score = missing_data_score + np.random.normal(scale=missing_data_score)
        score_dict["score"] = score

        return score_dict