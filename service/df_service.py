import pandas as pd


class DFService():
    @staticmethod
    def update_df(df_new=None, df_start=None, path_df_new=None, path_df_start=None, save_result=False, path_save=None,
                  file_name_save='data.hdf5'):
        assert (df_start is not None) or (path_df_start is not None), \
            'either df_start or path_df_start must be defined'
        assert not ((df_start is not None) and (path_df_start is not None)), \
            'define only one input between df_start and path_df_start'
        assert (df_new is not None) or (path_df_new is not None), \
            'either df_new or path_df_new must be defined'
        assert not ((df_new is not None) and (path_df_new is not None)), \
            'define only one input between df_new and path_df_new'
        assert (not save_result) or (save_result and path_save is not None), \
            'if save_result input is set to True, path_save input has to be defined'

        if df_start is not None and isinstance(df_start, pd.DataFrame):
            pass
        elif path_df_start is not None:
            df_start = pd.read_hdf(str(path_df_start))

        if df_new is not None and isinstance(df_new, pd.DataFrame):
            pass
        elif path_df_new is not None:
            df_new = pd.read_hdf(path_df_new)

        df_output = pd.concat([df_start, df_new], axis=0)

        if save_result:
            if not str(path_save).endswith(".hdf5"):
                path_save_file = path_save.joinpath(file_name_save)
                path_save.mkdir(parents=True, exist_ok=True)
            else:
                path_save_file = path_save
            print(f"INFO | pre-analysis | update_df | Storing dataframe to {str(path_save_file)}...")
            df_output.to_hdf(str(path_save_file), key="all_events", complevel=9)
            print(f"INFO | pre-analysis | update_df | Stored successfully.")

        return df_output
