import numpy as np
import pandas as pd


class DataHelper:
    def __init__(self):
        self.raw_k2020 = pd.read_csv("Data/kaggle_survey_2020_responses.csv", header=[1])
        self.raw_k2021 = pd.read_csv("Data/kaggle_survey_2021_responses.csv", header=[1])
        self.raw_k2022 = pd.read_csv("Data/kaggle_survey_2022_responses.csv", header=[1])

        self.raw_k2020 = self.raw_k2020.iloc[:, [0, 1] + list(range(3, 20)) + list(range(53, 107)) + [107]]
        self.raw_k2021 = self.raw_k2021.iloc[:, [0, 1] + list(range(3, 20)) + list(range(59, 115)) + [127]]
        self.raw_k2022 = self.raw_k2022.iloc[:, [0, 1, 3, 24] + list(range(29, 45)) + list(range(75, 134)) + [145, 158]]

        self.diff_raw_2020 = self.diff_raw_2021 = self.diff_raw_2022 = None
        self.raw = None

    def get_raw_df(self):
        for dataset, year in zip([self.raw_k2020, self.raw_k2021, self.raw_k2022], ["2020", "2021", "2022"]):
            dataset["Year"] = year

        self._find_diff_raw()
        self._fill_na_values()

        self.raw = pd.concat([self.raw_k2020, self.raw_k2021, self.raw_k2022], axis=0, sort=False)
        return self.raw

    def _find_diff_raw(self):
        self.diff_raw_2020 = list(
            set([item for item in self.raw_k2021.columns.to_list() if item not in self.raw_k2020.columns] +
                [item for item in self.raw_k2022.columns.to_list() if item not in self.raw_k2020.columns]))

        self.diff_raw_2021 = list(
            set([item for item in self.raw_k2020.columns.to_list() if item not in self.raw_k2021.columns] +
                [item for item in self.raw_k2022.columns.to_list() if item not in self.raw_k2021.columns]))

        self.diff_raw_2022 = list(
            set([item for item in self.raw_k2020.columns.to_list() if item not in self.raw_k2022.columns] +
                [item for item in self.raw_k2021.columns.to_list() if item not in self.raw_k2022.columns]))

    def _fill_na_values(self):
        for col in self.diff_raw_2020:
            self.raw_k2020[col] = np.nan

        for col in self.diff_raw_2021:
            self.raw_k2021[col] = np.nan

        for col in self.diff_raw_2022:
            self.raw_k2022[col] = np.nan
