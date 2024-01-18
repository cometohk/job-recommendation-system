import warnings
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

alt.data_transformers.enable("vegafusion")
warnings.filterwarnings('ignore')

from helper.common_helper import CommonHelper
from helper.data_helper import DataHelper

helper = CommonHelper()
data_helper = DataHelper()


st.sidebar.title("Statistical tendencies")
st.sidebar.write("Statistics on key skillsets of the top 5 most surveyed job titles over the years 2020-2022")

st.header("Statistical tendencies")
st.markdown("<hr>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    kaggle_2020 = pd.read_csv("Data/kaggle_survey_2020_responses.csv", header=[1])
    kaggle_2021 = pd.read_csv("Data/kaggle_survey_2021_responses.csv", header=[1])
    kaggle_2022 = pd.read_csv("Data/kaggle_survey_2022_responses.csv", header=[1])

    def filter_age_work_exp(age, experience, threshold):
        if "+" in age:
            age = 70  # There's only one category lol.
        elif "-" in age:
            age = int(age.split("-")[0])  # Take the youngest age category.

        if "-" in experience:
            experience = int(experience.split("-")[1].split()[0])  # Take the highest working experience.
        elif "+" in experience:
            experience = 20
        elif "<" in experience:
            experience = 1
        else:
            experience = 0

        return (age - experience) < threshold

    age_mapping = {
        '25-29': '25-29',
        '30-34': '30-34',
        '40-44': '35-44',
        '35-39': '35-44',
        '22-24': '<25',
        '45-49': '45+',
        '50-54': '45+',
        '60-69': '45+',
        '55-59': '45+',
        '70+': '45+'
    }

    # salary encoding: 0-50k:1, 50-100k:2, 100-250k:3, 250-500k:4, >500k:5
    salary_mapping = {
        '$0-999': 1,
        '100,000-124,999': 2,
        '10,000-14,999': 1,
        '40,000-49,999': 1,
        '30,000-39,999': 1,
        '50,000-59,999': 2,
        '1,000-1,999': 1,
        '5,000-7,499': 1,
        '60,000-69,999': 2,
        '70,000-79,999': 2,
        '150,000-199,999': 3,
        '20,000-24,999': 1,
        '15,000-19,999': 1,
        '125,000-149,999': 3,
        '7,500-9,999': 1,
        '2,000-2,999': 1,
        '90,000-99,999': 2,
        '80,000-89,999': 2,
        '25,000-29,999': 1,
        '3,000-3,999': 1,
        '4,000-4,999': 1,
        '200,000-249,999': 3,
        '250,000-299,999': 4,
        '300,000-499,999': 4,
        '> $500,000': 5,
        '300,000-500,000': 4,
        '$500,000-999,999': 5,
        '>$1,000,000': 5
    }

    experience_mapping = {
        '3-5 years': '1-5',
        '1-2 years': '<1',
        '5-10 years': '6-10',
        '20+ years': '20+',
        '< 1 years': '<1',
        '10-20 years': '11-20'
    }

    ml_experience_mapping = {
        'Under 1 year': '<1',
        '1-2 years': '1-5',
        '2-3 years': '1-5',
        'I do not use machine learning methods': '<1',
        '3-4 years': '1-5',
        '4-5 years': '1-5',
        '5-10 years': '6-10',
        '10-20 years': '11-20',
        '20 or more years': '20+'
    }

    education_mapping = {
        'Master’s degree': 'Postgrad',
        'Bachelor’s degree': 'Bachelor',
        'Doctoral degree': 'Postgrad',
        'Professional doctorate': 'Postgrad',
        'Professional degree': 'Postgrad',
        'Some college/university study without earning a bachelor’s degree': 'No degree',
        'No formal education past high school': 'No degree'
    }

    filter_job = ["Student", "Other", "Currently not employed", np.nan]

    kaggle_2020 = kaggle_2020[~kaggle_2020[
        "Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].isin(
        filter_job)]

    kaggle_2020_qrt1 = kaggle_2020['Duration (in seconds)'].quantile(0.25)
    kaggle_2020_qrt3 = kaggle_2020['Duration (in seconds)'].quantile(0.75)
    kaggle_2020_iqr = kaggle_2020_qrt3 - kaggle_2020_qrt1

    kaggle_2020_lb = kaggle_2020_qrt1 - (1.5 * kaggle_2020_iqr)
    kaggle_2020_ub = kaggle_2020_qrt3 + (1.5 * kaggle_2020_iqr)

    kaggle_2020 = kaggle_2020[(kaggle_2020["Duration (in seconds)"] >= kaggle_2020_lb) & (
                kaggle_2020["Duration (in seconds)"] <= kaggle_2020_ub)]

    kaggle_2020 = kaggle_2020.dropna(subset=["For how many years have you been writing code and/or programming?",
                                             "What is your age (# years)?",
                                             "Select the title most similar to your current role (or most recent title if retired): - Selected Choice",
                                             "Duration (in seconds)",
                                             "In which country do you currently reside?",
                                             "For how many years have you used machine learning methods?",
                                             "What is your current yearly compensation (approximate $USD)?"])

    kaggle_2020 = kaggle_2020[kaggle_2020["In which country do you currently reside?"] != "Other"]
    kaggle_2020 = kaggle_2020[kaggle_2020[
                                  "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"] != "I prefer not to answer"]

    kaggle_2020 = kaggle_2020[~kaggle_2020.apply(lambda x: filter_age_work_exp(x["What is your age (# years)?"],
                                                                               x[
                                                                                   "For how many years have you been writing code and/or programming?"],
                                                                               18), axis=1)]

    kaggle_2020_ps1a = kaggle_2020.iloc[:, [3] + list(range(7, 20))]
    kaggle_2020_ps1b = kaggle_2020.iloc[:, [3] + list(range(66, 82))]
    kaggle_2020_ps2 = kaggle_2020.iloc[:, [4, 5] + list(range(7, 20)) + list(range(53, 65)) + list(range(66, 107))]
    kaggle_2020_ps3 = kaggle_2020.iloc[:, [1, 4, 5, 6, 65, 118]]

    filter_job = ["Student", "Other", "Currently not employed", np.nan]

    kaggle_2021 = kaggle_2021[~kaggle_2021[
        "Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].isin(
        filter_job)]

    kaggle_2021_qrt1 = kaggle_2021['Duration (in seconds)'].quantile(0.25)
    kaggle_2021_qrt3 = kaggle_2021['Duration (in seconds)'].quantile(0.75)
    kaggle_2021_iqr = kaggle_2021_qrt3 - kaggle_2021_qrt1

    kaggle_2021_lb = kaggle_2021_qrt1 - (1.5 * kaggle_2021_iqr)
    kaggle_2021_ub = kaggle_2021_qrt3 + (1.5 * kaggle_2021_iqr)

    kaggle_2021 = kaggle_2021[(kaggle_2021["Duration (in seconds)"] >= kaggle_2021_lb) & (
                kaggle_2021["Duration (in seconds)"] <= kaggle_2021_ub)]

    kaggle_2021 = kaggle_2021.dropna(subset=["For how many years have you been writing code and/or programming?",
                                             "What is your age (# years)?",
                                             "Select the title most similar to your current role (or most recent title if retired): - Selected Choice",
                                             "Duration (in seconds)",
                                             "In which country do you currently reside?",
                                             "For how many years have you used machine learning methods?",
                                             "What is your current yearly compensation (approximate $USD)?"])

    kaggle_2021 = kaggle_2021[kaggle_2021["In which country do you currently reside?"] != "Other"]
    kaggle_2021 = kaggle_2021[kaggle_2021[
                                  "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"] != "I prefer not to answer"]

    kaggle_2021 = kaggle_2021[~kaggle_2021.apply(lambda x: filter_age_work_exp(x["What is your age (# years)?"],
                                                                               x[
                                                                                   "For how many years have you been writing code and/or programming?"],
                                                                               18), axis=1)]

    kaggle_2021_ps1a = kaggle_2021.iloc[:, [3] + list(range(7, 20))]
    kaggle_2021_ps1b = kaggle_2021.iloc[:, [3] + list(range(72, 90))]
    kaggle_2021_ps2 = kaggle_2021.iloc[:, [4, 5] + list(range(7, 20)) + list(range(59, 71)) + list(range(72, 115))]
    kaggle_2021_ps3 = kaggle_2021.iloc[:, [1, 4, 5, 6, 71, 127]]

    filter_job = ["Student", "Other", "Currently not employed", np.nan]

    kaggle_2022 = kaggle_2022[~kaggle_2022[
        "Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].isin(
        filter_job)]

    kaggle_2022_qrt1 = kaggle_2022['Duration (in seconds)'].quantile(0.25)
    kaggle_2022_qrt3 = kaggle_2022['Duration (in seconds)'].quantile(0.75)
    kaggle_2022_iqr = kaggle_2022_qrt3 - kaggle_2022_qrt1

    kaggle_2022_lb = kaggle_2022_qrt1 - (1.5 * kaggle_2022_iqr)
    kaggle_2022_ub = kaggle_2022_qrt3 + (1.5 * kaggle_2022_iqr)

    kaggle_2022 = kaggle_2022[(kaggle_2022["Duration (in seconds)"] >= kaggle_2022_lb) & (
                kaggle_2022["Duration (in seconds)"] <= kaggle_2022_ub)]

    kaggle_2022 = kaggle_2022.dropna(subset=["For how many years have you been writing code and/or programming?",
                                             "What is your age (# years)?",
                                             "Select the title most similar to your current role (or most recent title if retired): - Selected Choice",
                                             "Duration (in seconds)",
                                             "In which country do you currently reside?",
                                             "For how many years have you used machine learning methods?",
                                             "What is your current yearly compensation (approximate $USD)?"])

    kaggle_2022 = kaggle_2022[kaggle_2022["In which country do you currently reside?"] != "Other"]
    kaggle_2022 = kaggle_2022[kaggle_2022[
                                  "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"] != "I prefer not to answer"]

    kaggle_2022 = kaggle_2022[~kaggle_2022.apply(lambda x: filter_age_work_exp(x["What is your age (# years)?"],
                                                                               x[
                                                                                   "For how many years have you been writing code and/or programming?"],
                                                                               18), axis=1)]

    kaggle_2022_ps1a = kaggle_2022.iloc[:, [3] + list(range(30, 45))]
    kaggle_2022_ps1b = kaggle_2022.iloc[:, [3] + list(range(91, 106))]
    kaggle_2022_ps2 = kaggle_2022.iloc[:, [24, 145] + list(range(30, 45)) + list(range(75, 90)) + list(range(91, 134))]
    kaggle_2022_ps3 = kaggle_2022.iloc[:, [1, 24, 145, 29, 90, 158]]

    # Annotation of year to the dataset before merger.
    for dataset_ps1a, dataset_ps1b, dataset_ps2, dataset_ps3, year in zip(
            [kaggle_2020_ps1a, kaggle_2021_ps1a, kaggle_2022_ps1a],
            [kaggle_2020_ps1b, kaggle_2021_ps1b, kaggle_2022_ps1b], [kaggle_2020_ps2, kaggle_2021_ps2, kaggle_2022_ps2],
            [kaggle_2020_ps3, kaggle_2021_ps3, kaggle_2022_ps3], [2020, 2021, 2022]):
        dataset_ps1a["Year"] = year
        dataset_ps1b["Year"] = year
        dataset_ps2["Year"] = year
        dataset_ps3["Year"] = year

    # For Problem Statement 1A
    difference_ps1a_202X = [item for item in kaggle_2022_ps1a.columns.to_list() if item not in kaggle_2020_ps1a.columns]
    difference_ps1a_2022 = [item for item in kaggle_2020_ps1a.columns.to_list() if item not in kaggle_2022_ps1a.columns]

    for col in difference_ps1a_202X:
        kaggle_2020_ps1a[col] = np.nan
        kaggle_2021_ps1a[col] = np.nan

    for col in difference_ps1a_2022:
        kaggle_2022_ps1a[col] = np.nan

    # For Problem Statement 2: issue
    difference_ps2_2020 = [item for item in kaggle_2021_ps2.columns.to_list() if item not in kaggle_2020_ps2.columns]

    ## Problem Statement 1A categories renaming
    rename_ps1a_202X_coldict = {}

    for col in kaggle_2020_ps1a.columns:
        if "country" in col:
            rename_ps1a_202X_coldict[col] = "Country"
        elif "programming languages" in col:
            rename_ps1a_202X_coldict[col] = col.split(" ")[-1]

    kaggle_2020_ps1a = kaggle_2020_ps1a.rename(columns=rename_ps1a_202X_coldict)
    kaggle_2021_ps1a = kaggle_2021_ps1a.rename(columns=rename_ps1a_202X_coldict)
    kaggle_2022_ps1a = kaggle_2022_ps1a.rename(columns=rename_ps1a_202X_coldict)

    ## Problem Statement 1B categories renaming
    rename_ps1b_2020_coldict = {}

    for col in kaggle_2020_ps1b.columns:
        if "country" in col:
            rename_ps1b_2020_coldict[col] = "Country"
        elif "machine learning" in col:
            rename_ps1b_2020_coldict[col] = f"Machine Learning - {col.split('- Selected Choice -')[1].strip()}"

    rename_ps1b_2021_coldict = {}

    for col in kaggle_2021_ps1b.columns:
        if "country" in col:
            rename_ps1b_2021_coldict[col] = "Country"
        elif "machine learning" in col:
            rename_ps1b_2021_coldict[col] = f"Machine Learning - {col.split('- Selected Choice -')[1].strip()}"

    rename_ps1b_2022_coldict = {}

    for col in kaggle_2022_ps1b.columns:
        if "country" in col:
            rename_ps1b_2022_coldict[col] = "Country"
        elif "machine learning" in col:
            rename_ps1b_2022_coldict[col] = f"Machine Learning - {col.split('- Selected Choice -')[1].strip()}"

    kaggle_2020_ps1b = kaggle_2020_ps1b.rename(columns=rename_ps1b_2020_coldict)
    kaggle_2021_ps1b = kaggle_2021_ps1b.rename(columns=rename_ps1b_2021_coldict)
    kaggle_2022_ps1b = kaggle_2022_ps1b.rename(columns=rename_ps1b_2022_coldict)

    ## Problem Statement 2 categories renaming
    rename_ps2_2020_coldict = rename_ps2_2021_coldict = rename_ps2_2022_coldict = {}
    kaggle_ps2_sort_dict = {"kaggle_2020": (kaggle_2020_ps2, rename_ps2_2020_coldict),
                            "kaggle_2021": (kaggle_2021_ps2, rename_ps2_2021_coldict),
                            "kaggle_2022": (kaggle_2022_ps2, rename_ps2_2022_coldict)}

    for key, value in kaggle_ps2_sort_dict.items():
        for col in value[0].columns:
            if "formal education" in col:
                value[1][col] = "Education"
            elif "title" in col:
                value[1][col] = "Job Title"
            elif "programming languages" in col:
                value[1][col] = f"Programming Language - {col.split('- Selected Choice -')[1].strip()}"
            elif "data visualization" in col:
                value[1][col] = f"Data Visualization - {col.split('- Selected Choice -')[1].strip()}"
            elif "machine learning" in col:
                value[1][col] = f"Machine Learning - {col.split('- Selected Choice -')[1].strip()}"
            elif "ML algorithms" in col:
                value[1][col] = f"ML Algorithms - {col.split('- Selected Choice -')[1].strip()}"
            elif "computer vision" in col:
                value[1][col] = f"Computer Vision - {col.split('- Selected Choice -')[1].strip()}"
            elif "natural language processing" in col:
                value[1][col] = f"NLP - {col.split('- Selected Choice -')[1].strip()}"

    kaggle_2020_ps2 = kaggle_2020_ps2.rename(columns=rename_ps2_2020_coldict)
    kaggle_2021_ps2 = kaggle_2021_ps2.rename(columns=rename_ps2_2021_coldict)
    kaggle_2022_ps2 = kaggle_2022_ps2.rename(columns=rename_ps2_2022_coldict)

    difference_ps2_2020 = list(
        set([item for item in kaggle_2021_ps2.columns.to_list() if item not in kaggle_2020_ps2.columns] + [item for item
                                                                                                           in
                                                                                                           kaggle_2022_ps2.columns.to_list()
                                                                                                           if
                                                                                                           item not in kaggle_2020_ps2.columns]))
    difference_ps2_2021 = list(
        set([item for item in kaggle_2020_ps2.columns.to_list() if item not in kaggle_2021_ps2.columns] + [item for item
                                                                                                           in
                                                                                                           kaggle_2022_ps2.columns.to_list()
                                                                                                           if
                                                                                                           item not in kaggle_2021_ps2.columns]))
    difference_ps2_2022 = list(
        set([item for item in kaggle_2020_ps2.columns.to_list() if item not in kaggle_2022_ps2.columns] + [item for item
                                                                                                           in
                                                                                                           kaggle_2021_ps2.columns.to_list()
                                                                                                           if
                                                                                                           item not in kaggle_2022_ps2.columns]))

    ## Problem Statement 3 categories renaming
    rename_ps3_202X_coldict = {}
    kaggle_ps3_sort_dict = {"kaggle_2020": (kaggle_2020_ps3, rename_ps3_202X_coldict),
                            "kaggle_2021": (kaggle_2021_ps3, rename_ps3_202X_coldict),
                            "kaggle_2022": (kaggle_2022_ps3, rename_ps3_202X_coldict)}

    for key, value in kaggle_ps3_sort_dict.items():
        for col in value[0].columns:
            if "age" in col:
                value[1][col] = "Age Details"
            elif "formal education" in col:
                value[1][col] = "Education Details"
            elif "title" in col:
                value[1][col] = "Job Title"
            elif "writing code" in col:
                value[1][col] = "Programming Experience Details"
            elif "machine learning" in col:
                value[1][col] = "ML Programming Experience Details"
            elif "yearly compensation" in col:
                value[1][col] = "Salary Details"

    kaggle_2020_ps3 = kaggle_2020_ps3.rename(columns=rename_ps3_202X_coldict)
    kaggle_2021_ps3 = kaggle_2021_ps3.rename(columns=rename_ps3_202X_coldict)
    kaggle_2022_ps3 = kaggle_2022_ps3.rename(columns=rename_ps3_202X_coldict)

    for col in kaggle_2020_ps3.columns.to_list():
        if "Age Details" in col:
            kaggle_2020_ps3["Age"] = kaggle_2020_ps3["Age Details"].replace(age_mapping)
            kaggle_2021_ps3["Age"] = kaggle_2021_ps3["Age Details"].replace(age_mapping)
            kaggle_2022_ps3["Age"] = kaggle_2022_ps3["Age Details"].replace(age_mapping)
        elif "Education Details" in col:
            kaggle_2020_ps3["Education"] = kaggle_2020_ps3["Education Details"].replace(education_mapping)
            kaggle_2021_ps3["Education"] = kaggle_2021_ps3["Education Details"].replace(education_mapping)
            kaggle_2022_ps3["Education"] = kaggle_2022_ps3["Education Details"].replace(education_mapping)
        elif "ML Programming Experience Details" in col:
            kaggle_2020_ps3["ML Programming Experience"] = kaggle_2020_ps3["ML Programming Experience Details"].replace(
                ml_experience_mapping)
            kaggle_2021_ps3["ML Programming Experience"] = kaggle_2021_ps3["ML Programming Experience Details"].replace(
                ml_experience_mapping)
            kaggle_2022_ps3["ML Programming Experience"] = kaggle_2022_ps3["ML Programming Experience Details"].replace(
                ml_experience_mapping)
        elif "Programming Experience Details" in col:
            kaggle_2020_ps3["Programming Experience"] = kaggle_2020_ps3["Programming Experience Details"].replace(
                experience_mapping)
            kaggle_2021_ps3["Programming Experience"] = kaggle_2021_ps3["Programming Experience Details"].replace(
                experience_mapping)
            kaggle_2022_ps3["Programming Experience"] = kaggle_2022_ps3["Programming Experience Details"].replace(
                experience_mapping)
        elif "Salary Details" in col:
            kaggle_2020_ps3["Salary"] = kaggle_2020_ps3["Salary Details"].replace(salary_mapping)
            kaggle_2021_ps3["Salary"] = kaggle_2021_ps3["Salary Details"].replace(salary_mapping)
            kaggle_2022_ps3["Salary"] = kaggle_2022_ps3["Salary Details"].replace(salary_mapping)

    ##median salaries
    # median salaries for age
    med_salary_age_2020 = {}
    for age in kaggle_2020_ps3['Age'].unique():
        med_salary_age_2020[age] = kaggle_2020_ps3[kaggle_2020_ps3['Age'] == age]['Salary'].median()

    med_salary_age_2021 = {}
    for age in kaggle_2021_ps3['Age'].unique():
        med_salary_age_2021[age] = kaggle_2021_ps3[kaggle_2021_ps3['Age'] == age]['Salary'].median()

    med_salary_age_2022 = {}
    for age in kaggle_2022_ps3['Age'].unique():
        med_salary_age_2022[age] = kaggle_2022_ps3[kaggle_2022_ps3['Age'] == age]['Salary'].median()

    # median salary for prog exp
    med_salary_progexp_2020 = {}
    for progexp in kaggle_2020_ps3['Programming Experience'].unique():
        med_salary_progexp_2020[progexp] = kaggle_2020_ps3[kaggle_2020_ps3['Programming Experience'] == progexp][
            'Salary'].median()

    med_salary_progexp_2021 = {}
    for progexp in kaggle_2021_ps3['Programming Experience'].unique():
        med_salary_progexp_2021[progexp] = kaggle_2021_ps3[kaggle_2021_ps3['Programming Experience'] == progexp][
            'Salary'].median()

    med_salary_progexp_2022 = {}
    for progexp in kaggle_2022_ps3['Programming Experience'].unique():
        med_salary_progexp_2022[progexp] = kaggle_2022_ps3[kaggle_2022_ps3['Programming Experience'] == progexp][
            'Salary'].median()

    # median salary for ML exp
    med_salary_MLexp_2020 = {}
    for MLexp in kaggle_2020_ps3['ML Programming Experience'].unique():
        med_salary_MLexp_2020[MLexp] = kaggle_2020_ps3[kaggle_2020_ps3['ML Programming Experience'] == MLexp][
            'Salary'].median()

    med_salary_MLexp_2021 = {}
    for MLexp in kaggle_2021_ps3['ML Programming Experience'].unique():
        med_salary_MLexp_2021[MLexp] = kaggle_2021_ps3[kaggle_2021_ps3['ML Programming Experience'] == MLexp][
            'Salary'].median()

    med_salary_MLexp_2022 = {}
    for MLexp in kaggle_2022_ps3['ML Programming Experience'].unique():
        med_salary_MLexp_2022[MLexp] = kaggle_2022_ps3[kaggle_2022_ps3['ML Programming Experience'] == MLexp][
            'Salary'].median()

    # median salary for edu
    med_salary_edu_2020 = {}
    for edu in kaggle_2020_ps3['Education'].unique():
        med_salary_edu_2020[edu] = kaggle_2020_ps3[kaggle_2020_ps3['Education'] == edu]['Salary'].median()

    med_salary_edu_2021 = {}
    for edu in kaggle_2021_ps3['Education'].unique():
        med_salary_edu_2021[edu] = kaggle_2021_ps3[kaggle_2021_ps3['Education'] == edu]['Salary'].median()

    med_salary_edu_2022 = {}
    for edu in kaggle_2022_ps3['Education'].unique():
        med_salary_edu_2022[edu] = kaggle_2022_ps3[kaggle_2022_ps3['Education'] == edu]['Salary'].median()

    # new columns with median salaries
    for col in kaggle_2020_ps3.columns.to_list():
        if "Age" in col:
            kaggle_2020_ps3["Med_Salary_Age"] = kaggle_2020_ps3["Age"].replace(med_salary_age_2020)
            kaggle_2021_ps3["Med_Salary_Age"] = kaggle_2021_ps3["Age"].replace(med_salary_age_2021)
            kaggle_2022_ps3["Med_Salary_Age"] = kaggle_2022_ps3["Age"].replace(med_salary_age_2022)
        elif "Education" in col:
            kaggle_2020_ps3["Med_Salary_Edu"] = kaggle_2020_ps3["Education"].replace(med_salary_edu_2020)
            kaggle_2021_ps3["Med_Salary_Edu"] = kaggle_2021_ps3["Education"].replace(med_salary_edu_2021)
            kaggle_2022_ps3["Med_Salary_Edu"] = kaggle_2022_ps3["Education"].replace(med_salary_edu_2022)

        elif "ML Programming Experience" in col:
            kaggle_2020_ps3["Med_Salary_MLExp"] = kaggle_2020_ps3["ML Programming Experience"].replace(
                med_salary_MLexp_2020)
            kaggle_2021_ps3["Med_Salary_MLExp"] = kaggle_2021_ps3["ML Programming Experience"].replace(
                med_salary_MLexp_2021)
            kaggle_2022_ps3["Med_Salary_MLExp"] = kaggle_2022_ps3["ML Programming Experience"].replace(
                med_salary_MLexp_2022)

        elif "Programming Experience" in col:
            kaggle_2020_ps3["Med_Salary_ProgExp"] = kaggle_2020_ps3["Programming Experience"].replace(
                med_salary_progexp_2020)
            kaggle_2021_ps3["Med_Salary_ProgExp"] = kaggle_2021_ps3["Programming Experience"].replace(
                med_salary_progexp_2021)
            kaggle_2022_ps3["Med_Salary_ProgExp"] = kaggle_2022_ps3["Programming Experience"].replace(
                med_salary_progexp_2022)

    for col in difference_ps2_2020:
        kaggle_2020_ps2[col] = np.nan

    for col in difference_ps2_2021:
        kaggle_2021_ps2[col] = np.nan

    for col in difference_ps2_2022:
        kaggle_2022_ps2[col] = np.nan

    kaggle_ps1a = pd.concat([kaggle_2020_ps1a, kaggle_2021_ps1a, kaggle_2022_ps1a], axis=0, sort=False)
    kaggle_ps1b = pd.concat([kaggle_2020_ps1b, kaggle_2021_ps1b, kaggle_2022_ps1b], axis=0, sort=False)
    kaggle_ps2 = pd.concat([kaggle_2020_ps2, kaggle_2021_ps2, kaggle_2022_ps2], axis=0, sort=False)
    kaggle_ps3 = pd.concat([kaggle_2020_ps3, kaggle_2021_ps3, kaggle_2022_ps3], axis=0, sort=False)

    kaggle_ps3['Salary Details'] = kaggle_ps3['Salary Details'].str.replace('$', '')
    kaggle_ps3['Salary Details'] = kaggle_ps3['Salary Details'].str.replace('>', ' ')
    kaggle_ps3['Salary Details'] = kaggle_ps3['Salary Details'].str.replace(',', '')
    kaggle_ps3['Salary Details'] = kaggle_ps3['Salary Details'].str.replace(' 1000000', '1000000-1000000')
    kaggle_ps3['Salary Details'] = kaggle_ps3['Salary Details'].str.replace('  500000', '500000-500000')

    kaggle_ps3[['sal_upper_range', 'sal_lower_range']] = kaggle_ps3['Salary Details'].str.split('-', expand=True)
    kaggle_ps3['sal_upper_range'] = kaggle_ps3['sal_upper_range'].astype('int')
    kaggle_ps3['sal_lower_range'] = kaggle_ps3['sal_lower_range'].astype('int')
    kaggle_ps3['Salary'] = ((kaggle_ps3['sal_upper_range'] + kaggle_ps3['sal_lower_range']) / 2)

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Data Analyst',
                                       "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
                                       kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Business Analyst',
                                       "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
                                       kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Machine Learning Engineer',
                                       "Machine Learning/ MLops Engineer", kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Program/Project Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Product/Project Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(kaggle_ps3['Job Title'] == 'Product Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps3['Job Title'])

    kaggle_ps3['Job Title'] = np.where(
        kaggle_ps3['Job Title'] == "Manager (Program, Project, Operations, Executive-level, etc)",
        "Manager (Program, Project, Product, Operations, Executive-level, etc)", kaggle_ps3['Job Title'])
    

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Data Analyst',
                                       "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
                                       kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Business Analyst',
                                       "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
                                       kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Machine Learning Engineer',
                                       "Machine Learning/ MLops Engineer", kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Program/Project Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Product/Project Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(kaggle_ps2['Job Title'] == 'Product Manager',
                                       "Manager (Program, Project, Operations, Executive-level, etc)",
                                       kaggle_ps2['Job Title'])

    kaggle_ps2['Job Title'] = np.where(
        kaggle_ps2['Job Title'] == "Manager (Program, Project, Operations, Executive-level, etc)",
        "Manager (Program, Project, Product, Operations, Executive-level, etc)", kaggle_ps2['Job Title'])

    kaggle_ps1b.rename(
        columns={
            "Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - PyTorch Lightning": "Machine Learning - PyTorch Lightning",
            "Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Huggingface": "Machine Learning - Huggingface",

        }, inplace=True
    )
    return kaggle_ps1a, kaggle_ps1b, kaggle_ps2, kaggle_ps3


def filter_and_plot_bar(data_frame, starting_words, years):
    selected_years = " ".join([str(year) for year in years])
    title = "Popularity of " + starting_words + " for year " + selected_years
    
    df = data_frame.copy()
    respondents = {2020: df[df['Year']==2020].shape[0], 2021: df[df['Year']==2021].shape[0], 2022: df[df['Year']==2022].shape[0]}
    df_columns = df.columns.to_list()
    columns_names = [i for i in df_columns if i.startswith(starting_words)]
    columns_names.append('Year')
    columns_names.append("Job Title")
    df = df[columns_names]
    df = df.melt(id_vars=['Job Title', 'Year']).dropna()
    df = df[df['Year'].isin(years)]
    total_respondents = 0
    for year in years:
        total_respondents += respondents[year] 
    df['respondents'] = ''
    for year in [2020, 2021, 2022]:
        df.loc[df['Year'] == year, 'respondents'] = respondents[year]
    df = df['value'].value_counts().reset_index(name='counts')
    df = df.rename(columns={'value': starting_words})
    df['Percentage'] = df['counts']/total_respondents
    df['Percentage*100'] = df['Percentage']*100
    sort = alt.EncodingSortField(
        field="counts", op='count', order='descending')
    count=df.shape[0]
    colors=[]
    if count>5:
        for i in range(5):
            colors.append('purple')
        for i in range(count-5):
            colors.append('lightpink')
    else:
        colors=['pink']*20
    bar = alt.Chart(df).mark_bar(
        color='pink',
    ).encode(
        alt.X(starting_words, sort=sort, axis=alt.Axis(labelAngle=45)),
        alt.Y('Percentage*100', title='Percentage'),
        color=alt.Color(starting_words,
                     scale=alt.Scale(
                         domain=df.sort_values(['Percentage*100'],ascending=False)[starting_words].tolist(),
                         range=colors)),
        tooltip=[
            alt.Tooltip(starting_words),
            # alt.Tooltip('counts'),
            alt.Tooltip("Percentage", format='.2%')    
        ],
    ).properties(
        title=title,
        height=600)
    return bar

kaggle_ps1a, kaggle_ps1b, kaggle_ps2, kaggle_ps3 = load_data()

#Objective and scope writeup
st.write('#### Objective')
helper.add_para("For this section, the objective is to answer the following question: \"What are the required skillsets of a data science and machine learning professional?\"")

st.write('#### Scope')
helper.add_para("To answer the question above, we will explore a simple statistic: the proportion of respondents who utilise a particular tool/framework/algorithm on a regular basis for each of the top 5 job roles.")
helper.add_para("For our analysis, we will focus on the tools/frameworks/algorithms for the following skillsets that are important to the work of a data scientists and machine learning professional:")

problem_list = ["Programming Language", 
                "Data Visualisation libraries or tools", 
                "Machine Learning Frameworks", 
                "Machine Learning Algorithms", 
                "Computer Vision Methods", 
                "Natural Language Processing Methods"
                ]

helper.add_list(problem_list, numbered=True)

#Description of filter
st.write("#### Statistics on key skillsets for a data science and machine learning professional")
helper.add_para("To begin, you can select which of the top 5 job titles and the time period you are interested to explore using the filters below.")
st.markdown("""
<style>
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

option = st.selectbox(
    'Which Job are you interested',
    ('All','Data Analyst (Business, Marketing, Financial, Quantitative, etc)',
     'Data Scientist',
     'Manager (Program, Project, Product, Operations, Executive-level, etc)',
     'Research Scientist',
     'Software Engineer'))

def filter_job_before_year(data_frame, job):
    if job=='All':
        return data_frame
    return data_frame[data_frame["Job Title"]==job]

kaggle_ps2=filter_job_before_year(kaggle_ps2,option)

available_years = [2020, 2021, 2022]

# year_filter = st.multiselect(
#     "Choose year", available_years, [2020, 2021, 2022]
# )


year_filter=st.select_slider(
    'Please select a year',
    options=available_years,
    value=(2020,2022))
PL = filter_and_plot_bar(kaggle_ps2, 'Programming Language', year_filter)
CV = filter_and_plot_bar(kaggle_ps2, 'Computer Vision', year_filter)
NLP = filter_and_plot_bar(kaggle_ps2, 'NLP', year_filter)
DV = filter_and_plot_bar(kaggle_ps2, 'Data Visualization', year_filter)
MLA = filter_and_plot_bar(kaggle_ps2, 'ML Algorithms', year_filter)
ML = filter_and_plot_bar(kaggle_ps2, 'Machine Learning', year_filter)
st.markdown('<p class="small-font">Note: For optimal experience, please view the page in full screen, and mouseover data points/labels for more information. \
            Top 5 tools/frameworks/algorthms are coloured in purple in the charts below.</p>', unsafe_allow_html=True)

#including analysis within checkbox
if st.checkbox('Show Programming Language Statistic'):
    st.write('#### Programming Language')
    st.altair_chart(PL, theme="streamlit", use_container_width=True)
    with st.expander("Analysis for Programming Language skillset"):
        helper.add_para("Python is the essential programming language for all data science and machine learning professionals, "
                        "with over 85% of respondents indicating that they use it on a regular basis. In addition to being proficient in Python, "
                        "data science and machine learning professionals are likely to be familiar with SQL and R. Knowledge of other languages, such as Javascript, Java and C++, "
                        "are common among software engineers, but may not be as important for other job types, where only around 20% or "
                        "less of respondents indicated that they use these other languages on a regular basis.")
        
if st.checkbox('Show Data Visualization Statistic'):
    st.write('#### Data Visualization libraries or tools')
    st.altair_chart(DV, theme="streamlit", use_container_width=True)
    with st.expander("Analysis for Data Visualization skillset"):
        helper.add_para("Given the importance of Python in the field of machine learning and data science, it comes as no surprise that the most popular "
                        "data visualisation libraries/tools across all job types are python-supported libraries/tools. Matplotlib leads the way with at least "
                        "65% of respondents for every job type indicating regular usage, followed by Seaborn, plotly, and ggplot. Across job types, data scientists"
                        " have the highest utilisation of data visualisation tools, with more than 80% using Matplotlib, and "
                        "more than 70% using Seaborn on a regular basis. ")

if st.checkbox('Show Machine Learning Frameworks Statistic'):
    st.write('#### Machine Learning Frameworks')
    st.altair_chart(ML, theme="streamlit", use_container_width=True,)
    with st.expander("Analysis for Machine Learning frameworks skillset"):
        helper.add_para("Continuing the trend seen in programming language and data visualisation libraries/tools, Python-supported Machine Learning frameworks rank "
                        "among the most used frameworks. With more than 60% of respondents using it, Scikit-learn is the essential framework for a "
                        "data science and machine learning professional. Other frameworks highly utilised across all job types are Tensorflow, Keras, "
                        "Xgboost and PyTorch, with the highest usage seen by data scientists (~30% to 50% for these frameoworks).")
        
if st.checkbox('Show ML Algorithms Statistic'):
    st.write('#### ML Algorithms')
    st.altair_chart(MLA, theme="streamlit", use_container_width=True)
    with st.expander("Analysis for Machine Learning Algorithm skillset"):
        helper.add_para("Data science and machine learning professionals use a fairly broad range of ML algorithms in their work. Linear regression tops the list, "
                        "with more than 60% of respondents employing the algorithm, followed by decision trees/random forest, gradient boosting machines, "
                        "convolutional neural networks and bayesian approaches. Outside of these popular algorithms, there is also familiarity "
                        "with recurrent and dense neural networks, particularly among data scientists and research scientists, who have a fairly high regular usage (~25%) among respondents.")
        
if st.checkbox('Show Computer Vision Methods Statistic'):
    st.write('#### Computer Vision Methods')
    st.altair_chart(CV, theme="streamlit", use_container_width=True)
    with st.expander("Analysis for Computer Vision Methods skillset"):
        helper.add_para("Familiarity with computer vision methods does not appear to be essential for most data science and machine learning professionals. "
                        "A relatively small proportion of respondents utilise computer vision methods on a regular basis, with only around 20% of all respondents "
                        "using the most popular method of image classification and other general networks on a regular basis. Intuitively, this is not surprising "
                        "as computer vision is a fairly specific area within the field of data science and machine learning, with research scientists being the job "
                        "type that utilises such methods most regularly. For research scientsits, the most highly utilised computer vision methods are "
                        "image classification, image segmentation and general purpose image/video tools.")

if st.checkbox('Show NLP Methods Statistic'):
    st.write('#### NLP Methods')
    st.altair_chart(NLP, theme="streamlit", use_container_width=True)
    with st.expander("Analysis for NLP methods skillset"):
        helper.add_para("Similar to computer vision, NLP methods is a fairly specialised field within data science and machine learning, and NLP methods are generally not "
                        "highly utilised on a regular basis (generally ~10% or less) among respondents, with the exception of data scientists and research scientists. For these job types, word embedding/vectors, "
                        "transformer language models and encoder-decoder models are the most popular methods, with regular usage rates ranging from 10-20%.")

st.write('#### Analysis (Overall)')
helper.add_para("Now that we have analysed the various charts, we can use the insights gleaned and revisit the question set out at the start of this section: "
                "\"What are the required skillset for a data science and machine learning professional?\"")

with st.expander("Required skills for data science and machine learning professional"):
    helper.add_para("A data science and machine learning professional is generally proficient in Python and SQL, and familiar with " 
                    "common data visualisation libraries such as Matplotlib and Seaborn. They typically have experience in machine learning frameworks "
                    "such as Scikit-learn, and are able to employ linear regression models/decision trees.")
    
    helper.add_para("The common skillsets mentioned above are generally sufficient for data analysts and managers in their work, while "
                    "software engineers are likely to be proficient in other programming languages such as Javascript, Java and C++ as well.")
    
    helper.add_para("For data scientists and research scientists, on top of the above-mentioned general skills, they are usually familiar with "
                    "other data visualisation libraries such as plotly and/or ggplot, as well as other ML algorithms such as gradient boosting machines, "
                    "neural networks and bayesian approaches and ML frameworks such as Tensorflow, Keras, Xgboost and PyTorch. Data scientists and "
                    "research scientists with an interest in NLP or Computer Vision are likely to utilise the important methods in those fields "
                    "(such as word embedding/vectors, transformer language models and encoder-decoder models for NLP, and image classification, "
                    "segmentation and general purpose image/video tools for computer vision).")
