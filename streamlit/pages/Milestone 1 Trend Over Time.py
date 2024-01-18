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

st.set_page_config(page_title="Trend Over Time", page_icon="📈", layout="wide")

st.sidebar.title("Trend over time")
st.sidebar.write("Trend of median salary and educational qualifications for top 5 most surveyed job titles over the years 2020-2022")

st.header("Trend over time")
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


def filter_by_education_and_plot(dataframe, Job_title):
    
    title="Trend of education level in participant for "+ Job_title
    
    df=dataframe.copy()
    
    test_df_edu = df[df['Job Title'] == Job_title][['Education', 'Year']]
    test_df_edu['count'] = 1
    test_df_edu_2020 = test_df_edu[test_df_edu['Year']== 2020]
    test_df_edu_2020 = test_df_edu_2020.groupby(['Education', 'Year'])['count'].count().reset_index()
    test_df_edu_2020['Percentage'] = test_df_edu_2020['count']/test_df_edu_2020['count'].sum()

    test_df_edu_2021 = test_df_edu[test_df_edu['Year']== 2021]
    test_df_edu_2021 = test_df_edu_2021.groupby(['Education', 'Year'])['count'].count().reset_index()
    test_df_edu_2021['Percentage'] = test_df_edu_2021['count']/test_df_edu_2021['count'].sum()

    test_df_edu_2022 = test_df_edu[test_df_edu['Year']== 2022]
    test_df_edu_2022 = test_df_edu_2022.groupby(['Education', 'Year'])['count'].count().reset_index()
    test_df_edu_2022['Percentage'] = test_df_edu_2022['count']/test_df_edu_2022['count'].sum()
    test_df_edu_2022['Percentage*100'] = 100 * test_df_edu_2022['Percentage']
    test_df_edu = pd.concat([test_df_edu_2020, test_df_edu_2021, test_df_edu_2022])
    
    test_df_edu['Percentage*100'] = 100 * test_df_edu['Percentage']
    
    base=alt.Chart(test_df_edu).encode(
        alt.X("Year:N"),
        alt.Y('Percentage*100',scale=alt.Scale(domain=[0, 100]), title='Percentage'),
        alt.Color('Education'),
        tooltip=[
            alt.Tooltip('Year'),
            alt.Tooltip("Percentage",format='.2%')
        ],

    ).properties(
    title=title,
    height=600,
    width=300)
    
    l = base.mark_line(size=5, opacity=0.8).interactive()
    p = base.mark_point(size=200).interactive()
    
    final=l+p
    return final

kaggle_ps1a, kaggle_ps1b, kaggle_ps2, kaggle_ps3 = load_data()
kaggle_ps3_copy=kaggle_ps3.copy()

new_df=kaggle_ps3['Job Title'].value_counts().reset_index(name='counts')
new_df['Percentage']=100*new_df['counts']/(new_df['counts'].sum())

new_df['Cumulative Percentage']=new_df['Percentage'].cumsum()

new_df.index = new_df.index + 1

# Objective and scope writeup
st.write("#### Objective")
helper.add_para("The goal of this section is to extract insights about the most common job titles amongst the respondents of the Kaggle Machine Learning & Data Science Survey.")

st.write('#### Scope')
helper.add_para("We will focus on the 5 most common jobs among respondents in the field of Machine Learning and Data Science, "
                "namely Data Scientists, Data Analysts, Managers, Research Scientist and Managers. " 
                "Together, these 5 jobs account for approximately 80% of respondents in our clean dataset. More info on the job distribution among the collected data is set out below.")

#filter on job distribution
with st.expander("Job distribution among respondents"):
    st.dataframe(new_df, use_container_width=True)

    helper.add_para("Note: We have recategorised certain job titles in the data to ensure consistency across the datasets for years 2020-2022 as follows:")

    ps1_list = ["**Data Analyst** comprises of the categories \"*Data Analyst*\", \"*Data Analyst (Business, Marketing, Financial, Quantitative etc.)*\", and \"*Business Analyst*\"",
                "**Manager** comprises of the categories \"*Product/Project Manager*\", \"*Product Manager*\", \"*Program Manager*\", and \"*Manager (Program, Project, Operations, Executive-level, etc)*\"",
                "**Machine Learning/MLops Engineer** comprises of the categories \"*Machine Learning Engineer*\" and \"*Machine Learning/MLops Engineer*\""]
    helper.add_list(ps1_list, numbered=True)
    helper.add_para(" ")


top_5_Job = ['Data Analyst (Business, Marketing, Financial, Quantitative, etc)',
             'Data Scientist',
             'Manager (Program, Project, Product, Operations, Executive-level, etc)',
             'Research Scientist',
             'Software Engineer']

kaggle_ps3 = kaggle_ps3[kaggle_ps3['Job Title'].isin(top_5_Job)]
grouped_df = kaggle_ps3.groupby(['Job Title', 'Year']).median(
    numeric_only=True).add_suffix('_Count').reset_index()
grouped_df = grouped_df.rename(columns={'Salary_Count': 'Median Salary'})

#Description of filter
st.write("#### Trend 1: Median Salary of top 5 most common job types in data science and machine learning")
helper.add_para("To begin, you can select which of the top 5 job titles you are interested to explore for this section using the filter below.")

job_filter = st.multiselect(
"Choose the job title you are interested", top_5_Job, top_5_Job
)

grouped_df=grouped_df[grouped_df['Job Title'].isin(job_filter)]

st.markdown("""
<style>
.small-font {
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="small-font">Note: For optimal experience, please view the page in full screen, and mouseover data points/labels for more information. \
            Where there are overlapping points between the line charts, please select the job title of interest by itself using the filter to view the data points of interest</p>', unsafe_allow_html=True)

helper.add_para(" ")

base = alt.Chart(grouped_df).encode(
    alt.X('Year:N'),
    alt.Y('Median Salary', title='Median Salary'),
    alt.Color('Job Title'),
    tooltip=[
        alt.Tooltip("Median Salary"),
        alt.Tooltip('Job Title'),
        alt.Tooltip('Year'),

    ],
).properties(
    width=500,

    title="Median salary of top 5 most common job type across 3 years").interactive()

l = base.mark_line(size=5, opacity=0.8).interactive()
p = base.mark_point(size=200).encode(
    #     size=alt.Size('Job Title')
).interactive()

x = p+l
y = x.configure_legend(
    labelFontSize=15,
    labelLimit=500,
    symbolSize=100,
    symbolStrokeWidth=10
)

st.altair_chart(y, theme="streamlit", use_container_width=True,)

#Analysis of trend 1
with st.expander("Analysis of Trend 1"):
    helper.add_para("When it comes to salary, we have chosen to analyse the median salary as it is more robust to outliers as compared to the mean or mode statistics, "
                    "and would give a better picture of the average salary that one could expect for a particular job title.")
    
    helper.add_para("It is interesting to note that salaries across the top 5 job types generally fell slightly from 2020 to 2021 before rebounding significantly in 2022. "
                    f"This trend could be the possible result of the economic impact of the {helper.add_link('global COVID-19 pandemic', 'https://en.wikipedia.org/wiki/COVID-19_pandemic')} which was the prelude to the "
                    f"{helper.add_link('COVID-19 recession', 'https://en.wikipedia.org/wiki/COVID-19_recession')}, beginning in 2020 and ending in 2022.")
    
    helper.add_para("Across all 3 years, managers have the highest median salary, reaching a high of almost $75,000 in 2022, followed by Research Scientists (US$45k), "
                    "Data Scientists (US$45k), Software Engineers (US$22.5k) and Data Analysts (US$12.5k). ")
    
    helper.add_para("While we observe that managers have the highest average pay among the top 5 job titles, it is important to note that salaries are affected by a combination of factors, "
                    "such as company size, experience, educational qualifications and geographical locations. As such, it is clear that further exploration of additional features (e.g. age, "
                    "educational profiles, geographical locations of respondents for the various job types) is needed in order to yield clearer insight on the salary trends for the various job types, "
                    "which we hope to cover in more detail in our final report. ")


# st.subheader("Trend of education level in participant for Different Jobs")

#filter on trend 2
st.write("#### Trend 2: Educational Qualifications of top 5 most common job types in data science and machine learning")
helper.add_para("For this section, please select which of the top 5 job titles you are interested to explore below.")

option = st.selectbox(
    'Please select the job title you are interested in',
    ('Data Analyst (Business, Marketing, Financial, Quantitative, etc)',
             'Data Scientist',
             'Manager (Program, Project, Product, Operations, Executive-level, etc)',
             'Research Scientist',
             'Software Engineer'))

st.markdown('<p class="small-font">Note: For optimal experience, please view the page in full screen, and mouseover data points/labels for more information.</p>', unsafe_allow_html=True)
helper.add_para(" ")


edu_trend_chart=filter_by_education_and_plot(kaggle_ps3_copy, option)

st.altair_chart(edu_trend_chart, theme="streamlit", use_container_width=True,)

#analysis of trend 2
with st.expander("Analysis of Trend 2"):
    
    helper.add_para("More than 90% of all respondents across the top 5 job titles hold at least a bachelor’s degree. "
                    "Postgraduate degrees (master’s, doctorates) are the most common educational qualifications among data science and machine learning professionals, "
                    "which is not a surprising finding given the highly technical nature of the field. This trend is consistent among all 5 job titles, and across all 3 years of the survey. "
                    "In particular, a postgraduate qualification appears to be a necessity if you wish to pursue a career as a data scientist, manager, or a research scientist, "
                    "with more than 70% of respondents in these fields holding such qualifications. ")
    
    helper.add_para("It is clear that for now, there is still a very strong emphasis on educational qualifications in the field of data science and machine learning. "
                    "Nonetheless, there are nascent signs that this is slowly changing, as the proportion of respondents who do not hold any degrees has steadily increased "
                    "from 2020 to 2022 across all job roles. This trend is most clearly seen in data analysts, managers and software engineers, which has seen the "
                    "proportion of such respondents rise from between 3-6 % in 2020 to between 7+% in 2022. It will be interesting to observe whether this trend continues "
                    "in the coming years as generative AI tools lower the knowledge barrier for the field of data science and machine learning.")
