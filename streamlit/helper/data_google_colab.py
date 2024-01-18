import numpy as np
import pandas as pd


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

    kaggle_ps1b.rename(
        columns={
            "Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - PyTorch Lightning": "Machine Learning - PyTorch Lightning",
            "Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Huggingface": "Machine Learning - Huggingface",

        }, inplace=True
    )
    return kaggle_ps1a, kaggle_ps1b, kaggle_ps2, kaggle_ps3