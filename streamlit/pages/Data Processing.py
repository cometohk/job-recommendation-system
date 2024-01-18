import pandas as pd
import streamlit as st
import altair as alt

from helper.common_helper import CommonHelper
from helper.data_helper import DataHelper
from helper.data_google_colab import load_data

helper = CommonHelper()
data_helper = DataHelper()

st.sidebar.title("Exploratory Data Analysis")
st.sidebar.write("EDA write-up for data pro-processing and problem statements.")

st.header("Dataset pre-processing and EDA")
st.markdown("<hr>", unsafe_allow_html=True)

# Description of the combined Kaggle dataset
st.write("#### Description of Kaggle dataset")
helper.add_para("Within the combined Kaggle dataset from 2020 to 2022, our team observed that the dataset comprises "
                "mainly on machine learning and data science survey questions, and that it is a subset of the overall "
                "data science community represented globally across all three surveys. There were multiple types of "
                "questions that were asked to the respondents, including some but are not limited to:")

with st.expander("Questions on demographics"):
    demo_questions = ["In which country do you currently reside?",
                      "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?",
                      "Select the title most similar to your current role (or most recent title if retired): - Selected Choice",
                      "What is your current yearly compensation (approximate $USD)?"]
    helper.add_list(demo_questions, numbered=True)
    helper.add_para(" ")

with st.expander("Questions on programming languages"):
    prog_questions = ["What programming languages do you use on a regular basis? (Select all that apply) - Python",
                      "What programming languages do you use on a regular basis? (Select all that apply) - Java",
                      "What programming languages do you use on a regular basis? (Select all that apply) - C++"]
    helper.add_list(prog_questions, numbered=True)
    helper.add_para(" ")

with st.expander("Questions on machine learning"):
    ml_questions = [
        "Which of the following ML algorithms do you use on a regular basis? (Select all that apply): -Linear or Logistic Regression",
        "Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Decision Trees or Random Forests",
        "Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Bayesian Approaches"]
    helper.add_list(ml_questions, numbered=True)
    helper.add_para(" ")

helper.add_para("Our dataset also comprises of a diverse group of respondents globally, with a wide variety of "
                "responses towards job titles, education levels, and the duration taken within the survey. You may "
                "find this in the tables down below.")

df = data_helper.get_raw_df()

with st.expander("Country: Kaggle datasets from 2020 to 2022"):
    st.write("###### Top 5 counts for: 'In which country do you currently reside in?'")
    st.dataframe(df["In which country do you currently reside?"].value_counts().head(5), use_container_width=True)

with st.expander("Education: Kaggle dataset from 2020 to 2022"):
    st.write("###### Top 5 counts for: 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?")
    st.dataframe(df['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().head(5), use_container_width=True)

with st.expander("Duration: Time taken to complete the survey for Kaggle dataset from 2020 to 2022 (takes time to load)"):
    duration_df = df[df["Duration (in seconds)"] < 6000]

    st.write("###### Distribution curve for the time taken to complete the survey for Kaggle dataset from 2020 to 2022")

    hist_chart = alt.Chart(duration_df).mark_bar().encode(
        alt.X('Duration (in seconds):Q', bin=alt.Bin(maxbins=200), title="Time taken (seconds)"),
        alt.Y("count():Q", title="Frequency")
    ).properties(
        width="container",
        height=300
    )

    st.altair_chart(hist_chart, use_container_width=True)

    st.write("###### General statistics for the time taken to complete the survey for Kaggle dataset from 2020 to 2022")
    st.dataframe(df["Duration (in seconds)"].describe(), use_container_width=True)

helper.add_para("Upon inspection of our dataset, our team observed multiple issues with the combined dataset, mainly:")

problem_list = ["Some of the respondents did not answer the Country question, or simply identify themselves with a "
                "country called \"Others\", which is extremely difficult to work with;",
                "Some of the respondents likewise did not answer for the Education question, and this presents "
                "difficulties with the advent of missing or null values within our dataset;",
                "The time taken for the respondents to answer around 35 to 45 questions is staggering: some "
                "respondents took around 20 seconds, while others required 11 days to answer this survey; and",
                "Some of the data headers needs to be renamed for simplicity, and some of the data are not compliant "
                f"to certain standards when plotting graphs (e.g. countries not compliant with the "
                f"{helper.add_link('ISO 3166-1 standard', 'https://en.wikipedia.org/wiki/ISO_3166-1')}."]

helper.add_list(problem_list, numbered=True)

helper.add_para("With these issues at hand, there is more than enough justification for our team to start data "
                "pre-processing in order to clean the data so that our team may conduct its own data analysis and "
                "discover new insights.")

# Data pre-processing
st.write("#### Data preprocessing of Kaggle dataset")

helper.add_para("Our team first starts with the data pre-processing by removing data points that does not fit the "
                "criteria in solving the two problem statements stated earlier in the Homepage, mainly:")

conditions_list = ["Filtering out the jobs that are not relevant ('Student', 'Other', 'Currently not employed');",
                   "Dropping out those who answered the survey too quickly or slowly with the "
                   f"{helper.add_link('IQR method', 'https://online.stat.psu.edu/stat200/lesson/3/3.2')} as outliers;",
                   "Dropping out those with null answers in questions such as age, country, years of programming "
                   "experience, years of ML experience, job title, education, and yearly compensation; and",
                   "Filtering out responses who 'prefer not to answer' to questions related to age and education."]

helper.add_list(conditions_list,numbered=True)

helper.add_para("Our team have also took steps to rename the columns in order for us to traverse the combined dataset "
                "easily and to understand the granularity of the dataset we are dealing with, so that we can better "
                "plot our graphs in a more easier and readable format. Examples include:")

questions_df = pd.DataFrame({
    "Old Questions": ["In which country do you currently reside?",
                      "What is your current yearly compensation (approximate $USD)?",
                      "What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"],
    "New Questions": ["Country", "Salary Details", "Education"]
})

st.markdown(questions_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

helper.add_para(" ")  # Just to add a paragraph as a line break, since I am lazy to code out a proper break LOL
helper.add_para("Once our team has completed the data pre-processing, our dataset shrank down to 17 547 entries which "
                "is still sufficient for us to continue on with our evaluation. We move on to the next phase: "
                "extracting out the features necessary for us to conduct our exploratory data analysis.")

# Exploratory data analysis
st.write("#### Exploratory data analysis of the Kaggle dataset")

st.write("##### Problem Statement 1")

helper.add_para("For the first problem statement to show the trend of median salaries and education level for the top "
                "5 most surveyed job types over the years 2020 - 2022, we extract out the main features and showcase "
                "the raw dataset used:")

ps1a, ps1b, ps2, ps3 = load_data()

ps3_filter = ps3.columns[:11].to_list()

with st.expander("Raw Kaggle dataset for PS1: Trends over time"):
    st.dataframe(ps3[ps3_filter].head(5), use_container_width=True)

    helper.add_para("Some of the features our team have extracted out for the first problem statement includes:")

    ps1_list = ["Age details", "Education details", "Programming experience", "Job title", "Salary Details"]

    helper.add_list(ps1_list, numbered=True)
    helper.add_para(" ")

helper.add_para("However, our team noticed that for specific categories such as Salary Details, our dataset comprises "
                "mainly of categorical values and not numerical values. This represents a challenge to us as it "
                "will be difficult to transform this type of data appropriately when it comes to plotting. As a "
                "result, we have decided to use encoding techniques as well as taking the average between the "
                "upper range and lower range in order to give our team something quantifiable from the dataset.")

with st.expander("Raw Kaggle dataset with new features for PS1: Trends over time"):
    st.dataframe(ps3.head(5), use_container_width=True)

helper.add_para("With this prepared dataset for the first problem statement, we move forward with the analysis on the "
                f"trends over time in the {helper.add_link('next page', 'http://localhost:8501/PS1:%20Trends%20over%20time')}.")

st.write("##### Problem Statement 2")

helper.add_para("As for the second problem statement to show the statistics of the skillsets (programming languages, "
                "computer vision, NLP, etc.) of the top 5 most surveyed job types in the period 2020 - 2022, we "
                "extract out the main features and showcase the raw dataset used:")

with st.expander("Raw Kaggle dataset for PS2: Statistics"):
    st.dataframe(ps2.head(5), use_container_width=True)

    helper.add_para("Some of the features our team have extracted out for the second problem statement includes:")

    ps2_list = ["Education details", "Job title", "Programming languages known", "Data visualization skills known",
                "ML Algorithms known"]

    helper.add_list(ps2_list, numbered=True)
    helper.add_para(" ")

helper.add_para("With this prepared dataset for the second problem statement, we move forward with the analysis on the "
                f"trends over time in the {helper.add_link('next page', 'http://localhost:8501/PS2:%20Statistics')}.")


st.sidebar.success("Select a section above.")