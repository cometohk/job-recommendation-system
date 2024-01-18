import streamlit as st
from st_pages import Page, show_pages, add_page_title
import warnings
warnings.filterwarnings('ignore')

from helper.common_helper import CommonHelper


helper = CommonHelper()

st.set_page_config(
    page_title="IT5006 Group 11: Homepage",
    page_icon="👋",
)

st.sidebar.title("Homepage")
st.sidebar.write("Homepage for IT5006: Group 11")


# Title of the homepage
st.header("IT5006 Group 11: Milestone 1 👋")
st.markdown("<hr>", unsafe_allow_html=True)

# Members for IT5006 team, and any disclaimers
st.write("#### Members of the team")

members = ["Cui Xiuqun – xiuqun@u.nus.edu", "Kiat Hui Khang – e1112593@u.nus.edu", "Tu Weile - tuweile@u.nus.edu"]
helper.add_list(entries=members, numbered=True)

# Prerequisites of project (python dependencies)
st.write("#### Python module dependencies pre-requisites")
helper.add_para("As the project functions contain Python dependencies, it is recommended to install the following "
                "dependencies in order to enjoy the full functionalities of the project submission. Ensure that these "
                "dependencies exist in your virtual environment before continuing.")

dependencies = ["vegafusion", "altair", "st-pages"]
helper.add_list(dependencies)

st.write("##### Installation of Python dependencies")
st.markdown("To install the dependencies in Python, you may use `pip install` as shown below within the Terminal:")
st.code("pip install vegafusion", language="bash")
st.code("pip install vegafusion-python-embed", language="bash")
st.code("pip install altair", language="bash")
st.code("pip install st-pages", language="bash")

helper.add_para("Should there be any further issues with the dependencies, please refer to the <u>requirement.txt</u> "
                "within the project files attached for this submission to install the necessary dependencies. You may "
                "also e-mail us at our contact as attached above for any clarifications.")

# Scope and Prelude
st.write("#### Preamble of Project and Problem Statements")

helper.add_para(text=f"For the first milestone of our project, our team visualizes and explores the combined datasets "
                     f"from the {helper.add_link(text='Kaggle Annual Machine Learning and Data Science Survey', link='https://www.kaggle.com/c/kaggle-survey-2020/')} "
                     f"from all three years [{helper.add_link(text='2020', link='https://www.kaggle.com/c/kaggle-survey-2020/')}, "
                     f"{helper.add_link(text='2021', link='https://www.kaggle.com/c/kaggle-survey-2021/')}, "
                     f"{helper.add_link(text='2022', link='https://www.kaggle.com/c/kaggle-survey-2022/')}] in order to "
                     f"conduct an exploratory analysis to derive insights from the data combined. Before continuing on "
                     f"with the analysis, our team performed the required data cleaning and preparation beforehand in "
                     f"order to prepare these datasets for our eventual application. You may find the data "
                     f"preprocessing in a separate page for your own reference.")

helper.add_para(text=f"For the exploratory data analysis, our team has evaluated the combined datasets and have broken "
                     f"down to the following problem statements we would like to solve:")

with st.expander("Problem Statement 1.1: Show the trend of <x> over the years 2020-2022, where <x> is selectable by users."):
    st.write("1. Show the trend of <u>median salaries</u> and <u>education level</u> for the top 5 most surveyed job "
             "types over the years 2020 - 2022.", unsafe_allow_html=True)
    helper.add_para(" ")

with st.expander("Problem Statement 1.2: Show [statistics] of <y> in <period>, where <y> and <period> are selectable by users."):
    st.write("2. Show the statistics of <u>the skillsets (programming languages, computer vision, NLP, etc.)</u> of "
             "the <u>top 5 most surveyed job types</u> in period 2020 - 2022.", unsafe_allow_html=True)
    helper.add_para(" ")

helper.add_para(text="While our team shall reserve the proper write-up of our justifications for these problem "
                     "statements in the fourth milestone, nevertheless we describe the dataset and preprocessing "
                     f"along with the exploratory analysis insights in the {helper.add_link(text='next page', link='http://localhost:8501/Exploratory%20Data%20Analysis')}.")

helper.add_para(text="If there are any clarifications or issues, please do not hesitate to let the team know via "
                     "e-mail!")

st.sidebar.success("Select a section above.")

