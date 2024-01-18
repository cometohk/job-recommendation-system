<h1>Job Recommendation System using Kaggle Data Science and Machine Learning survey Dataset</h1>

<h2>Project Information</h2>

This was a team project for my Fundamentals of Data Analytics class at the National University of Singapore. We were tasked to develop a web application (using Streamlit) using combined datasets from <i>Kaggle's Annual MAchine Learning and Data Science Survey</i> for 2022-2022. 

The web application comprises of two main milestones:
1) Exploratory Data Analysis: Use Streamlit's multipage app feature to offer 2 visualisation functionalities, i.e. trend over time and statistical tendencies, to allow users to explore the data to derive insights.  
2) Recommendation system: Add a recommendation functionality to allow app users to input their personal/professional profiles, and show recommended job roles based on the similarity of their profiles with those in the dataset.

This was the highest scoring project for the class. 

<h2>My main contributions</h2>
<ul>
<li>Ideating on how we would use the Kaggle Data Science and Machine Learning survey dataset, and data preprocessing steps</li>
<li>Formulation of problem statement, scoping of dataset, and analysis for Exploratory Data Analysis milestone</li>
<li>Feature selection, model building, model selection, model evaluation, model tuning and model validation for Job Recommendation System milestone. </li>
</ul>
Project report: IT5006 Project Report.pdf

<h2>Objective</h2>

Analyse profiles of the top 5 most common job roles and provide a profile-based recommendation of suitable job roles in the Data Science and Machine Learning sector.

<h2>Outcomes</h2>

Our recommendation engine takes in a job applicant's profile, which includes information on their education background, work experience, roles and responsibilities in previous jobs, current salary, as well as their technical skillsets in the areas of programming, data visualisation, machine learning, compter vision and natural language processing, and provides the most suitable job role (among the top 5 most common job roles) in the Data Science/Machine Learning sector that best fit their profiles.

<h2>Dataset</h2>

We used data from the Kaggle annual Data Science and Machine Learning survey, which is available on the <a href="https://www.kaggle.com/c/kaggle-survey-2022/"> Kaggle website.</a>

<h2>Approach</h2>
Our approach for building the job recommendation system is set out below.
<img src="https://github.com/cometohk/job-recommendation-system/assets/136663463/3982fdc1-450e-4ef2-a74a-8edb36a94348" alt="Approach">

Our approach for model building and selection comprised of 3 main steps:
1) Building simple classification models such as Naive Bayes, Logistic Regression and Decision Tree;
2) Building more complex ensemble models such as AdaBoost, BaggingClassifiers and Random Forest, and comparing the
results of the various models after cross-validation with the stratified k-folds (5 folds) technique; and
3) Tuning the selected model’s parameters to achieve better performance.

We evaluated each model's performance using various classification metrics, i.e. accuracy, precision, recall, f1, and area under receiver operating characteristic multi-class one-vs-all curve scores. As the models were multi-class models, we had macro-averaged the scores across the 5 classes for each fold, and then averaged the scores across the 5-folds. We selected the best performing model, which was a simple Logistic Regression model, which achieved accuracy of 0.57, or approximately almost 3 times as accurate as a random predictor. Please refer to the Jupyter Notebook "Code for model generation.ipynb" for the code base for model building.

The streamlit app utilises the pre-trained model, and allows users to input information on their profile, and generate a job-role recommendation for the user. 

Thank you for visiting.

<h3>Note:</h3> Please refer to the requirement.txt file in the streamlit folder for the required packages and versions for our streamlit app. 
