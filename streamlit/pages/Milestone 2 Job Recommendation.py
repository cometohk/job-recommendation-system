
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
from sklearn.linear_model import LogisticRegression
from PIL import Image

feature_header=['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
       'For how many years have you been writing code and/or programming?',
       'Approximately how many times have you used a TPU (tensor processing unit)?',
       'For how many years have you used machine learning methods?',
       'Does your current employer incorporate machine learning methods into their business?',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB',
       'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js ',
       'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh ',
       'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ',
       'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow ',
       'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras ',
       'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch ',
       'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost ',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks',
       'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks',
       'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)',
       'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)',
       'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Object detection methods (YOLOv3, RetinaNet, etc)',
       'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
       'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Generative Networks (GAN, VAE, etc)',
       'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)',
       'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)',
       'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Contextualized embeddings (ELMo, CoVe)',
       'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-3, BERT, XLnet, etc)',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models',
       'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning']




st.sidebar.title("Job Recommendation")
st.sidebar.write("Job Recommendation based on the criteria you have selected")

st.header("Job Recommendation")

with st.form("my_form"):
   #Q1
   st.markdown("***Q1 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?***")
   qualification = st.selectbox('Pick a Qualification', ['No formal education past high school','Some college/university study without earning a bachelor’s degree', 'Bachelor’s degree','Master’s degree', 'Doctoral degree',
       'Professional degree', 
       'Professional doctorate'])
   
   #Q2
   st.markdown("***Q2 For how many years have you been writing code and/or programming?***")
   coding_year = st.selectbox('Pick a Choice', ['< 1 years','1-2 years','1-3 years','3-5 years','5-10 years','10-20 years','20+ years'])

    #Q3
   st.markdown("***Q3 Approximately how many times have you used a TPU (tensor processing unit)?***")
   tpu = st.selectbox('Pick a Choice', ['Never','Once','2-5 times','6-25 times','More than 25 times'])
   
    #Q4
   st.markdown("***Q4 For how many years have you used machine learning methods?***")
   ml_year = st.selectbox('Pick a Choice', ['Under 1 year','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-20 years','20 or more years','I do not use machine learning methods',])


    #Q5
   st.markdown("***Q5 Does your current employer incorporate machine learning methods into their business?***")
   ml_status = st.selectbox('Pick a Choice', ['No (we do not use ML methods)',
                                            
                                            'We are exploring ML methods (and may one day put a model into production)',

                                            'We use ML methods for generating insights (but do not put working models into production)',
                                            
                                            'We recently started using ML methods (i.e., models in production for less than 2 years)',

                                            'We have well established ML methods (i.e., models in production for more than 2 years)',
                                            
                                            'I do not know',
                                            ])
   #Q6
   st.markdown("***Q6 What programming languages do you use on a regular basis? (Select all that apply)***")
   Q6_Python = st.checkbox('Python')
   Q6_R = st.checkbox('R')
   Q6_SQL = st.checkbox('SQL')
   Q6_C = st.checkbox('C')
   Q6_Cpp = st.checkbox('C++')
   Q6_Java = st.checkbox('Java')
   Q6_Javascript = st.checkbox('Javascript')
   Q6_Bash = st.checkbox('Bash')
   Q6_MATLAB = st.checkbox('MATLAB')
   Q6_Other = st.checkbox('Other')

    #Q7
   st.markdown("***Q7 What data visualization libraries or tools do you use on a regular basis?  (Select all that apply)***")
   Q7_Matplotlib = st.checkbox('Matplotlib')
   Q7_Seaborn = st.checkbox('Seaborn')
   Q7_PlotlyL = st.checkbox('Plotly / Plotly Express')
   Q7_Ggplot = st.checkbox('Ggplot / ggplot2')
   Q7_Shiny = st.checkbox('Shiny')
   Q7_D3 = st.checkbox('D3 js')
   Q7_Bokeh = st.checkbox('Bokeh')


    #Q8
   st.markdown("***Q8 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply)***")
   Q8_Scikit = st.checkbox('Scikit-learn')
   Q8_TensorFlow = st.checkbox('TensorFlow')
   Q8_Keras = st.checkbox('Keras')
   Q8_PyTorch = st.checkbox('PyTorch')
   Q8_Xgboost = st.checkbox('Xgboost')


    #Q9
   st.markdown("***Q9 Which of the following ML algorithms do you use on a regular basis? (Select all that apply)***")
   Q9_LLR = st.checkbox('Linear or Logistic Regression')
   Q9_DT = st.checkbox('Decision Trees or Random Forests')
   Q9_GBM = st.checkbox('Gradient Boosting Machines (xgboost, lightgbm, etc)')
   Q9_BA = st.checkbox('Bayesian Approaches')
   Q9_DNN = st.checkbox("""Dense Neural Networks (MLPs, etc)""")
   Q9_CNN = st.checkbox('Convolutional Neural Networks')
   Q9_RNN = st.checkbox('Recurrent Neural Networks')

    #Q10
   st.markdown("***Q10 Which categories of computer vision methods do you use on a regular basis?  (Select all that apply)***")
   Q10_GP = st.checkbox("""General purpose image/video tools (PIL, cv2, skimage, etc)""")
   Q10_IS = st.checkbox("""Image segmentation methods (U-Net, Mask R-CNN, etc)""")
   Q10_OD = st.checkbox("""Object detection methods (YOLOv3, RetinaNet, etc)""")
   Q10_IC = st.checkbox("""Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)""")
   Q10_GN = st.checkbox("""Generative Networks (GAN, VAE, etc)""")


    #Q11
   st.markdown("***Q11 Which of the following natural language processing (NLP) methods do you use on a regular basis? (Select all that apply)***")
   Q11_WE = st.checkbox("""Word embeddings/vectors (GLoVe, fastText, word2vec)""")
   Q11_ED = st.checkbox("""Encoder-decorder models (seq2seq, vanilla transformers)""")
   Q11_CE = st.checkbox("""Contextualized embeddings (ELMo, CoVe)""")
   Q11_TL = st.checkbox("""Transformer language models (GPT-3, BERT, XLnet, etc)""")


    #Q12
   st.markdown("***Q12 Select any activities that make up an important part of your role at work: (Select all that apply)***")
   Q12_AU = st.checkbox("""Analyze and understand data to influence product or business decisions""")
   Q12_BRD = st.checkbox("""Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data""")
   Q12_BP = st.checkbox("""Build prototypes to explore applying machine learning to new areas""")
   Q12_BRML = st.checkbox("""Build and/or run a machine learning service that operationally improves my product or workflows""")
   Q12_EI = st.checkbox("""Experimentation and iteration to improve existing ML models""")
   Q12_RA = st.checkbox("""Do research that advances the state of the art of machine learning""")






   submit=st.form_submit_button('Submit my picks')




feature1=qualification
feature2=coding_year 
feature3=tpu
feature4=ml_year
feature5=ml_status

feature6='Python' if Q6_Python else "NA"
feature7='R' if Q6_R  else "NA"
feature8='SQL' if Q6_SQL else "NA"
feature9= 'C' if Q6_C else "NA"
feature10='C++' if Q6_Cpp else "NA"
feature11='Java' if Q6_Java else "NA"
feature12='Javascript' if Q6_Javascript else "NA"
feature13='Bash' if Q6_Bash else "NA"
feature14='MATLAB' if Q6_MATLAB else "NA"
feature15= 'Other' if Q6_Other else "NA"

feature16='Matplotlib' if Q7_Matplotlib else "NA"
feature17='Seaborn' if Q7_Seaborn else "NA"
feature18='Plotly / Plotly Express' if Q7_PlotlyL else "NA"
feature19='Ggplot / ggplot2' if Q7_Ggplot else "NA"
feature20='Shiny' if Q7_Shiny else "NA"
feature21='D3 js' if Q7_D3 else "NA"
feature22='Bokeh' if Q7_Bokeh else "NA"

feature23='Scikit-learn' if Q8_Scikit else "NA"
feature24='TensorFlow' if Q8_TensorFlow else "NA"
feature25='Keras' if Q8_Keras else "NA"
feature26='PyTorch' if Q8_PyTorch else "NA"
feature27='Xgboost' if Q8_Xgboost else "NA"

feature28='Linear or Logistic Regression' if Q9_LLR else "NA"
feature29='Decision Trees or Random Forests' if Q9_DT else "NA"
feature30='Gradient Boosting Machines (xgboost, lightgbm, etc)' if Q9_GBM else "NA"
feature31='Bayesian Approaches' if Q9_BA else "NA"
feature32='Dense Neural Networks (MLPs, etc)' if Q9_DNN else "NA"
feature33='Convolutional Neural Networks' if Q9_CNN else "NA"
feature34='Recurrent Neural Networks' if Q9_RNN else "NA"

feature35='General purpose image/video tools (PIL, cv2, skimage, etc)' if Q10_GP else "NA"
feature36='Image segmentation methods (U-Net, Mask R-CNN, etc)' if Q10_IS else "NA"
feature37='Object detection methods (YOLOv3, RetinaNet, etc)' if Q10_OD else "NA"
feature38='Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)' if Q10_IC else "NA"
feature39= 'Generative Networks (GAN, VAE, etc)' if Q10_GN else "NA"

feature40='Word embeddings/vectors (GLoVe, fastText, word2vec)' if Q11_WE else "NA"
feature41='Encoder-decorder models (seq2seq, vanilla transformers)' if Q11_ED else "NA"
feature42='Contextualized embeddings (ELMo, CoVe)' if Q11_CE else "NA"
feature43='Transformer language models (GPT-3, BERT, XLnet, etc)' if Q11_TL else "NA"

feature44='Analyze and understand data to influence product or business decisions' if Q12_AU  else "NA"
feature45='Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data' if Q12_BRD  else "NA"
feature46='Build prototypes to explore applying machine learning to new areas' if Q12_BP  else "NA"
feature47='Build and/or run a machine learning service that operationally improves my product or workflows' if Q12_BRML  else "NA"
feature48='Experimentation and iteration to improve existing ML models' if Q12_EI  else "NA"
feature49='Do research that advances the state of the art of machine learning' if Q12_RA else "NA"

test_data=[feature1,   feature2, feature3,  feature4,  feature5,  feature6,  feature7,  feature8,  feature9,  feature10,    
        feature11,  feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, 
        feature21,  feature22, feature23, feature24, feature25, feature26, feature27, feature28, feature29, feature30,
        feature31,  feature32, feature33, feature34, feature35, feature36, feature37, feature38, feature39, feature40, 
        feature41,   feature42, feature43, feature44, feature45, feature46, feature47,feature48, feature49, ]

# st.write(test_data)

test_data = pd.DataFrame(test_data).T
test_data.columns=feature_header

def fit_transform_data(data, encoder=None):
    cols_to_encode = ['For how many years have you been writing code and/or programming?',
            'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
            'Approximately how many times have you used a TPU (tensor processing unit)?',
            'For how many years have you used machine learning methods?',
        'Does your current employer incorporate machine learning methods into their business?']

    data.reset_index(drop=True, inplace=True)
    single_select_data = data[cols_to_encode]
    multiselect_data=data.drop(cols_to_encode, axis=1)

    for column in multiselect_data.columns:
        multiselect_data[column] = np.where(multiselect_data[column] =="NA", 0, 1)

    if encoder==None:
        encoder=OneHotEncoder(sparse=False)
        encoded_data = encoder.fit_transform(single_select_data)
    else:
        encoded_data = encoder.transform(single_select_data)
        
    df_encoded = pd.DataFrame(encoded_data, columns = encoder.get_feature_names_out(cols_to_encode))
    data_encoded = pd.concat([df_encoded,multiselect_data], axis =1)
    data_encoded = data_encoded.astype(int)
    return data_encoded, encoder



Oh_encoder=load('Oh_encoder.joblib')
logreg_model=load('logreg_model.joblib')
encoded_test_data,encoder=fit_transform_data(test_data,encoder=Oh_encoder)
result=logreg_model.predict(encoded_test_data)[0]



if submit:
    st.snow()
    st.write(f"Your Recommended job is : :blue[{result}]")
    
    if result=="Data Scientist":
        job = Image.open('Data_Scientist.png')
        st.image(job)
        
    if result=="Data Analyst (Business, Marketing, Financial, Quantitative, etc)":
        job = Image.open('data_analyst.jpg')
        st.image(job)
    
    if result=="Software Engineer":
        job = Image.open('software_engineer.jpg')
        st.image(job)
        
    if result=="Research Scientist":
        job = Image.open('research_scientist.jpg')
        st.image(job)
        
    if result=="Machine Learning/ MLops Engineer":
        job = Image.open('ML.jpg')
        st.image(job)
        