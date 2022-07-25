# Project Title - Heart_Attack_Analysis_Prediction_Web_App
 This project consists of simple web app build using Streamlit to analyse and predict Heart Attack.
 
 ## CVDs & Heart Attack
According to World Health Organisation (WHO), every year around 17.9 million deaths are due to cardiovascular diseases (CVDs) predisposing CVD becoming the leading cause of death globally. CVDs are a group of disorders of the heart and blood vessels, if left untreated it may cause heart attack. Heart attack occurs due to the presence of obstruction of blood flow into the heart.

 ### Prevention is better than cure
After many years of research, scientists and clinicians discovered that, the probability of one’s getting heart attack can be determined by analysing the patient’s age, gender, exercise induced angina, number of major vessels, chest pain indication, resting blood pressure, cholesterol level, fasting blood sugar, resting electrocardiographic results, and maximum heart rate achieved.

## Simple app that can help millions of life
Here is the interface of the Web App. You can access the app at anytime, anywhere with no cost at all! 
<img width="1414" alt="Heart_Attack_Prediction_Web_App" src="https://user-images.githubusercontent.com/106498393/180777488-66e6439b-ed98-4f0f-a3ba-6095db572076.png">

### The Best Model (> 80% accuracy)
#### Machine learning
From 14 variables, I chose the best 7 variables that gives more than 60% (continuous var. using Logistic Regression) & 40% (categorical var. using Cramers' Corrected Matrix).

From there, we trained using various Classifier using Pipelines Model and managed to score 87% accuracy.

<img width="364" alt="Heart_Accuracy_report" src="https://user-images.githubusercontent.com/106498393/180778438-c562d8cf-2611-4384-8222-c3a230ef8712.png">

Then, we proceed with the best model using Logistic Regression MinMaxScaler and score 80% accuracy.

<img width="645" alt="Heart_Best_model_accuracy" src="https://user-images.githubusercontent.com/106498393/180778400-a290c0d8-c2fb-4162-9212-cc1b46ed2d09.png">

Last but not least, confusion_matrix is generated.

<img width="364" alt="Heart_Confusion_Matrix" src="https://user-images.githubusercontent.com/106498393/180778373-12e75ae9-a7df-4144-bd5e-1f1b42767179.png">

 #### Data Source
The model has been analysed and trained using below dataset.
Credit to : Rashik Rahman.[Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

## Execution
You can download the data from the link provided and execute the code in this repo:

* heart_train
* heart_modules
* heart_app

***If you have an idea to improve the model, you are welcome to contribute to the code. Hit me up!***

P/S: I've tried to include all variables, but only get 78% accuracy. The code presented is by far get the best model.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
