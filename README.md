# OpenClassroom
# Project : Implement a credit risk scoring model
### Project7 of the datascientist formation of OpenClassroom
The objective of this project is to develop a scoring model for the company “Prêt à depenser” in order to predict the probability that a customer will repay his loan or not.
This probability will be used to decide whether to grant the loan to the customer or not. 
To explain the decision to the customer, “Prêt à depenser” wants to present him an interactive dashboard.

### DATA used
Data about historical loan
https://www.kaggle.com/c/home-credit-default-risk/data

### Code
P7_1_Data_Preparation.ipynb :data exploration and preparation based on the existing kernel Kaggle: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. Fit the nearest neighbors algorithm to the training data and the save the output in neigh_model.sav to be used in the dashboard.<br/>
P7_2_Modelisation.ipynb  : Train and test a lightGBM model. Save the output model in file finalized_model.sav and the list of inportant features in features_importance.csv to be used in the dashboard<br/>
P7_3_Dashboard.py : The Dashboard implemented using the streamlit library<br/>
RadarPlot.py: code to plot a radar plot allowing to compare data of a customer with others<br/>

### Requirements
numpy==1.20.3<br/>
seaborn==0.11.1<br/>
streamlit==0.84.0<br/>
matplotlib==3.3.4<br/>
plotly==5.1.0<br/>
pandas==1.3.0<br/>
lightgbm==3.2.1<br/>
pandas==1.3.0<br/>
sklearn==0.0<br/>

## Dashboard Scoring Credit on Heroku
https://creditdashboard.herokuapp.com/ <br/>
Files to deploy the Dashboard on heroku: <br/>
Procfile <br/>
requirements.txt <br/>
Setup.sh
