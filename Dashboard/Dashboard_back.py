import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import numpy as np

st.write("""
# Dashboard credit risk evaluation
""")


# load the model from disk
filename = '../notebook/finalized_model.sav'
model_reloaded = pickle.load(open(filename, 'rb'))
#read the data
app_train_domain = pd.read_csv('../input/app_train_domain.csv')

#select features used to make the prediction
feats = [f for f in app_train_domain.columns if f not in ['TARGET','SK_ID_CURR',
                                                      'SK_ID_BUREAU','SK_ID_PREV','index']]
#request the user to enter the ID on which to apply the prediction
user_input = int(st.sidebar.text_input("Enter SK_ID_CURR",100002))

if user_input not in app_train_domain['SK_ID_CURR'].tolist():
    st.sidebar.write("The id entered is unknown")
else:
    # Data to predict
    pred_features, pred_labels = app_train_domain.loc[app_train_domain['SK_ID_CURR']==user_input][feats],app_train_domain.loc[app_train_domain['SK_ID_CURR']==user_input]['TARGET']
    pred_features=pred_features.values.reshape(1, -1)
    print(pred_features)
    # prediction result
    y_pred_b = model_reloaded.predict_proba(pred_features)[:, 1]
   
    def calculate_pred(y_pred):
        if y_pred<=0.06:
            y_pred_t = 0
        else:
            y_pred_t = 1
        return y_pred_t
    
    prediction_t=calculate_pred(y_pred_b)
    print('prediction:', prediction_t)
    
    if prediction_t == 1:
        st.sidebar.write("The credit cannot be granted")
    else:
        st.sidebar.write("The credit can be granted")
    
    y_pred_g = 1 - y_pred_b
    # data
    label = [ "Credit not granted","Credit granted"]
    val = [0.94,0.06]
    
    # append data and assign color
    label.append("")
    val.append(sum(val))  # 50% blank
    colors = ['red', 'green', 'white']
    
    # plot
    fig = plt.figure(figsize=(8,6),dpi=100)
    plt.title("Probability of being a bad payer")
    ax = fig.add_subplot(1,1,1)
    ax.pie(val, labels=label, colors=colors)
    ax.add_artist(plt.Circle((0, 0), 0, color='white'))
    st.pyplot(fig)
   
 

    


fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Probability credit not repaid'],
    y=[10],
    name='credit granted',
    marker=dict(
        color='green',
        #line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
    x=['Probability credit not repaid'],
    y=[90],
    name='credit not granted',
    marker=dict(
        color='red',
        #line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )
))
fig.add_trace(go.Scatter(
            x=['Probability credit not repaid'], y=[50],
            #xaxis='x'+str(i+1), yaxis='y'+str(i+1),
            name="Customer probability credit not repaid",
            mode='markers', marker={'size': 10, 'color': 'black'},
            #text=ratings[i], hoverinfo='text', showlegend=False
    ))
fig.update_layout(barmode='stack')

st.plotly_chart(fig)