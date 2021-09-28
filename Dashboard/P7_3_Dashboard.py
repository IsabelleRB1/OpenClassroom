import pandas as pd
import streamlit as st

import pickle

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.impute import SimpleImputer 
#import sys
#sys.path.append('..\\utils')
#import SessionState
#import RadarPlot as rp


def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    res = [(d-y1) / (y2-y1) * (x2-x1) + x1
           for d, (y1, y2) in zip(data, ranges)]
    return res


class RadarChart():
    def __init__(self, fig, variables, ranges,
                     n_ordinate_levels=6):
            angles = np.arange(0, 360, (360. / len(variables)))

            axes = [fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True,
                                 label="axes{}".format(i))
                    for i in range(len(variables))]

            axes[0].set_thetagrids(angles, labels=[])

            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)

            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i],
                                   num=n_ordinate_levels)
                gridlabel = ["{}".format(round(x, 2))
                             for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1]  # hack to invert grid
                    # gridlabels aren't reversed
                gridlabel[0] = ""  # clean up origin
                ax.set_rgrids(grid, labels=gridlabel, angle=angles[i], fontsize=6)
                # ax.spines["polar"].set_visible(False)
                ax.set_ylim(*ranges[i])

            ticks = angles
            ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
            ticklabels = variables
            ax.set_xticklabels(ticklabels, fontsize=1)  # définit les labels

            angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
            angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
            angles1 = np.rad2deg(angles1)
            labels = []
            for label, angle in zip(ax.get_xticklabels(), angles1):
                x, y = label.get_position()
                lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va())
                lab.set_rotation(angle)
                lab.set_fontsize(9)
                lab.set_fontweight('bold')
                labels.append(lab)
            ax.set_xticklabels([])

            # variables for plotting
            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]


    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        l = self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
        return l

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, categories, *args, **kw):
        self.ax.legend(*args, **kw)

    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform=self.ax.transAxes, *args, **kw)
        
def radar_plot(cust_features, desc_features_train, categories, xsize=0.1, ysize=0.05):
    features_train_1 = []
    features_train_0 = []
    
    # List of descriptive variables
    var_descriptives = [col for col in categories]
    
    
    features_train_1 = desc_features_train[desc_features_train['Difficulties payment']=='yes'][categories].mean()
    features_train_0 = desc_features_train[desc_features_train['Difficulties payment']=='no'][categories].mean()
    cust_features_sel = cust_features[categories] 
    
    # Min,max descriptive varaibles
    var_descript_ranges = [list(desc_features_train[categories].describe().loc[['min', 'max'], var])
                           for var in var_descriptives]
   
 

   
    fig = plt.figure(figsize = (6, 6))
    lax = []
    data_plot = np.array(cust_features_sel)
    radar = RadarChart(fig, var_descriptives,
                           var_descript_ranges)
    l, = radar.plot(data_plot, color='b', linewidth=1.0)
    lax.append(l)
    radar.fill(data_plot, alpha=0.2, color='b')
    data_plot = np.array(features_train_0)
    l, = radar.plot(data_plot, color='g', linewidth=1.0)
    lax.append(l)
    data_plot = np.array(features_train_1)
    l, = radar.plot(data_plot, color='r', linewidth=1.0)
    lax.append(l)
    
    # Titre du radarchart

    #radar.title(title='test',
    #           color='r',
    #           size=14)
  
    radar.ax.legend(handles = lax, labels=['Customer','Credit granted','Credit not granted'], loc=3, bbox_to_anchor=(0,0,1,1), bbox_transform=fig.transFigure,prop={'size': 6} )
    # radar.legend()
    st.pyplot(fig)


st.set_page_config(layout="wide")

st.sidebar.title("""
    Credit risk evaluation
    """)
st.write('')
st.write('')
st.write('')
st.write('')


# load the model from disk
filename = 'notebook/finalized_model.sav'
model_reloaded = pickle.load(open(filename, 'rb'))
filename_neigh='notebook/neigh_model.sav'
neigh_model= pickle.load(open(filename_neigh, 'rb'))
#read the train and test data
app_train_domain = pd.read_csv('input/app_train_domain_trunc.csv')

app_train_domain.set_index('SK_ID_CURR',inplace=True)
app_test_domain = pd.read_csv('input/app_test_domain.csv')
app_test_domain.set_index('SK_ID_CURR',inplace=True)

app_train_domain.drop(app_train_domain[app_train_domain['ANNUITY_INCOME_PERCENT']>1].index, inplace=True)

#select features used to make the prediction
feats = [f for f in app_train_domain.columns if f not in ['TARGET','SK_ID_CURR',
                                                      'SK_ID_BUREAU','SK_ID_PREV','index']]
#request the user to enter the ID on which to apply the prediction
#session_state = SessionState.get(user_input=100002) 
loan_id = int(st.sidebar.text_input("Enter the loan request id of the customer",100002))
user_input=loan_id
list_id = app_train_domain.index.to_list()
list_id = list_id + app_test_domain.index.to_list()

if user_input not in list_id:
    st.sidebar.write("The id entered is unknown")
else:
    pred_features = []
    pred_labels = []
    desc_features = []
    #The loan id belongs to the training data
    if user_input in app_train_domain.index.to_list():
        # Data to predict
        pred_features, pred_labels = app_train_domain.loc[user_input][feats],app_train_domain.loc[user_input]['TARGET']
        pred_features=pred_features.values.reshape(1, -1)
    #The loan id belongs to the test data  
    else:
        if user_input in app_test_domain.index.tolist():
            # Data to predict
            pred_features = app_test_domain.loc[user_input][feats]
            pred_features = pred_features.values.reshape(1, -1)
            
  
    # prediction result
    y_pred_b = model_reloaded.predict_proba(pred_features)[:, 1]
   
    
    st.sidebar.write("The probability, the customer will not repay the loan is:",'%.2f' % float(y_pred_b*100),'%')
    def calculate_pred(y_pred):
        if y_pred<=0.08:
            y_pred_t = 0
        else:
            y_pred_t = 1
        return y_pred_t
    
    #Classify the loan appplication as "Payment difficulties" or "No payment difficulties"
    prediction_t=calculate_pred(y_pred_b)

    
    if prediction_t == 1:
        st.sidebar.write("The credit cannot be granted")
    else:
        st.sidebar.write("The credit can be granted")
    
    #Draw a bar chart to represent the probability for the customer to not repay his credit
    layout = go.Layout(autosize=True,width=300,height=500)
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(
        x=['Probability credit not repaid'],
        y=[8],
        name='credit granted',
        marker=dict(
            color='green',
            #line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        x=['Probability credit not repaid'],
        y=[92],
        name='credit not granted',
        marker=dict(
            color='red',
            #line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))
    prob = int(y_pred_b*100)
    print(prob)
    fig.add_trace(go.Scatter(
                x=['Probability credit not repaid'], y=[prob],
                #xaxis='x'+str(i+1), yaxis='y'+str(i+1),
                name="Customer probability credit not repaid",
                mode='markers', marker={'size': 10, 'color': 'black'},
                #text=ratings[i], hoverinfo='text', showlegend=False
        ))
    fig.update_layout(barmode='stack')
    
    st.sidebar.plotly_chart(fig)
    
    #Compare the value of main features of the customers with its closes neigbors
    #select the K=5 nearest neighbours of the loan  
   # train_df_knn = app_train_domain[app_train_domain['TARGET'].notnull()]
    #test_df_knn = app_test_domain
    #train_df_knn_feats = train_df_knn[feats]
    #test_df_knn_feats = test_df_knn[feats]

    #input missing values
  
    imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
    #imputer=imputer.fit(train_df_knn_feats)
    #train_df_knn_feats=imputer.transform(train_df_knn_feats)
    pred_features_knn = pred_features
    imputer=imputer.fit(pred_features_knn)
    pred_features_knn = imputer.transform(pred_features_knn)
   
    #find the neighbors of the training set
    #neigh = NearestNeighbors(n_neighbors=5)
    #neigh.fit(train_df_knn_feats)

    index_neighbors = neigh_model.kneighbors(pred_features_knn.reshape(1, -1),return_distance=False)
    index_neighbors_id = app_train_domain.iloc[index_neighbors[0]].index
    
    
    
    #Select main features to compare 
    cat_list = ['TARGET','CREDIT_TERM','DAYS_BIRTH', 
                      'DAYS_EMPLOYED', 'ANNUITY_INCOME_PERCENT','AMT_ANNUITY',
                      'CREDIT_INCOME_PERCENT','AMT_CREDIT','EXT_SOURCE_3','EXT_SOURCE_2']
    desc_features_train = app_train_domain[cat_list]
    desc_features_test = app_test_domain[cat_list[1:len(cat_list)]]
    col_dict = {'TARGET':'Difficulties payment','CREDIT_TERM':'Credit duration in year','DAYS_BIRTH':'Age', 
                      'DAYS_EMPLOYED':'Employment duration in year', 'ANNUITY_INCOME_PERCENT':'Annuity over income in %',
                      'AMT_ANNUITY':'Annuity amount in $',
                      'CREDIT_INCOME_PERCENT':'Credit amount over income in %',
                      'AMT_CREDIT':'Credit amount in $','EXT_SOURCE_3':'External source 3',
                      'EXT_SOURCE_2':'External source 2'}
    desc_features_train = desc_features_train.rename(columns=col_dict)
    desc_features_test = desc_features_test.rename(columns=col_dict)
   
   
    #convert features
    desc_features_train['Annuity over income in %'] =  (100*desc_features_train['Annuity over income in %'])
    desc_features_train['Age'] =  round(desc_features_train['Age']/365,1)
    desc_features_train['Employment duration in year'] = round(-desc_features_train['Employment duration in year']/365,1)
    desc_features_train['Difficulties payment'] = desc_features_train['Difficulties payment'].map({1:'yes', 0:'no'})
    
    desc_features_test['Annuity over income in %'] =  (100*desc_features_test['Annuity over income in %'])
    desc_features_test['Age'] =  round(-desc_features_test['Age']/365,1)
    desc_features_test['Employment duration in year'] = round(-desc_features_test['Employment duration in year']/365,1)

 
    cat_list = desc_features_train.columns
    cat_list_float = desc_features_train.columns[1:len(cat_list)]
    desc_features_train_print = pd.DataFrame()
    desc_features_train_print['Difficulties payment'] = desc_features_train['Difficulties payment']
    
    for cat in cat_list_float:
        desc_features_train_print[cat] = pd.to_numeric(desc_features_train[cat],errors='coerce')
        desc_features_train_print[cat] = desc_features_train_print[cat].map('{:,.1f}'.format)
    
    
   
    cust_features =  pd.DataFrame()
    cust_features_print =  pd.DataFrame()
    
    st.write("Main features concerning the customer")
    if user_input in app_train_domain.index.to_list():
        cust_features = pd.to_numeric(desc_features_train[cat_list_float].loc[user_input])
        cust_features_print = cust_features
        cust_features_print = cust_features_print.map('{:,.1f}'.format)
 
       
    else :
        cust_features = pd.to_numeric(desc_features_test[cat_list_float].loc[user_input])
        cust_features_print = cust_features
        cust_features_print = cust_features_print.map('{:,.1f}'.format)
 
    cust_features_print = cust_features_print.to_frame().T
   
    st.table(cust_features_print)
    
   
    st.write("Main features of similar customers")
    st.table(desc_features_train_print.loc[index_neighbors_id])
   
    #Compare the values of a selected set of features with the values of those features for all others customers 
    with st.beta_expander("Make comparison with others customers"):
        col1_1, col2_1 = st.beta_columns([10, 20]) 
        with col1_1:    
            st.write('Select features for which you want to compare the customer with others ones')
            option = [] 
            i=0
            with st.form(key='Selecting features'):
           
              for cat in cat_list_float:
                  option.append(st.checkbox(cat))
                  i = i + 1 
              submit_button = st.form_submit_button(label='Submit')
    
        with col2_1:
            if submit_button:
                cat_selected = [] 
                i=0
                while i < len(option):
                    if option[i] == True:
                        cat_selected.append(cat_list_float[i])
                    i = i+1
                radar_plot(cust_features,desc_features_train,cat_selected)
  
    #For categorical features, calculate for each possible values the number of loans with payment difficulties
    with st.beta_expander("Percentage of customers with payment difficulties for values of a particular feature"):
        col1_2, col2_2 = st.beta_columns([5, 20]) 
        #feature_histo = ['ORGANIZATION_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']
        feature_histo = ["Level of education",
                         "Family status",
                         'Type of housing',
                         'Gender'
                        ]
        feature_histo_dic = {"Level of education":'NAME_EDUCATION_TYPE',
                             "Family status":'NAME_FAMILY_STATUS',
                             'Type of housing':'NAME_HOUSING_TYPE',
                             'Gender':'GENDER'
                             }
        with col1_2:
            sel_feature = st.selectbox('Select the feature for which you want to make comparison:',feature_histo)
            select_feature = feature_histo_dic[sel_feature]
            app_train_col = pd.DataFrame(columns = [sel_feature,'Payment difficulties','Count'])
            
            col_features = [col for col in app_train_domain.columns if select_feature in col]
            for col in col_features:
                if user_input in app_train_domain.index.to_list():
                    if app_train_domain[col].loc[user_input]==1:
                        st.write('The {} of the customer is {}'.format(sel_feature,col).replace(select_feature+'_',''))
                else:
                     if app_test_domain[col].loc[user_input]==1:
                         st.write('The {} of the customer is {}'.format(sel_feature,col.replace(select_feature+'_','')))
        with col2_2:    
          
            for col in col_features:
                         
                count_pay_OK = len(app_train_domain[(app_train_domain['TARGET']==0) & (app_train_domain[col]==1)].index)
                count_pay_NOK = len(app_train_domain[(app_train_domain['TARGET']==1) & (app_train_domain[col]==1)].index)
                perc_ok ="{0:.1f}".format(((count_pay_OK/(count_pay_OK+count_pay_NOK))*100))
                perc_nok ="{0:.1f}".format(((count_pay_NOK/(count_pay_OK+count_pay_NOK))*100))
                new_row_OK = {sel_feature:col.replace(select_feature+'_',''), 'Payment difficulties': 'No payment difficulties',
                              'Count':count_pay_OK,'Percentage':perc_ok}
                new_row_NOK = {sel_feature:col.replace(select_feature+'_',''), 'Payment difficulties': 'Payment difficulties', 
                               'Count':count_pay_NOK,'Percentage':perc_nok}
                #append row to the dataframe
                app_train_col = app_train_col.append(new_row_OK, ignore_index=True)
                app_train_col = app_train_col.append(new_row_NOK, ignore_index=True)
            
            fig = px.bar(app_train_col, x=sel_feature,
                         y="Count", color="Payment difficulties",
                         text='Percentage',
                         color_discrete_sequence=['#90ee90', '#ff4500'],
                         width =1200,
                         height=600)
            st.plotly_chart(fig)
           