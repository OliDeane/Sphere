import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as skl


def generate_train_test_data(training_data_frame , test_split = 0.2):
    '''
    Function takes in a dataframe containing all training data and outputs:
        - train / test data frame 
        - train x & y , test x & y numpy arrays for model training (label encoded y)
   
    Requirements: 
        - The first n-22 columns of the dataframe contain the features
        - Label encoding for activity is contained within column that is named 'max_label_encoded'
    '''
        
    train_df, test_df = train_test_split(training_data_frame, test_size= test_split)
      
    n = len(training_data_frame.columns)-21
    feature_cols = training_data_frame.columns[:n]
    
    train_x = train_df[feature_cols].values
    train_y_encoded = train_df['max_label_encoded'].values
    
    test_x = test_df[feature_cols].values
    test_y_encoded = test_df['max_label_encoded'].values
    
    #Convert test and train data into np arrays as required by sklearn 
    train_x = np.asanyarray(train_x)
    train_y_encoded = np.asanyarray(train_y_encoded)
    test_x = np.asanyarray(test_x)
    test_y_encoded = np.asanyarray(test_y_encoded)
    
    test_y = np.asanyarray(test_df[test_df.columns[n:n+20]].reset_index(drop=True))
    
    return train_df , test_df , train_x , train_y_encoded , test_x , test_y_encoded, test_y
        
    
    
def generate_probabilistic_predictions(y_pred):
    
    '''
    Input: model predictions as y_pred (label encoded array , single column)
    Output: pred_array -> Predictions over 20 column array (non-label encoded) , prob of 1 in prediction converted to 0.9525 and 0 converted to 0.0025
    '''
    
    #Construct label encoder that will be needed to do the inverse transformation of predicted y values
    label_encoder = preprocessing.LabelEncoder()
    input_classes = ['a_ascend', 'a_descend', 'a_jump' , 'a_loadwalk' , 'a_walk', 'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit' ,
                     't_sit_lie' , 't_sit_stand', 't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn']
    label_encoder.fit(input_classes)
    
    
    #Construct the prediction dataframe from the y_pred array, decode the label encoded value and contruct into 20 column array 
    df_pred = pd.DataFrame(y_pred, columns = ['Prediction'])
    df_pred['label'] = label_encoder.inverse_transform(df_pred['Prediction'])
    df_pred = pd.concat([df_pred ,pd.get_dummies(df_pred['label'])], axis=1)
    df_pred.drop(columns = ['Prediction', 'label'], inplace = True)
    
    #Transform the dataframe into an array
    df_y_pred = df_pred
    pred_array = np.asanyarray(df_y_pred)
    
    #Loop through the array to change 1 --> 0.9525 and 0 --> 0.0025 , this slightly amended distribution enables cross entropy loss to be calculated
    pred_array = pred_array*0.9525
    for i in range(len(pred_array)):
        for j in range(20):
            if pred_array[i][j] == 0:
                pred_array[i][j] = pred_array[i][j]+0.0025
            else:
                pass   
    
    return pred_array
    

###########################################################################################################
#Evaluation Metrics Calculation and Scoring Functions Below

def generate_true_scores(actual , predicted, model_name = 'Decision Tree' ):
    brier = brier_score(actual, predicted)
    cross = cross_entropy(actual, predicted)
    
    #Add error results to dataframe
    res = {'Model_Name': [model_name], 'brier score' : [brier]  ,'cross_entropy' : [cross]}
    results = pd.DataFrame(data=res)
    return results


#Define a function to calculate the Brier Score, takes in two arrays of 20 columns
def brier_score(actual_array, pred_array):
    class_weights = np.ones(20)
    total = 0 
    for i in range(len(actual_array)):
        row_score = 0
        for j in range(20):
            row_score += class_weights[j]*(actual_array[i][j]-pred_array[i][j])**2
        total+= row_score
    
    brier_score_final = total / len(actual_array)
    
    return brier_score_final   

#Define a function to calculate the Cross entropy loss, takes in two arrays of 20 columns
def cross_entropy(actual, predicted):
    total_loss = 0
    for i in range(len(actual)):
        loss = -np.sum(actual[i]*np.log(predicted[i]))
        total_loss += loss
    final_loss = total_loss / len(actual)
    return final_loss

def generate_pseudo_scores(test_y_encoded , y_pred_encoded, model_name = 'Decision Tree' ):
    Macro_F1 = skl.f1_score(test_y_encoded, y_pred_encoded , average = 'macro')
    Macro_precision_score = skl.precision_score(test_y_encoded, y_pred_encoded , average = 'macro')
    Macro_recall = skl.recall_score(test_y_encoded, y_pred_encoded , average = 'macro')
    #Add error results to data frame
    res = {'Model_Name': [model_name], 'Macro_F1' : [Macro_F1]  ,'Macro_precision_score' : [Macro_precision_score] , 'Macro_recall': [Macro_recall]}
    results = pd.DataFrame(data=res)
    return results




