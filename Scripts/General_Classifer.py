#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


import matplotlib.pyplot as plt
from matplotlib import *

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import itertools


# ## Functions to assist the main classifier, classification phase

# In[16]:


def Parameter_initialization():

    param_xgb = {'n_estimators':[200],
                   'min_samples_split':[302],
                   'max_depth':[11],
                   'min_samples_leaf':[1],
                   'max_features':[30],
                   'subsample':[0.8],
                   'learning_rate':[0.1]
                  }



    param_RF  = {'criterion':['gini','entropy'],
                  'n_estimators':[10,15,20,25,30],
                  'min_samples_leaf':[1,2,3],
                  'min_samples_split':[3,4,5,6,7], 
                  'random_state':[123]
                  }
    
    param_svm  = {'C':[ 1],
                      'gamma':[0.001],
                      'kernel' : ['rbf']
                      }

    param_KNN  = {'n_neighbors':[5,6,7,8,9,10],
                  'leaf_size':[1,2,3,5],
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree','kd_tree','brute']
                  }

    param_nn  = { 'solver': ['sgd', 'adam','lbfgs'],
              'alpha': 10.0 ** -np.arange(1, 7),
              'hidden_layer_sizes':np.arange(5, 12),
              'max_iter': [500,1000,1500],
             'activation': ['tanh', 'relu']
              }

    Model_dict = {'XGB':(GradientBoostingClassifier,0,param_xgb),
              'SVM':(SVC,0,param_svm),
              'KNN' :(KNeighborsClassifier,0,param_KNN),
              'RF':(RandomForestClassifier,0,param_RF),
              'NN': (MLPClassifier,0,param_nn)
              }
    
    return Model_dict


# In[10]:


def predict_values (classifier,X,Y,scoring):
    ##make the scoring metric a function
    if scoring in 'f1':
        scoring = 'f1_score'
    if scoring in 'neg_log_loss':
        scoring = 'log_loss'
    else:
        scoring = scoring + '_score'
    
    dtest_predictions = classifier.predict(X)
    dtest_predprob = classifier.predict_proba(X)
    if scoring == 'log_loss':
        dtest_score = eval(scoring)(y_true = Y,y_pred =dtest_predprob,labels = list(classifier.classes_))
    else:
        dtest_score = eval(scoring)(Y,dtest_predictions)
    return dtest_predictions, dtest_predprob , dtest_score


# In[11]:


def genral_purpose_gridsearch (model_name,alg,parameters,random_state,n_jobs,cv_folds,verbose,X_train,y_train,scoring,X_test,y_test):
    if model_name == 'SVM':
        alg = alg(probability=True,random_state=random_state)
    else :
        if model_name == 'KNN':
            alg = alg()
        else:
            alg = alg(random_state=random_state)
    gsearch1 = GridSearchCV(estimator = alg, 
    param_grid = parameters, scoring=scoring,n_jobs=n_jobs,iid=False, cv=cv_folds,verbose = verbose)

    gsearch1.fit(X_train,y_train)
    grid_test_predictions,grid_test_predictions_prob,test_accuracy_score = predict_values(gsearch1.best_estimator_,X_test,y_test,scoring)
    return gsearch1 , gsearch1.best_estimator_ , gsearch1.best_params_ , gsearch1.best_score_ ,grid_test_predictions , grid_test_predictions_prob,test_accuracy_score 


# ## Main Classifier

# In[23]:


def classifier_fit(alg_provided,alg,initialized_classifier,classifer_parameter,split_train_test,X,y,X_train,X_test,y_train,y_test,random_state,scoring,cv,cv_folds,printFeatureImportance, top_feat , gridsearch_usage , grid_serach_param , n_jobs , verbose):
    import matplotlib.pylab as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 12, 4
    
    ## Divide the provided data into train and test
    if split_train_test == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('X_train count : {} , X_test : {} '.format(len(X_train),len(X_test)))
    
    
    #################################################
    ## defining inputs
    if alg_provided == True:
        
        model_dict = {'chosen Alg' : (alg,classifer_parameter,grid_serach_param)}

    else:
        model_dict = Parameter_initialization()
        
    
    Output = {}
    Accuracy = {}
    Model = {}
    
    GS_Output = {}
    GS_Accuracy = {}
    GS_Model = {}
    
    for i in model_dict:
        print('Algorithim {}'.format(i))
        
        if alg_provided == True:
            if initialized_classifier == True:
                if i == 'SVM':
                    classifier = model_dict[i][0](**model_dict[i][1], probability = True).fit(X_train,y_train)
                else:   
                    classifier = model_dict[i][0](**model_dict[i][1]).fit(X_train,y_train)

        else:
            if i == 'SVM':
                classifier = model_dict[i][0](probability = True).fit(X_train,y_train)
            else:    
                classifier = model_dict[i][0]().fit(X_train,y_train)
        
        ## Score test set
        dtest_predictions,dtest_predprob,dtest_score = predict_values (classifier,X_test,y_test,scoring)
        
        ## Cross validation
        if cv == True:
            print('Cross Validation')
            Cross_validation_results = cross_val_score(classifier, X, y, cv=cv_folds, scoring = scoring)
            Output[i] = (dtest_predictions,dtest_predprob,dtest_score,classifier,np.mean(Cross_validation_results))
        
        ## collect default results in an output format
        print('{} for Algorithim {} is {}'.format(scoring,i,dtest_score))
        Output[i] = (dtest_predictions,dtest_predprob,dtest_score,classifier)
        Accuracy[i] = dtest_score
        Model[i] = classifier
        
        ##GridSearch
        if gridsearch_usage == True:
            print('Grid search')
            gsearch1 , gsearch1_best_estimator , gsearch1_best_params , gsearch1_best_score , grid_test_predictions, grid_test_predictions_prob , acc_score = genral_purpose_gridsearch(i,model_dict[i][0],model_dict[i][2],random_state,n_jobs,cv_folds,verbose,X_train,y_train,scoring, X_test,y_test)
        
            gs_classifier = gsearch1_best_estimator
    
            GS_Output[i] = (gsearch1,gsearch1_best_estimator,acc_score)
            GS_Accuracy[i] = acc_score
            GS_Model[i] = gsearch1_best_estimator
    #######################################################################################################################

    # Choosing Best of default based on accuracy
    if scoring.find('loss') != -1:
        max_value = np.min(list(Accuracy.values()))
    else:
        max_value = np.max(list(Accuracy.values()))
    position = int(np.where(list(Accuracy.values()) == max_value)[0])
    Max_Name = list(Accuracy.keys())[position]
    best_default_classifier = Output[Max_Name][3]
    print('Best Accuracy Performance Algorithim :{} with Accuracy {}'.format(Max_Name,max_value))
    
    ## Choosing Best of grid serach based on accuracy
    if gridsearch_usage == True:
        ##Here I check whether the metric is "loss" : smaller is better , "performance" : greater is better
        if scoring.find('loss') != -1:
            max_value = np.min(list(GS_Accuracy.values()))
        else:
            max_value = np.max(list(GS_Accuracy.values()))
            
        position = int(np.where(list(GS_Accuracy.values()) == max_value)[0])
        Max_Name = list(GS_Accuracy.keys())[position]
        best_gs_classifier = GS_Output[Max_Name][1]
        print('Best Accuracy Performance Algorithim :{} with Accuracy {}'.format(Max_Name,max_value))
        return Output,Accuracy,Model,GS_Output,GS_Accuracy,GS_Model,best_default_classifier,best_gs_classifier
    else:
        return Output,Accuracy,Model,GS_Output,GS_Accuracy,GS_Model,best_default_classifier
#######################################################################################################################
    ##After foor loop we chose the best and print the features and other graphs
    
    ## Choosing Best of default
#     max_value = np.max(list(Accuracy.values()))
#     position = int(np.where(list(Accuracy.values()) == max_value)[0])
#     Max_Name = list(Accuracy.keys())[position]
#     best_default_classifier = Output[Max_Name][3]
#     print('Best Accuracy Performance Algorithim :{} with Accuracy {}'.format(Max_Name,max_value))

                     
#     ## Choosing Best
#     max_value = np.max(list(GS_Accuracy.values()))
#     position = int(np.where(list(GS_Accuracy.values()) == max_value)[0])
#     Max_Name = list(GS_Accuracy.keys())[position]
#     best_gs_classifier = GS_Output[Max_Name][1]
#     print('Best Accuracy Performance Algorithim :{} with Accuracy {}'.format(Max_Name,Max))

#     ##Feature importance
#     if printFeatureImportance == True:
#         if alg_provided == False:
#             feature_importance_plot (printFeatureImportance,alg_provided,best_default_classifier,Model,top_feat)

#         else:
#             feature_importance_plot (printFeatureImportance,alg_provided,classifier,0,top_feat)
                     
#     plot_learning_curve(50,0.2,0,gsearch1_best_estimator,X,y,'neg_log_loss',-1,(7,5))
#     Plot_confusion_matrix(y_test,gsearch1_best_estimator,X_test,plt.cm.Blues)
#     if model == 'SVM' and Model_output['SVM'].kernel != 'linear':
#         print('SVM kernal is not Linear')
#     else:
#         feature_importance_plot (True,True,gsearch1_best_estimator,0,20)
       
        


# ## Plotting Evaluation

# In[17]:


def plot_learning_curve(n_splits,test_size,random_state,estimator,estimator_name,X,y,scoring_metric,n_jobs,figsize):
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    train_sizes, train_scores, test_scores = learning_curve(estimator,X,y,cv=cv,scoring = scoring_metric ,n_jobs = n_jobs)
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    
    plt.figure(figsize=figsize)
    plt.title('learning_curve')
    plt.ylabel(scoring_metric)
    plt.xlabel('training data size')
    plt.plot(train_sizes,train_scores_mean,'o-', color="r",label="Training score")
    plt.plot(train_sizes,test_scores_mean,'o-', color="g",label="Testing score")
    plt.legend(loc="best")
    plt.show()
    return plt


# In[19]:


def Plot_confusion_matrix(y_test,estimator,X_test,cmap):
    cm = confusion_matrix(y_test, estimator.predict(X_test))
    classes = estimator.classes_
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title('Confusion Matrix')
    tickmarks = np.arange(len(classes))
    plt.xticks(tickmarks, classes, rotation=45)
    plt.yticks(tickmarks,classes)


    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    return plt


# In[20]:


def feature_importance_plot (printFeatureImportance,alg_provided,classifier,Classifier_dict,top_feat,X):
    ##Feature importance
    if printFeatureImportance == True:
        if alg_provided == False:
            plt.subplot( len(Classifier_dict))
            for model in Classifier_dict:
                ##(gsearch1,gsearch1_best_estimator,acc_score)
                
                feat_imp = pd.Series(Classifier_dict[model].feature_importances_, X.columns).sort_values(ascending=False)
                Top20 = feat_imp.head(top_feat)

                Top20.plot(kind='bar', title='Feature Importances')
                plt.ylabel('Feature Importance Score')
                plt.show()

        else:
            feat_imp = pd.Series(classifier.feature_importances_, X.columns).sort_values(ascending=False)
            Top20 = feat_imp.head(top_feat)

            Top20.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()
        return plt


# In[ ]:




