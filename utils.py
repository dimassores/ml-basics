import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_2d_embedding(X, y, labels, size, title=None):
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': size})
        
    plt.xticks([]), plt.yticks([])
    
    if title is not None:
        plt.title(title)
        
def make_dist_plot(x, hue, data):
    sns.set(rc={'figure.figsize':(15,10)})
    sns.set(font_scale = 1)
    sns.histplot(data=data, 
                 x=x, 
                 hue=hue, 
                 multiple="dodge",
                 element="poly",
                 stat="density")
    plt.title(x, fontsize=20)
    plt.show()
    
def generate_grouped_dataset_to_prop_plot(df, x_column, split_column, aux_column):
    
    grouped = df[[split_column, x_column, aux_column]].groupby([x_column, split_column]).count()
    state = df[[x_column,  aux_column]].groupby([x_column]).count()
    prop_data = grouped.div(state, level = x_column).reset_index().rename(columns = {aux_column: 'prop'})
    
    return prop_data

def make_roc_curve(X_test, y_test, list_of_model_objects, list_of_model_names):

    
    for index, model in enumerate(list_of_model_objects):
        
        model_name = list_of_model_names[index]
        
        current_model_probs = model.predict_proba(X_test)
        current_model_probs = current_model_probs[:, 1]
        current_auc = roc_auc_score(y_test, current_model_probs)
        
        print("%s: ROC AUC=%.3f" % (model_name, current_auc))
        current_fpr, current_tpr, _ = roc_curve(y_test, current_model_probs)
        
        plt.plot(current_fpr, current_tpr, marker='.', label= model_name)
        
    # naive model
    
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    
    # plot setup
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()