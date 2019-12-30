import random

import pandas as pd
import openTSNE as ot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score,roc_curve, auc
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import label_binarize

def make_tsne(data: pd.DataFrame, perplexities_range: list = [50, 500],
              exaggerations: list = [12, 1.2],
              momentums: list = [0.6, 0.94], random_state=0,initial_split:int=1) -> ot.TSNEEmbedding:
    """
    Make tsne embedding. Uses openTSNE Multiscale followed by optimizations. Each optimization has exaggeration and
    momentum parameter - these are used sequentially from exaggerations and momenutms lists, which must be of same
    lengths. There are as many optimizations as are lengths of optimization parameter lists.
    :param data: Samples in rows,features in columns. Must be numeric
    :param perplexities_range: Used for openTSNE.affinity.Multiscale
    :param exaggerations: List of exaggeration parameters for sequential optimizations
    :param momentums: List of momentum parameters for sequential optimizations
    :param random_state: random state
    :param initial_split: tSNE embedding is first performed on N/initial_split samples where N is N of samples
        and then extended to the remaining samples. If 1 tSNE is performed on whole dataset at once.
    :return: Embedding: functions as list of lists, where 1st object in nested list is x position and 2nd is y.
        There is one nested list for each sample
    """
    if len(exaggerations) != len(momentums):
        raise ValueError('Exagerrations and momenutms list lengths must match')
    if initial_split<1:
        raise ValueError('Initial split must be at least 1, meaning that data is not split.')
    
    if initial_split>1:
        np.random.seed(random_state)
        n=data.shape[0]
        indices = np.random.permutation(list(range(n)))
        reverse = np.argsort(indices)
        sub1, sub2 = data.iloc[indices[:int(n/initial_split)],:], data.iloc[indices[int(n/initial_split):],:]
    else:
        sub1=data
    
    init1 = ot.initialization.pca(sub1,random_state=random_state)
    affinities1 = ot.affinity.Multiscale(sub1, perplexities=perplexities_range,
                                                        metric="cosine", n_jobs=30, random_state=random_state)
    embedding1 = ot.TSNEEmbedding(init1, affinities1, negative_gradient_method="fft", n_jobs=30)
    for exaggeration, momentum in zip(exaggerations, momentums):
        embedding1 = embedding1.optimize(n_iter=250, exaggeration=exaggeration, momentum=momentum)
    
    if initial_split>1:
        embedding2 = embedding1.prepare_partial(sub2, k=3,perplexities=np.array(perplexities_range)/100)
        embedding_init = np.vstack((embedding1, embedding2))[reverse]
        embedding_init=embedding_init/ (np.std(embedding_init[:, 0]) * 10000)
        
        affinities = ot.affinity.Multiscale(data, perplexities=perplexities_range,
                                                           metric="cosine", n_jobs=30, random_state=random_state)
        embedding = ot.TSNEEmbedding(embedding_init,affinities,learning_rate=1000,negative_gradient_method="fft",n_jobs=30,
                          random_state=random_state)
        for exaggeration, momentum in zip(exaggerations, momentums):
            embedding = embedding.optimize(n_iter=250, exaggeration=exaggeration, momentum=momentum)
        
        
    else:
        embedding=embedding1
    
    return embedding

def plot_tsne(tsnes: list, classes: list = None, names: list = None, legend: bool = False,
              plotting_params: dict = {'s': 0.2,'alpha':0.2},title=None,colour_dict=None,fig_data=None,
              order_legend:list=None):
    """
    Plot tsne embedding
    :param tsne: List of embeddings, as returned by make_tsne
    :param classes: List of class annotations (dict), one per tsne.
    If not None colour each item in tSNE embedding by class.
    Keys: names matching names of tSNE embedding, values: class
    :param names: List of lists. Each list contains names for items in corresponding tSNE embedding.
    :param legend: Should legend be added
    :param plotting_params: plt.scatter parameters. Can be: 1.) List with dicts (single or nested, as below) for
    each tsne,  2.) dict with class names as keys and parameters dicts as values, 3.) dict of parameters.
    :return:
    """
    if fig_data is None:
        fig, ax = plt.subplots()
    else:
        fig,ax=fig_data
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title is not None:
        fig.suptitle(title)
    if classes is None:
        data = pd.DataFrame()
        for tsne in tsnes:
            if not isinstance(tsne,np.ndarray):
                tsne=np.array(tsne)
            x = [x[0] for x in tsne]
            y = [x[1] for x in tsne]
            data = data.append(pd.DataFrame({'x': x, 'y': y}))
        ax.scatter(data['x'],data['y'] , alpha=0.5, **plotting_params)
    else:
        if len(tsnes) != len(names):
            raise ValueError('N of tSNEs must match N of their name lists')
        data = pd.DataFrame()
        for tsne, name, group in zip(tsnes, names, range(len(tsnes))):
            if not isinstance(tsne,np.ndarray):
                tsne=np.array(tsne)
            x = [x[0] for x in tsne]
            y = [x[1] for x in tsne]
            data = data.append(pd.DataFrame({'x': x, 'y': y, 'name': name, 'group': [group] * len(x)}))
        if names is not None and classes is not None:
            classes_extended = []
            for row in data.iterrows():
                row=row[1]
                group_classes=classes[int(row['group'])]
                name=row['name']
                if name in group_classes.keys():
                    classes_extended.append(group_classes[name])
                else:
                    classes_extended.append('NaN')
            data['class']=classes_extended
        class_names = set(data['class'])
        all_colours = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c',
                       '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',
                       '#ffd8b1',
                       '#000075', '#808080', '#000000']
        if colour_dict is None:
            all_colours = all_colours * (len(class_names) // len(all_colours) + 1)
            selected_colours = random.sample(all_colours, len(class_names))
            # colour_idx = range(len(class_names))
            colour_dict = dict(zip(class_names, selected_colours))

        for group_name,group_data in data.groupby('group'):
            for class_name,class_data in group_data.groupby('class'):
                plotting_params_point=[]
                if isinstance(plotting_params,list):
                    plotting_params_group=plotting_params[int(group_name)]
                    plotting_params_class=plotting_params_group
                else:
                    plotting_params_class = plotting_params
                if isinstance(list(plotting_params_class.values())[0], dict):
                    plotting_params_class = plotting_params_class[class_name]
                else:
                    plotting_params_class = plotting_params_class
                if 'marker' not in plotting_params_class.keys():
                    plotting_params_class['marker']='o'
                ax.scatter(class_data['x'],class_data['y'],
                       c=[colour_dict[class_name] for class_name in class_data['class']],
                       label=class_name, **plotting_params_class)
        if legend:
            handles, labels = fig.gca().get_legend_handles_labels()
            if order_legend is not None:
                labels=[str(label) for label in labels]
                order_legend=[str(label) for label in order_legend]
                legend_order_dict=dict(zip(order_legend,range(len(order_legend))))
                legend_dict=dict(zip(labels,handles))
                legend_with_order={legend_order_dict[label]:label for label in labels}
                labels=[label for idx, label in sorted(legend_with_order.items(), key=lambda item: item[0])]
                handles=[legend_dict[label] for label in labels]      


            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = ax.legend( handles,labels,loc='center left', bbox_to_anchor=(1, 0.5))
            for handle in legend.legendHandles:
                handle._sizes = [10]
                handle.set_alpha(1)




def analyse_tsne(data1,data2,col_data,label:str='cell_type',labels=['region','cell_types_fine'],perplexities_range: list = [50, 500],
              exaggerations: list = [12, 1.2],
              momentums: list = [0.6, 0.94],initial_split:int=1,tsne1=None,colour_dicts=None):
    """
    Makes tsne on data1 and embeds data2. Plots both overlaping with data2 having higher alpha.
    Trains calssifier on data1 and predicts cell type on data2.
    :param data: Scaled expression data, cells in rows, genes in columns. Data1 - reference, data2 - test.
    :param col_data: Data describing cells. Cell names in index, data in columns.
    :param label: A column in cell_data. Colours by this in tSNE and uses as y for classification.
    """
    if tsne1 is None:
        tsne1=make_tsne(data1,perplexities_range=perplexities_range,
                    exaggerations=exaggerations,momentums=momentums,initial_split=initial_split)
    tsne2 = tsne1.prepare_partial(data2.loc[:,data1.columns],initialization="median",k=30)
    for exaggeration, momentum in zip(exaggerations, momentums):
        tsne2 = tsne2.optimize(n_iter=50, exaggeration=exaggeration, momentum=momentum)
        
    col_data1=col_data.loc[data1.index,:]
    col_data2=col_data.loc[data2.index,:]
    
    for lab in labels:
        legend=True
        if len(col_data[lab].unique())>15:
            legend=False
        colour_dict=None
        if colour_dicts is not None:
            if lab in colour_dicts.keys():
                colour_dict=colour_dicts[lab]
        plot_tsne([tsne1,tsne2], classes=[dict(zip(col_data.index,col_data[lab])),
                                  dict(zip(col_data.index,col_data[lab]))
                                  ], names=[data1.index,data2.index], legend=legend,
              plotting_params = [{'alpha': 0.2,'s':0.1},{'alpha': 1,'s':0.1}],title=lab,colour_dict=colour_dict)
    
    colour_dict=None
    if colour_dicts is not None:
        if label in colour_dicts.keys():
            colour_dict=colour_dicts[label]
    plot_tsne([tsne1,tsne2], classes=[dict(zip(col_data.index,col_data[label])),
                                  dict(zip(col_data.index,col_data[label]))
                                  ], names=[data1.index,data2.index], legend=True,
              plotting_params = [{'alpha': 0.2,'s':0.1},{'alpha': 1,'s':0.1}],title=label,colour_dict=colour_dict)
    
    classifier = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne1,col_data1[label])
    
    evaluate_classifier(classifier=classifier,data=pd.DataFrame(tsne2,index=data2.index),col_data=col_data,label=label)
    return tsne1,tsne2,classifier

def embed_tsne_new(data1,data2,col_data1,label:str='cell_type',perplexities_range: list = [50, 500],
              exaggerations: list = [12, 1.2],
              momentums: list = [0.6, 0.94],initial_split:int=1,tsne1=None,colour_dicts=None):
    """
    Makes tsne on data1 and embeds data2. Plots both overlaping with data2 having higher alpha.
    Trains calssifier on data1 and predicts cell type on data2.
    :param data: Scaled expression data, cells in rows, genes in columns. Data1 - reference, data2 - test.
    :param col_data: Data describing cells. Cell names in index, data in columns.
    :param label: A column in cell_data. Colours by this in tSNE and uses as y for classification.
    """
    if tsne1 is None:
        tsne1=make_tsne(data1,perplexities_range=perplexities_range,
                    exaggerations=exaggerations,momentums=momentums,initial_split=initial_split)
    tsne2 = tsne1.prepare_partial(data2.loc[:,data1.columns],initialization="median",k=30)
    for exaggeration, momentum in zip(exaggerations, momentums):
        tsne2 = tsne2.optimize(n_iter=50, exaggeration=exaggeration, momentum=momentum)
    
    col_data1=col_data1.loc[data1.index,:]
    
    classifier = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne1,col_data1[label])
    predictions=predict(classifier,data=pd.DataFrame(tsne2,index=data2.index))
    
    plot_tsne([tsne1,tsne2],
          [dict(zip(data1.index,['reference']*data1.shape[0])),
          dict(zip(data2.index,['added']*data2.shape[0]))],
          [data1.index,data2.index],
          legend=True,plotting_params = [{'s': 0.1,'alpha':0.05},{'s': 0.1,'alpha':1}],
              colour_dict={'reference':'#082173','added':'#850927'},title='reference and added data')
    
    colour_dict=None
    if colour_dicts is not None:
         if label in colour_dicts.keys():
                colour_dict=colour_dicts[label]
    plot_tsne([tsne1,tsne2],
          [dict(zip(col_data1.index,col_data1[label])),predictions],
          [data1.index,data2.index],
          legend=True,plotting_params = [{'s': 0.1,'alpha':0.05},{'s': 1,'alpha':1}],title=label,colour_dict=colour_dict)
   
    return tsne1,tsne2,classifier, predictions

def tsne_add(tsne1,data2:pd.DataFrame,
              exaggerations: list = [2, None],momentums: list = [0.5, 0.8], n_iters_optimize:int=[5,100]):
    tsne2 = tsne1.prepare_partial(data2,initialization="median",k=30)
    for exaggeration, momentum,n_iter in zip(exaggerations, momentums,n_iters_optimize):
        tsne2 = tsne2.optimize(n_iter=n_iter, exaggeration=exaggeration, momentum=momentum, 
                               max_grad_norm=0.25,learning_rate=0.1)
    return tsne2
    

def make_log_regression(data1,data2,col_data,label='cell_type',
                        logreg={'penalty':'l1','C':0.8,'random_state':0,'solver':'saga','n_jobs':30},log_reg=None):
    if log_reg is None:
        log_reg=LogisticRegression(**logreg).fit(data1, col_data.loc[data1.index,label])
    
    print('** Used features per class **')
    features=data1.columns.values
    class_feats=[]
    for group,weights in zip(log_reg.classes_,log_reg.coef_):
        class_feats.append({'Class':group,'N features':len(features[weights!=0])})
    print(pd.DataFrame(class_feats))
    print('** Statistics train **')
    predicted=evaluate_classifier(classifier=log_reg,data=data1,col_data=col_data,label=label)
    print('** Statistics test **')
    predicted=evaluate_classifier(classifier=log_reg,data=data2,col_data=col_data,label=label)
    return log_reg,predicted
    
def evaluate_classifier(classifier,data,col_data,label):
    prediction=classifier.predict(data)
    truth=col_data.loc[data.index,label].values
    precision, recall, fscore, support = score(y_true=truth, y_pred=prediction,labels=classifier.classes_)
    precision_all, recall_all, fscore_all, support_all=score(
        y_true=truth, y_pred=prediction,labels=classifier.classes_,average='weighted')
    classes=np.append(classifier.classes_, 'weighted average')
    precisions=np.append(precision,precision_all)
    recalls=np.append(recall,recall_all)
    fscores=np.append(fscore,fscore_all)
    supports=np.append(support,support_all)
    if len(classes) == len(precisions) == len (recalls)==len(fscores)==len(supports):
        print(pd.DataFrame({'class':classes,
             #'precision':np.around(precisions,2),
             #'recall':np.around(recalls,2),
             'fscore':np.around(fscores,2),
            'support':supports}))
    else:
        print('class:',classes,
              #'; precision:',np.around(precisions,2),
             #'; recall:',np.around(recalls,2),
             '; fscore:',np.around(fscores,2),
            '; support',np.around(supports,2))
    prediction_p=classifier.predict_proba(data)
    print('ROC AUC score (weighted average)',round(roc_auc_score(y_true=truth, y_score=prediction_p,
                                                           multi_class='ovr', average='weighted'),2))
    print('ROC AUC score (micro)',round(roc_auc_micro(classifier=classifier, x=data,y=truth),2))
    return pd.DataFrame(prediction,index=data.index)

def roc_auc_micro(classifier, x:pd.DataFrame,y):
    #Micro
    n_classes=len(classifier.classes_)
    if hasattr(classifier,'decision_function'):
        y_score = classifier.decision_function(x)
    else:
        y_score = classifier.predict_proba(x)
    y = label_binarize(y,classes=classifier.classes_)
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    return auc(fpr, tpr)

    
def predict(classifier,data:pd.DataFrame):
    prediction=classifier.predict(data)
    return dict(zip(data.index,prediction))

def savePickle(file, object):
    f = open(file, 'wb')
    pickle.dump(object, f)
    f.close()


def loadPickle(file):
    pkl_file = open(file, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result

def normalise_counts(counts=pd.DataFrame):
    scale=counts.sum(axis=1)
    counts=counts.T
    counts=counts/scale
    counts=counts.T
    counts=counts*1000000
    counts=counts+1
    return np.log(counts)
    
