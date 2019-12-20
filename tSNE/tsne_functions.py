import random

import pandas as pd
import openTSNE as ot
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.linear_model import LogisticRegression

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
              plotting_params: dict = {'s': 1}):
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
    if classes is None:
        data = pd.DataFrame()
        for tsne in tsnes:
            x = [x[0] for x in tsne]
            y = [x[1] for x in tsne]
            data = data.append(pd.DataFrame({'x': x, 'y': y}))
        plt.scatter(data['x'],data['y'] , alpha=0.5, **plotting_params)
    else:
        if len(tsnes) != len(names):
            raise ValueError('N of tSNEs must match N of their name lists')
        data = pd.DataFrame()
        for tsne, name, group in zip(tsnes, names, range(len(tsnes))):
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
        all_colours = all_colours * (len(class_names) // len(all_colours) + 1)
        selected_colours = random.sample(all_colours, len(class_names))
        # colour_idx = range(len(class_names))
        colour_dict = dict(zip(class_names, selected_colours))

        fig = plt.figure()
        ax = plt.subplot(111)
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
                ax.scatter(class_data['x'],class_data['y'],
                       c=[colour_dict[class_name] for class_name in class_data['class']],
                       label=class_name, **plotting_params_class)
        if legend:
            handles, labels = fig.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
            for handle in legend.legendHandles:
                handle._sizes = [10]
                handle.set_alpha(1)


def analyse_tsne(data1,data2,col_data,label:str='cell_type',labels=['region'],perplexities_range: list = [50, 500],
              exaggerations: list = [12, 1.2],
              momentums: list = [0.6, 0.94],initial_split:int=1,tsne1=None):
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
    tsne2 = tsne1.prepare_partial(data2,initialization="median",k=30)
    col_data1=col_data.loc[data1.index,:]
    col_data2=col_data.loc[data2.index,:]
    
    for lab in labels:
        plot_tsne([tsne1,tsne2], classes=[dict(zip(col_data1.index,col_data1[lab])),
                                  dict(zip(col_data2.index,col_data2[lab]))
                                  ], names=[data1.index,data2.index], legend=True,
              plotting_params = [{'alpha': 0.2,'s':1},{'alpha': 1,'s':1}])
    plot_tsne([tsne1,tsne2], classes=[dict(zip(col_data1.index,col_data1[label])),
                                  dict(zip(col_data2.index,col_data2[label]))
                                  ], names=[data1.index,data2.index], legend=True,
              plotting_params = [{'alpha': 0.2,'s':1},{'alpha': 1,'s':1}])
    
    classifier = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne1,col_data1[label])
    
    evaluate_classifier(classifier=classifier,data=tsne2,col_data2=col_data,label=label)
    return tsne1,tsne2,classifier

def make_log_regression(data1,data2,col_data,label='cell_type',
                        logreg={'penalty':'l1','C':0.8,'random_state':0,'solver':'saga','n_jobs':8}):
    log_reg=LogisticRegression(**logreg).fit(data1, col_data.loc[data1.index,label])
    print('** Statistics train **')
    evaluate_classifier(classifier=log_reg,data=data1,col_data=col_data,label=label)
    print('** Statistics test **')
    evaluate_classifier(classifier=log_reg,data=data2,col_data=col_data,label=label)
    return log_reg
    
def evaluate_classifier(classifier,data,col_data,label):
    prediction=classifier.predict(data)
    truth=col_data.loc[data.index,label].values
    labels=list(set(col_data.loc[data.index,label]))
    precision, recall, fscore, support = score(y_true=truth, y_pred=prediction,labels=labels)
    precision_all, recall_all, fscore_all, support_all=score(
        y_true=truth, y_pred=prediction,labels=labels,average='weighted')
    print(pd.DataFrame({'class':np.append(classifier.classes_,
               'weighted average'),'precision':np.append(precision,precision_all),
             'recall':np.append(recall,recall_all),
             'fscore':np.append(fscore,fscore_all),
            'N true':np.append(support,support_all)}))
    prediction_p=classifier.predict_proba(data)
    print('ROC AUC score (weighted average)',roc_auc_score(y_true=truth, y_score=prediction_p,
                                                           multi_class='ovr', average='weighted'))
    
def savePickle(file, object):
    f = open(file, 'wb')
    pickle.dump(object, f)
    f.close()


def loadPickle(file):
    pkl_file = open(file, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result

