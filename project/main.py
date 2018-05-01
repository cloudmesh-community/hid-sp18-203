import requests
import matplotlib
matplotlib.use('agg')
from flask import Flask
from flask import Response, request
import numpy as np
from sklearn.externals.joblib import Memory
from os import listdir
import pandas as pd
import random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import _tree
import json
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


app = Flask(__name__)


def download_data(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)


def data_partition(test_ratio, x, y):
    global x_train, y_train, y_test, x_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=0,
                                                        stratify=y)


def get_data(file_path):
    data = pd.read_csv(file_path)
    y = data.quality  # The thing we want to predict
    #x = data.drop('quality', axis=1)  # the rest of the data
    x = data.drop(['free sulfur dioxide', 'fixed acidity', 'citric acid', 'quality'], axis=1)
    #x = data.drop(['free sulfur dioxide','quality','fixed acid'], axis=1)

    return x, y


# The following four functions were borrowed from
# https://aysent.github.io/2015/11/08/random-forest-leaf-visualization.html
# They are used to draw plots of the random forest
def leaf_depths(tree, node_id=0):
    '''
    tree.children_left and tree.children_right store ids
    of left and right chidren of a given node
    '''
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    '''
    If a given node is terminal, 
    both left and right children are set to _tree.TREE_LEAF
    '''
    if left_child == _tree.TREE_LEAF:

        '''
        Set depth of terminal nodes to 0
        '''
        depths = np.array([0])

    else:

        '''
        Get depths of left and right children and
        increment them by 1
        '''
        left_depths = leaf_depths(tree, left_child) + 1
        right_depths = leaf_depths(tree, right_child) + 1

        depths = np.append(left_depths, right_depths)

    return depths


def leaf_samples(tree, node_id=0):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child == _tree.TREE_LEAF:

        samples = np.array([tree.n_node_samples[node_id]])

    else:

        left_samples = leaf_samples(tree, left_child)
        right_samples = leaf_samples(tree, right_child)

        samples = np.append(left_samples, right_samples)

    return samples


def draw_tree(ensemble, tree_id=0):
    plt.figure(figsize=(8, 8))
    plt.subplot(211)

    tree = ensemble.estimators_[tree_id].tree_

    depths = leaf_depths(tree)
    plt.hist(depths, histtype='step', color='#9933ff',
             bins=range(min(depths), max(depths) + 1))

    plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)

    plt.subplot(212)

    samples = leaf_samples(tree)
    plt.hist(samples, histtype='step', color='#3399ff',
             bins=range(min(samples), max(samples) + 1))

    plt.xlabel("Number of samples in leaf nodes (tree %s)" % tree_id)

    plt.savefig('vis_one.png')


def draw_ensemble(ensemble):
    plt.figure(figsize=(8, 8))
    plt.subplot(211)

    depths_all = np.array([], dtype=int)

    for x in ensemble.estimators_:
        tree = x.tree_
        depths = leaf_depths(tree)
        depths_all = np.append(depths_all, depths)
        plt.hist(depths, histtype='step', color='#ddaaff',
                 bins=range(min(depths), max(depths) + 1))

    plt.hist(depths_all, histtype='step', color='#9933ff',
             bins=range(min(depths_all), max(depths_all) + 1),
             weights=np.ones(len(depths_all)) / len(ensemble.estimators_),
             linewidth=2)
    plt.xlabel("Depth of leaf nodes")

    samples_all = np.array([], dtype=int)

    plt.subplot(212)

    for x in ensemble.estimators_:
        tree = x.tree_
        samples = leaf_samples(tree)
        samples_all = np.append(samples_all, samples)
        plt.hist(samples, histtype='step', color='#aaddff',
                 bins=range(min(samples), max(samples) + 1))

    plt.hist(samples_all, histtype='step', color='#3399ff',
             bins=range(min(samples_all), max(samples_all) + 1),
             weights=np.ones(len(samples_all)) / len(ensemble.estimators_),
             linewidth=2)
    plt.xlabel("Number of samples in leaf nodes")

    plt.savefig('vis_all.png')


@app.route('/')
def index():
    data =  {}
    data['This is Jack Clarke\'s REST API Project'] = True
    data[' '] = ""
    data['Data must be partitioned before running any experiments'] = True
    data['Partition Data'] = '/api/data/partition/[filename]/[Percentage of data to test with]'
    data['Barebones Random Forest'] = '/api/experiment/rf'
    data['Improved Random Forest'] = '/api/experiment/rf/improved'
    data['Visualize one Tree in Random Forest'] = '/api/experiment/rf/vis'
    data['Visualize all Trees in Random Forest'] = '/api/experiment/rf/vis-all'
    data[''] = " "
    data['Included Data Set'] = 'red-wine.csv'
    json_data = json.dumps(data)
    resp = Response(json_data, status=200, mimetype='application/json')
    return resp


@app.route('/api/download/data')
def download():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    #download_data(url=url, filename='red-wine.csv')
    return "Data included in repository, for some reason every time I downloaded the zip file became corrupted."


@app.route('/api/data/partition/<filename>/<ratio>')
def format_data(filename, ratio):
    x, y = get_data(filename)
    ratio = float(ratio)
    data_partition(ratio, x, y)
    return "Successfully Partitioned"


@app.route('/api/experiment/rf')
def randomForest():
    rf = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state= random.randint(1, 9999))
    rf.fit(x_train, y_train)
    predicted_train = rf.predict(x_train)
    predicted_test = rf.predict(x_test)
    test_score = r2_score(y_test, predicted_test)
    data = {}
    data['OOB Score'] = rf.oob_score_
    data['R2 Score'] = test_score
    errors = abs(predicted_test - y_test)

    importances = pd.DataFrame(
        {'feature': x_train.columns, 'importance': np.round(rf.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    print(importances.head(20))

    data['Mean Absolute Error:'] = round(np.mean(errors), 2), 'degrees.'
    #data['Mean Squared: '] = mean_squared_error(y_test, y_pred)
    #data['First Five Predictions'] = predicted_test[0:4]
    json_data = json.dumps(data)
    resp = Response(json_data, status=200, mimetype='application/json')
    return resp


@app.route('/api/experiment/rf/improved')
def randomForestImproved():
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=1000,
                                                                                   random_state=0))#random.randint(1, 9999)))
    hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                       'randomforestregressor__max_depth': [None, 5, 3, 1]}

    clf = GridSearchCV(pipeline, hyperparameters, cv=10)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    data = {}
    errors = abs(y_pred - y_test)
    data['Mean Absolute Error:'] = round(np.mean(errors), 2), 'degrees.'
    data['Best Parameters: '] = clf.best_params_
    data['R2 Score: '] = r2_score(y_test, y_pred)
    data['Mean Squared: '] = mean_squared_error(y_test, y_pred)
    json_data = json.dumps(data)
    resp = Response(json_data, status=200, mimetype='application/json')
    return resp

@app.route('/api/experiment/rf/vis')
def visualizor():
    x, y = get_data('red-wine.csv')
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=random.randint(1, 9999))
    rf.fit(x_train, y_train)
    draw_tree(rf)
    return('Saved Graph')

@app.route('/api/experiment/rf/vis-all')
def visualAll():
    x, y = get_data('red-wine.csv')
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=random.randint(1, 9999))
    rf.fit(x_train, y_train)
    draw_ensemble(rf)
    return('Saved Graph')


if __name__ == '__main__':
    app.run(debug=True)
