''' 
Author  :   Hasan Basri Akçay 
linkedin:   <https://www.linkedin.com/in/hasan-basri-akcay/> 
medium  :   <https://medium.com/@hasan.basri.akcay>
kaggle  :   <https://www.kaggle.com/hasanbasriakcay>
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
        

def numerical_ttt_dist(train=None, test=None, val=None, features=[], agg_func='mean', target='target', bin_num=10, ncols=3, figsize=16, ylim=(), sharey=False):
    '''
    The Numerical Train Test Target Distribution function helps us to understand data distribution better. It can plot train, test, validation, and the target in 
    one graph for each feature.

    Args:
        train (DataFrame): the train dataframe containing features and target columns.
        test (DataFrame): the test dataframe containing features.
        val (DataFrame): the validation dataframe containing features and target columns.
        features (list): Numeric features in the dataframe.
        agg_func (str): The Pandas aggregation functions for the target.
        target (str): The target feature name.
        bin_num (int): Bin number for NumPy linspace.
        ncols (int): Column number of the graph.
        figsize (int): Figure size.
        ylim (tuple): Y-limits of the axes.
        sharey (bool): Controls sharing of properties among y (sharey) axes.

    Returns:
        DataFrame
    '''

    # Default Parameter Preparations
    if len(features) == 0:
        features = train.columns.tolist()
    
    alpha = 1 / sum(x is not None for x in [train, test, val])

    if len(ylim) == 0:
        np_ylim = np.zeros((len(features), 2))
        for index, feature in enumerate(features):
            mi = min(train[feature].min(), _return_min(test,feature), _return_min(val,feature))
            ma = max(train[feature].max(), _return_max(test,feature), _return_max(val,feature))
            bins = np.linspace(mi, ma, bin_num)

            out, bins = pd.cut(train[feature].values, bins=bins, retbins=True, right=False)
            train['bin']=out
            bins_target = train.groupby(['bin']).agg({target:agg_func})
            np_ylim[index] = min(bins_target[target]), max(bins_target[target])
            
            if val is not None:
                out, bins = pd.cut(val[feature].values, bins=bins, retbins=True)
                val['bin']=out
                val_bins_target = val.groupby(['bin']).agg({target:agg_func})
                np_ylim[index] = min(np_ylim[index][0], val_bins_target[target].min()), max(np_ylim[index][1], val_bins_target[target].max())
        ymin, ymax = np_ylim[:,0].min(), np_ylim[:,1].max()
        distance = ymax - ymin
        ymin, ymax = ymin-distance/10, ymax+distance/10
    else:
        ymin, ymax = ylim[0], ylim[1]

    # Graph and Stat Creations
    nrows = int(len(features) / ncols)
    if len(features) % ncols != 0:
        nrows += 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize, round(nrows*figsize/ncols)), sharey=sharey)
    df_stats = pd.DataFrame(columns=['feature', 'train_trend_changes', 'test_trend_changes', 'val_trend_changes', 'train_test_trend_corr',
                                    'train_val_trend_corr', 'val_test_trend_corr', 'train_target_trend_changes', 'val_target_trend_changes',
                                    'train_val_target_trend_corr'])
    for index, (ax,feature) in enumerate(zip(axes.ravel()[:len(features)],features)):
        # Graph
        mi = min(train[feature].min(), _return_min(test,feature), _return_min(val,feature))
        ma = max(train[feature].max(), _return_max(test,feature), _return_max(val,feature))
        bins = np.linspace(mi, ma, bin_num)
        
        out, bins = pd.cut(train[feature].values, bins=bins, retbins=True, right=False)
        train['bin']=out
        bins_target = train.groupby(['bin']).agg({'bin':'count', target:agg_func})
        bins_target.columns = ['bin_count', target]
        
        ax.hist(train[feature], bins=bins, alpha=alpha, density=True, label='train', color='tab:cyan')
        if test is not None:
            test_out, _ = pd.cut(test[feature].values, bins=bins, retbins=True, right=False)
            test['bin'] = test_out
            test_bins_target = test.groupby(['bin']).agg({'bin':'count'})
            test_bins_target.columns = ['bin_count']
            ax.hist(test[feature], bins=bins, alpha=alpha, density=True, label='test', color='tab:red')
        if val is not None:
            val_out, _ = pd.cut(val[feature].values, bins=bins, retbins=True, right=False)
            val['bin'] = val_out
            val_bins_target = val.groupby(['bin']).agg({'bin':'count', target:agg_func})
            val_bins_target.columns = ['bin_count', target]
            ax.hist(val[feature], bins=bins, alpha=alpha, density=True, label='val', color='tab:gray')
        ax.set_xlabel(feature)
        
        ax2 = ax.twinx()
        mean_bins = [(bins[i]+bins[i+1])/2 for i in range(0, len(bins)-1)]
        ax2.scatter(mean_bins, bins_target[target], label='train target rate', color='tab:purple')
        if val is not None:  ax2.scatter(mean_bins, val_bins_target[target], label='val target rate', color='tab:gray')
        ax2.set_ylim([ymin, ymax])
        if index == 0: ax2.legend(loc='upper right'), ax.legend(loc='lower left')
        if (index+1) % ncols == 0 or (index+1)==len(features): ax2.set_ylabel('Target Rate')
        if (index+1) % ncols == 1: ax.set_ylabel('Count')

        # Stats
        row_dict = {}
        row_dict['feature'] = feature
        row_dict['train_trend_changes'] = _find_trend_changes(bins_target['bin_count'])
        row_dict['train_target_trend_changes'] = _find_trend_changes(bins_target[target])

        if test is not None:
            row_dict['test_trend_changes'] = _find_trend_changes(test_bins_target['bin_count'])
            row_dict['train_test_trend_corr'] = np.corrcoef(bins_target['bin_count'].fillna(0), test_bins_target['bin_count'].fillna(0))[0, 1]
            if np.isnan(row_dict['train_test_trend_corr']): row_dict['train_test_trend_corr'] = 0
            if val is not None:
                row_dict['val_trend_changes'] = _find_trend_changes(val_bins_target['bin_count'])
                row_dict['val_target_trend_changes'] = _find_trend_changes(val_bins_target[target])
                row_dict['val_test_trend_corr'] = np.corrcoef(test_bins_target['bin_count'].fillna(0), val_bins_target['bin_count'].fillna(0))[0, 1]
                if np.isnan(row_dict['val_test_trend_corr']): row_dict['val_test_trend_corr'] = 0
        if val is not None:
            row_dict['val_trend_changes'] = _find_trend_changes(val_bins_target['bin_count'])
            row_dict['val_target_trend_changes'] = _find_trend_changes(val_bins_target[target])
            row_dict['train_val_trend_corr'] = np.corrcoef(bins_target['bin_count'].fillna(0), val_bins_target['bin_count'].fillna(0))[0, 1]
            row_dict['train_val_target_trend_corr'] = np.corrcoef(bins_target[target].fillna(0), val_bins_target[target].fillna(0))[0, 1]
            if np.isnan(row_dict['train_val_trend_corr']): row_dict['train_val_trend_corr'] = 0
            if np.isnan(row_dict['train_val_target_trend_corr']): row_dict['train_val_target_trend_corr'] = 0
        df_stats = df_stats.append(row_dict, ignore_index=True)
    train.drop('bin', axis=1, inplace=True)
    #if val is not None: val.drop('bin', axis=1, inplace=True)
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()
    df_stats.dropna(axis=1, inplace=True)
    if val is not None:  df_stats.sort_values('train_val_trend_corr', inplace=True, ascending=False)
    if test is not None:  df_stats.sort_values('train_test_trend_corr', inplace=True, ascending=False)
    return df_stats


def categorical_ttt_dist(train=None, test=None, val=None, features=[], target='target', ncols=3, agg_func='mean', figsize=16, ylim=(), sharey=False):
    '''
    The Categorical Train Test Target Distribution function helps us to understand data distribution better. It can plot train, test, validation, and the target in 
    one graph for each feature.

    Args:
        train (DataFrame): the train dataframe containing features and target columns.
        test (DataFrame): the test dataframe containing features.
        val (DataFrame): the validation dataframe containing features and target columns.
        features (list): Numeric features in the dataframe.
        agg_func (str): The Pandas aggregation functions for the target.
        target (str): The target feature name.
        ncols (int): Column number of the graph.
        figsize (int): Figure size.
        ylim (tuple): Y-limits of the axes.
        sharey (bool): Controls sharing of properties among y (sharey) axes.
    
    Returns:
        DataFrame
    '''

    # Default Parameter Preparations
    if len(features) == 0:
        features = train.columns.tolist()
    
    alpha = 1 / sum(x is not None for x in [train, test, val])

    if len(ylim) == 0:
        np_ylim = np.zeros((len(features), 2))
        for index, feature in enumerate(features):
            train_group = train.groupby(feature).agg({target:agg_func})
            np_ylim[index] = min(train_group[target]), max(train_group[target])
            
            if val is not None:
                val_group = val.groupby(feature).agg({target:agg_func})
                np_ylim[index] = min(np_ylim[index][0], val_group[target].min()), max(np_ylim[index][1], val_group[target].max())
        ymin, ymax = np_ylim[:,0].min(), np_ylim[:,1].max()
        distance = ymax - ymin
        ymin, ymax = ymin-distance/10, ymax+distance/10
    else:
        ymin, ymax = ylim[0], ylim[1]

    # Graph Creation
    nrows = int(len(features) / ncols)
    if len(features) % ncols != 0:
        nrows += 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize, round(nrows*figsize/ncols)), sharey=sharey)
    df_stats = pd.DataFrame(columns=['feature', 'train_nunique', 'test_nunique', 'val_nunique', 'train_test_rmse',
                                    'train_val_rmse', 'val_test_rmse', 'train_val_target_rmse', 
                                    'train_test_same_unique', 'train_val_same_unique', 'val_test_same_unique'])
    for index, (ax,feature) in enumerate(zip(axes.ravel()[:len(features)],features)):        
        # Graph
        train_counts = train[feature].value_counts()
        diff_test = []
        if test is not None: 
            test_counts = test[feature].value_counts()
            diff_test = set(test_counts.index) - set(train_counts.index)
            if val is not None: 
                val_counts = val[feature].value_counts()
                diff_val = set(val_counts.index) - set(train_counts.index)
                diff_test = set(list(diff_test)+list(diff_val))
            for value in diff_test:
                train_counts.loc[value] = 0
            train_counts.sort_index(inplace=True)
        if val is not None and len(diff_test) == 0: 
            val_counts = val[feature].value_counts()
            diff_test = set(val_counts.index) - set(train_counts.index)
            for value in diff_test:
                train_counts.loc[value] = 0
            train_counts.sort_index(inplace=True)
        
        train_group = train.groupby(feature).agg({feature:'count', target:agg_func})
        train_group.columns = [feature+'_count', target]
        train_group[feature+'_count'] /= train.shape[0]
        for value in diff_test:
            train_group.loc[value, :] = [np.nan,np.nan]
        train_group.sort_index(inplace=True)
        
        ax.bar(train_counts.index, train_counts.values, alpha=alpha, label='train', color='tab:cyan')
        if test is not None: ax.bar(test_counts.index, test_counts.values, alpha=alpha, label='test', color='tab:red')
        ax2 = ax.twinx()
        ax2.scatter(train_group.index, train_group[target], label='train target rate', color='tab:purple')
        
        if val is not None:  
            val_group = val.groupby(feature).agg({target:agg_func})
            ax.bar(val_counts.index, val_counts.values, alpha=alpha, label='val', color='tab:gray')
            ax2.scatter(val_group.index, val_group[target], label='val target rate', color='tab:gray')
        ax.set_xlabel(feature)
        ax2.set_ylim([ymin, ymax])
        
        if index == 0: ax2.legend(loc='upper right'), ax.legend(loc='lower left')
        if (index+1) % ncols == 0 or (index+1)==len(features): ax2.set_ylabel('Target Rate')
        if (index+1) % ncols == 1: ax.set_ylabel('Count')

        # Stats
        row_dict = {}
        row_dict['feature'] = feature
        row_dict['train_nunique'] = train[feature].nunique()
        max_rate = max(train.shape[0], _return_shape(test), _return_shape(val))

        if test is not None: 
            diff_test_stats = set(diff_test) - set(test_counts.index)            
            test_group = test.groupby(feature).agg({feature:'count'})
            test_group.columns = [feature+'_count']
            test_group[feature+'_count'] /= test.shape[0]
            for value in diff_test_stats:
                test_group.loc[value] = np.nan
            test_group.sort_index(inplace=True)

            row_dict['test_nunique'] = test[feature].nunique()
            row_dict['train_test_rmse'] = np.sqrt(np.sum((train_group[feature+'_count'] - test_group[feature+'_count'])**2))*max_rate
            row_dict['train_test_same_unique'] = set(train[feature].unique()) == set(test[feature].unique())

            if val is not None: 
                diff_val_stats = set(diff_test) - set(val_counts.index)            
                val_group = val.groupby(feature).agg({feature:'count', target:agg_func})
                val_group.columns = [feature+'_count', target]
                val_group[feature+'_count'] /= val.shape[0]
                for value in diff_val_stats:
                    val_group.loc[value, :] = [np.nan, np.nan]
                val_group.sort_index(inplace=True)

                row_dict['val_nunique'] = val[feature].nunique()
                row_dict['val_test_rmse'] = np.sqrt(np.sum((test_group[feature+'_count'] - val_group[feature+'_count'])**2))*max_rate
                row_dict['val_test_same_unique'] = set(val[feature].unique()) == set(test[feature].unique())
        
        if val is not None: 
            diff_val_stats = set(diff_test) - set(val_counts.index)            
            val_group = val.groupby(feature).agg({feature:'count', target:agg_func})
            val_group.columns = [feature+'_count', target]
            val_group[feature+'_count'] /= val.shape[0]
            for value in diff_val_stats:
                val_group.loc[value, :] = [np.nan, np.nan]
            val_group.sort_index(inplace=True)

            row_dict['val_nunique'] = val[feature].nunique()
            row_dict['train_val_rmse'] = np.sqrt(np.sum((train_group[feature+'_count'] - val_group[feature+'_count'])**2))*max_rate
            row_dict['train_val_target_rmse'] = np.sqrt(np.sum((train_group[target] - val_group[target])**2))
            row_dict['train_val_same_unique'] = set(train[feature].unique()) == set(val[feature].unique())
        df_stats = df_stats.append(row_dict, ignore_index=True)
    
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()
    df_stats.dropna(axis=1, inplace=True)
    if test is not None:  df_stats.sort_values('train_test_rmse', inplace=True)
    if val is not None:  df_stats.sort_values('train_val_target_rmse', inplace=True)
    return df_stats


def _return_min(data, feature):
    try:
        return data[feature].min()
    except:
        return 999999


def _return_max(data, feature):
    try:
        return data[feature].max()
    except:
        return -999999


def _return_shape(data):
    try:
        return data.shape[0]
    except:
        return 0


def _find_trend_changes(data):
    data_diff = data.diff()
    data_diff = data_diff > 0
    counts = 0
    for index in range(1,len(data_diff)-1):
        if data_diff[index] != data_diff[index+1]:
            counts+=1
    return counts