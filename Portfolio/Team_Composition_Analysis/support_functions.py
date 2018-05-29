
# coding: utf-8

# In[3]:

# Data structure management
import pandas as pd
import numpy as np
import pickle, sys, json

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Missing Data Tools
import fancyimpute, missingno

# ML Tools
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, silhouette_samples,silhouette_score, accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, scale,maxabs_scale, minmax_scale, Imputer
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering,MiniBatchKMeans,estimate_bandwidth
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import wminkowski
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier


# Inferrence Tools
from scipy import stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Miscellaneous
from functools import partial
from tqdm import tqdm

# Additional Requirements for StabilityValidation class.
from sklearn.metrics import pairwise_distances_argmin


def missing_values_table(df, summarize = None):
    '''
    Returns a dataframe that shows the number and percent of present and missing values
    as features, indexed by the columns of df.
    
    Parameters
    ---------
    df: pandas DataFrame
    
    summarize: str
        Allows for summation of reporting by eliminating columns with all missing values,
        all present values, or both from the returned table.
        ('full', 'empty', or 'both')
    
    '''
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    vals = df.notnull().sum()
    val_percent = 100 * vals / len(df)
    missing_table = pd.concat([missing, missing_percent, vals, val_percent], axis=1)     .rename(columns = {0 : 'missing', 1 : 'missing_percent', 2:'vals',3:'value_percent'})
    
    empty = pd.Series([True for c in df.columns], index=df.columns)
    full = empty.copy()
    
    if summarize in ['empty','both']:
        empty = missing_table[missing_table.vals == 0].index
        print(len(empty),'Columns with no values: ', list(empty))
        empty = missing_table.vals > 0
    if summarize in ['full','both']:
        full = missing_table[missing_table.missing == 0].index
        print('\n',len(full),'columns with all values present: ',list(full))
        full = missing_table.missing > 0

    return missing_table[(empty) & (full)]

def get_feature_scale_dict(df, c, ignore):
    df = df.set_index(c).select_dtypes(exclude = ['category', 'object']).drop(ignore, axis=1, errors='ignore')
    return df.groupby(df.index).agg(['mean', lambda x: x.std(ddof=0)]).to_dict('index')

def patch_scale(data, ref, ignore = None, scale_down = True, verbose = False):
    df = data.copy()
    not_scaled = []
    if ignore:
        cols = df.select_dtypes(exclude = ['category', 'object']).drop(ignore, axis =1, errors = 'ignore').columns
    else:
        cols = df.select_dtypes(exclude = ['category', 'object']).columns
    for c in cols:
        for p in ref:
            try:
                m = ref[p][c,'mean']
                sd = ref[p][c,'<lambda>']
            except:
                continue
            if np.isfinite(m) and sd != 0:
                if scale_down:
                    df.loc[df.patchno == p, c] = (df.loc[df.patchno == p, c] - m)/sd
                else:
                    df.loc[df.patchno == p, c] = (df.loc[df.patchno == p, c] * sd) + m
            else:
                not_scaled.append((p,c))
    if verbose and len(not_scaled) > 0:
        print('WARNING - Unable to transform:',not_scaled)
    return df

def df_rmse(df1,df2,n):
    '''
    Returns the Root Mean Square Error (RMSE) between imputed values and true values.
    
    Parameters
    ---------
    df1, df2: pandas DataFrames
    
    n: int
        Number of imputed values.
    '''
    return ((((df1 - df2)**2).sum().sum())/n)**.5

def add_missing(df,w):
    '''
    Returns a dataframe with missing values added randomly based on 
    
    Parameters
    ---------
    df: pandas DataFrame (containing no missing values).
    
    w: pandas Series
        weights representing % of missing values that should be spiked in
        for each feature in df.
    '''
    if df.shape[1] != len(w):
        return 'OOPS!!!'
    mask = np.empty((0,df.shape[0]),dtype=bool)
    np.random.seed(10)
    for c in df:
        if w[c] == 0:
            mask = np.concatenate((mask, np.zeros((1, df.shape[0]),dtype=bool)),axis=0)
        else:
            mask = np.concatenate((mask, np.random.choice([True, False],size=(1,df.shape[0]), p = [w[c],1-w[c]])),axis=0)
    return df.mask(mask.T)

def test_impute(df,fun, sample_size= 800, iters=20, round_bin=True, ignore_bin=False):
    '''
    Returns a list containing for each iteration the RMSE of imputed and true values between
    a dataframe and a predicted dataframe built by randomly adding missing values to the 
    features of the original dataframe according to the missing percent of each feature and 
    running an imputer function on the missing values.
    
    Parameters
    ---------
    df: pandas DataFrame
    
    fun: str
        The imputer function and its parameters.
    
    sample_size: int
        How many observations in df should be used to build the true and
        missing dataframes.
    
    iters: int
        Number of imputation iterations to perform on different samples of df.
    
    round_bin: boolean
        Whether or not to round the predictions of discrete binary features to 0/1.
    
    ignore_bin: boolean
        Whether or not to ignore binary variables from imputation completely.
    '''
    wts = missing_values_table(df).missing_percent/100
    X = df.dropna()
    b_cols = get_bin_cols(X).drop(['doubles','triples','quadras','pentas'], errors='ignore')
    rmse = []
    for i in range(iters):
        # Get a sample from non-nan section of DF
        X_true = X.sample(sample_size,replace=True, random_state=i)
        
        # Build a fake-df of same size as sample, with NaN's inserted
        # into columns at same rate as they are in the real data
        np.random.seed(i+1)
        X_miss = add_missing(X_true, wts)
        
        if ignore_bin:
            X_miss.drop(b_cols, axis=1, inplace = True, errors='ignore')
            X_true.drop(b_cols, axis=1, inplace = True, errors='ignore')
        
        # Fill nan's with method of choice
        X_fill = eval(fun)(X_miss)
        
        # Round binary categoricals up and down
        if round_bin and not ignore_bin:
            X_fill = pd.DataFrame(X_fill, index=X_true.index, columns=X_true.columns) 
            X_fill[b_cols] = X_fill[b_cols].applymap(lambda x: 0 if x <= .5 else 1)
        
        # Compute RMSE of the missing value imputations, and add to list
        rmse.append(df_rmse(X_true, X_fill, X_miss.isnull().sum().sum()))
    return rmse

def show_test(df_dict, procedure, verbose = True, **kwargs):
    full_results = []
    for k in df_dict:
        results = test_impute(df_dict[k].drop('result',axis=1).select_dtypes(exclude=['category','object']),                              procedure, **kwargs)
        if verbose:
            print(k,'RMSE mean of iteration\'s -',round(np.mean(results),3),'-- SD -', round(np.std(results),3))
        full_results.append(results)
    if verbose:
        print('FULL DATA RMSE -',round(np.mean(full_results),3), '-- SD -', round(np.std(full_results),3))
    return full_results

def keychain(d,r=''):
    for k,v in d.items():
        if isinstance(v, dict):
            yield from keychain(v,r+str(k)+'-')
        else:
            yield str(r+str(k))

def get_vif(df, regressors=None, response='result' , drops = None):
    '''
    df is a dataframe, regressors is an index of columns
    in df, and response is a string representing the outcome column in df.
    '''
    regressors = regressors or df.select_dtypes(exclude=['object','category']).columns
    regressors = regressors.drop(drops, errors = 'ignore') if drops else regressors
    cols = "+".join(regressors.drop(response, errors = 'ignore').tolist())
    outcome, pred = dmatrices(response + ' ~' + cols, df, return_type='dataframe')
    vif = pd.DataFrame()
    vif["Variance Inflation Factor"] = [variance_inflation_factor(pred.values, i) for i in range(pred.shape[1])]
    vif["Regressor"] = pred.columns
    vif.set_index('Regressor', inplace = True)
    return vif.round(3).sort_values(by = 'Variance Inflation Factor', ascending = False )
            
def scale_counts(df, g):
    scales = ['MinMaxScaler', 'MaxAbsScaler', 'RobustScaler','StandardScaler']
    counts = df.loc[g,:].weighted_picks.sort_values(ascending=False).to_frame()

    for scale in scales:
        scaler = eval(scale)()
        counts[scale] = scaler.fit_transform(counts.weighted_picks.values.reshape(-1,1))
    
    print('\n',g[:2])
    return counts.reset_index(level=['patchno','position'], drop=True).round(3).rename(columns={'weighted_picks':'w_pick'})            

def auto_feats(dict_df, thresh = .2, verbose = False):
    if verbose:
        print('-- Number of features with result correlation below {0:.2f} by position --'.format(thresh))
    feats = {}
    for k,df in dict_df.items():
        corr = df.select_dtypes(exclude=['object','category']).corr().result
        if verbose:
            print(k, ':', len(corr[abs(corr) <= thresh]))
        feats[k] = corr[abs(corr) <= thresh].index
    return feats

def tune_kmeans(df, features, name='dataset', n=6, plots = True, win_rates = None, win_thresh = .03, rand = 10):
    
    ks = range(2, n+1)
    inertias = []
    df.dropna(axis=1,how='all',inplace=True)

    # for each number of clusters
    print(name, '\n')
    for k in ks:
        kmc = KMeans(n_clusters = k, random_state=rand)
        kmc.fit(df.loc[:,features])
        inertias.append(kmc.inertia_)

        if win_rates:
            clust_win = get_win_rates(kmc.labels_,df.result)
            cmax, cmin = clust_win['mean'].max(), clust_win['mean'].min()
            if win_rates == 'all':
                print('\n', clust_win)
            elif win_rates == 'even' and             (cmin >= (.5  - win_thresh) or cmax() <= (.5 + win_thresh)):
                print('\n', clust_win)
            elif win_rates == 'uneven' and             (cmin <= (.5 - win_thresh) or cmax >= (.5 + win_thresh)):
                print('\n', clust_win)

    if plots:
    # inertia plot
        plt.plot(ks, inertias, '-o')
        plt.xlabel('clusters')
        plt.ylabel('inertia')
        plt.title(name)
        plt.xticks(ks)
        plt.show()
    return [ks,inertias]

def get_win_rates(labels, results):
    wins_by_c = pd.concat([labels.rename('cluster'), results.rename('result')], axis = 1)
    wins_by_c = wins_by_c.groupby("cluster")['result'].agg([np.mean, 'count'])    .rename(columns={"mean": "win_pct",'count':'players_in_cluster'})
    return wins_by_c

# NOTE: Adapted most of this code from -
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
def sil_plots(data, feats, c_min = 2, c_max = 7, plots = True):
    x_cols = np.matrix(data[feats])
    sil_score = {}
    for n in range(c_min,c_max,2):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 4)
        for j in range(1,3):
            k = n + j - 1
            if k <= c_max:   
                clusters = KMeans(n_clusters=k, random_state=10).fit_predict(x_cols)
                sil_score[k] = silhouette_score(x_cols, clusters)
                
                if plots:
                    sample_silhouette_values = silhouette_samples(x_cols, clusters)
                    eval('ax'+str(j)).set_xlim([-0.25, .75])
                    eval('ax'+str(j)).set_ylim([0, len(x_cols) + (k+1) * 10])
                    y_lower = 10

                    for i in range(k):
                        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
                        ith_cluster_silhouette_values.sort()
                        size_cluster_i = ith_cluster_silhouette_values.shape[0]

                        y_upper = y_lower + size_cluster_i
                        color = cm.spectral(float(i) / k)
                        eval('ax'+str(j)).fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)
                        eval('ax'+str(j)).text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                        y_lower = y_upper + 10

                    eval('ax'+str(j)).axvline(x=sil_score[k], color="red", linestyle="--")

                    eval('ax'+str(j)).set_title("Silhouette for {} clusters".format(k))
                    eval('ax'+str(j)).set_xlabel("")
                    eval('ax'+str(j)).set_ylabel("Cluster Labels")
                    eval('ax'+str(j)).set_yticks([])
                    eval('ax'+str(j)).set_xticks([-.5,-0.25,-.1, 0, 0.25, 0.5, 0.75])
        if plots:
            if (c_max - c_min + 1) % 2 == 1 and n == c_max:
                fig.delaxes(ax2)
            plt.show()
    return sil_score

def unpickle(filename):
    p = open(filename,"rb")
    item = pickle.load(p)
    p.close()
    return item
    
def enpickle(python_object, filename):
    p = open(filename,"wb")
    pickle.dump(python_object, p)
    p.close()
    
def get_bin_cols(df):
    df = df.select_dtypes(exclude=['object','category'])
    bin_cols = df.columns[df.nunique() <= 2]
    return bin_cols
    
def get_cluster_factors(df, spread = 1, c_label = 'cluster', scale_first = False, scale_bin = True):
    '''
    Assumes df already standardized on non-binary features.
    '''
    df = df.select_dtypes(exclude=['object','category'])
    big_factors = []
    bin_cols = None
    
    if scale_first: 
        bin_cols = get_bin_cols(df.drop(c_label, axis=1))
        df.loc[:,bin_cols] = scale(df.loc[:,bin_cols])
    if to_scale:
        cols = df.columns.drop(bin_cols, errors = 'ignore')
        cols = cols.drop(c_label)
        df.loc[:,cols] = scale(df.loc[:,cols])
        
    g = df.groupby('cluster').mean()
    for c in g:
        gap = max(g[c]) - min(g[c])
        if gap >= spread:
            # Includes ugly conversion workaround to python's finicky float rounding
            big_factors +=[(c, float("{0:.3f}".format(gap)))]
    
#     big_factors = [(b[0], float("{0:.3f}".format(b[1]))) for b in big_factors]
    
    return sorted(big_factors, reverse = True, key=lambda x: x[1])

def get_per_cluster_factors(df, spread = 1, c_label = 'cluster'):
    df = df.select_dtypes(exclude=['object','category'])
    sd = df.std()
    groups = df.groupby(c_label)
    for i,d1 in groups:
        big_factors = []
        g1 = d1.mean()
        for j,d2 in groups:
            if i!=j:
                g2 = d2.mean()
                for c in df.columns.drop(c_label):
                    if g1[c] - g2[c] >= spread*sd[c]:
                        big_factors.append((c, float("{0:.3f}".format((g1[c] - g2[c])/sd[c])), 'Above group {}'.format(j)))
        print('Group', i)
        print(sorted(big_factors, reverse = True, key=lambda x: x[1]), '\n')

def show_cluster_factors(df, spread=1, c_label = 'cluster', position = None, include = None, centered = False):
    factor_cols = include if include else df.columns
    p_title = 'Cluster Differences for ' + position if position else 'Cluster Differences'
         
    cols = [c[0] for c in get_cluster_factors(df[factor_cols], spread, c_label)]
    sd = df[cols].std()
    mu = df[cols].mean() if centered else 0
    cols += [c_label]
    
    ax = ((df[cols].groupby(c_label).mean() - mu)/sd).transpose().plot(kind = 'bar', rot = 45, figsize=(20, 10))
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.set_title(p_title)
    ax.title.set_fontsize(20)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')

    plt.show()
    
def top_champs(df, cluster_label, n=5, count_label = 'champion'):
    return df.groupby(cluster_label)[count_label]            .apply(lambda x: round((x.value_counts()/x.size).head(n), 4)*100)
    
def run_hdb(df, feats = None, min_c = 5, min_s = 5, min_span = True, plot = True, get_clf = True):
    if feats:
        df = df[feats]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c, min_samples=min_s,  gen_min_span_tree=min_span, prediction_data=True)
    clf = clusterer.fit(df)
    if plot:
        clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    if get_clf:
        return clf
    
    
    
def get_iqr(df, span = .5, as_series = False):
    if 0 < span < 1:
        if as_series:
            return df.quantile(.5 + span/2) - df.quantile(.5 - span/2)
        else:
            return (df.quantile(.5 + span/2) - df.quantile(.5 - span/2)).values
    else:
        print("ERROR --- 'span' must be between 0 and 1")

def mink_weights(df, smooth=1, span = .5):
    return 1/(get_iqr(df, span)**smooth +1)

def get_opp_labels(s):
    a = s.groupby(level=0).shift(1).fillna(0).astype(int)
    b = s.groupby(level=0).shift(-1).fillna(0).astype(int)
    return (a + b).astype(str)

def get_label_df(data, feats, clf, result, opp_labels = True,                 dummies = True, win_divergence = True, ref_data = False, bgm = False,
                 ms = False, multi_kmeans_group = None, hd_knn = False, mink_kwargs = {}):
    
    trial_df = result.copy()
    win_div = []
    if ms:
        bw_quantile = clf.bandwidth
    
    if ref_data:
        trial_df.reset_index(inplace = True)
        trial_df.set_index(['patchno', 'champion'], inplace = True)
    
    for k,df in data.items():
        if ms:
            est_bw_samples = int(df.shape[0]/2) if not ref_data else df.shape[0]
            bw = estimate_bandwidth(df[feats[k]].as_matrix(), quantile=bw_quantile,                                    n_samples=est_bw_samples, random_state=7)
            clf.bandwidth = bw
        if multi_kmeans_group:
            clf.n_clusters = multi_kmeans_group[k]
            
        if bgm:
            clusters = get_bgm_labels(df[feats[k]],clf)
        elif hd_knn:
            clusters = hdbscan_with_knn(df[feats[k]], clf, mink_kwargs = mink_kwargs)
        else:
            clusters = clf.fit(df[feats[k]]).labels_
        trial_df[k] = clusters.astype(str) if hd_knn else pd.Series(clusters,index=df.index).rename(k).astype(str)
        
        if win_divergence:
            win_rates = get_win_rates(trial_df[k], trial_df['result'])
            k_win_div = (                              np.abs(win_rates.win_pct - .5)*                              win_rates.players_in_cluster                              /win_rates.players_in_cluster.sum()                             ).sum()
            win_div.append(k_win_div)
            
        if opp_labels:
            trial_df['opp_' + k] = get_opp_labels(trial_df[k])
            
    if ref_data:
        trial_df.reset_index(inplace=True)
        trial_df.set_index(['gameid', 'team'], inplace = True)
    trial_df.drop(['patchno','champion'], axis=1, inplace = True)
        
    if dummies and win_divergence:
        return pd.get_dummies(trial_df), win_div
    elif dummies:
        return pd.get_dummies(trial_df)
    elif win_divergence:
        return trial_df, win_div
    else:
        return trial_df

def score_clusters(df, clf, verbose = True, results = True):
    X, X_test, y, y_test = train_test_split(df.drop('result', axis=1), df.result,test_size=.3, random_state = 13)
    scores = {'train':{},'test':{}}
    
    clf.fit(X, y)
    prob_train = clf.predict_proba(X)
    prob_test = clf.predict_proba(X_test)
    
    scores['accuracy_train'] = clf.score(X,y)
    scores['accuracy_test'] = clf.score(X_test,y_test)
    scores['AUC_train'] = roc_auc_score(y, prob_train[:,1])
    scores['AUC_test'] = roc_auc_score(y_test, prob_test[:,1])
    scores['log_loss_train'] = log_loss(y, prob_train)
    scores['log_loss_test'] = log_loss(y_test, prob_test)
    if verbose:
        print("\t  Training --\tTesting")
        print("Accuracy: {:0.3f}    --\t{:0.3f}".format(scores['accuracy_train'],scores['accuracy_test']))
        print("AUC:\t  {:0.3f}    --\t{:0.3f}".format(scores['AUC_train'],scores['AUC_test']))
        print("Log-Loss: {:0.3f}    --\t{:0.3f}".format(scores['log_loss_train'],scores['log_loss_test']))
    if results:
        return scores

def keras_classifier(unit1 = 150, unit2 = 50, n_feats = None):
    model = Sequential([
        Dense(unit1, activation = 'relu', input_shape=(n_feats,)),
        Dense(unit2, activation = 'relu'),
        Dense(1, activation = 'relu')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_estimators(num_feats, **kwargs):
    tr = DecisionTreeClassifier(max_depth=int(num_feats/4), **kwargs)
    rf = RandomForestClassifier(max_features= int(num_feats**.5), max_depth= 5, **kwargs)
    kc = KerasClassifier(build_fn=keras_classifier, verbose=0, n_feats = num_feats, epochs = 10, **kwargs)
    return (('tr',tr),('rf',rf),('kc',kc))

def test_keys():
    return {'Trim_features': {}, 'Auto_features_1': {}, 'Auto_features_2': {}, 'ref_data': {}, 'weighted_ref_data':{}}

def get_bgm_labels(df, clf, bgm_thresh = .1):
    clf2 = clf
    w = clf.fit(df).weights_
    clf2.n_components = max(2, sum(w > bgm_thresh))
    return clf2.fit(df).predict(df)

def hdbscan_with_knn(data, clf, thresh = None, mink_p = 1.5, mink_kwargs = None):
    df = data.copy()
    mc = clf.min_cluster_size
    ms = clf.min_samples
    metric = clf.metric
    clf_method = clf.cluster_selection_method
    
    try:
    # run hdbscan    
        if metric == 'wminkowski':
            mw = mink_weights(df, **mink_kwargs)
            metric = lambda x, y: wminkowski(x,y, p = mink_p, w = mw) 

        clusterer = HDBSCAN(min_cluster_size=mc, min_samples=ms, prediction_data = True,
                    metric = metric, cluster_selection_method=clf_method).fit(df)

        thresh = thresh if thresh else 1/max(2,len(clusterer.exemplars_))

    # get exemplars and labels
        exemplars = np.concatenate([e for e in clusterer.exemplars_])
        labels = np.concatenate([np.full((len(e)),fill_value=i)                                 for i,e in enumerate(clusterer.exemplars_)])

    # fit knn on exemplars
        knn = KNeighborsClassifier(n_neighbors=1).fit(exemplars, labels)

    # map top soft cluster probabilities to obs
        probs = np.max(all_points_membership_vectors(clusterer),axis = 1)
        df['top_prob'] = pd.Series(probs, index = df.index)

    # assign all points to outlier class (label:-1)
        df['label'] = -1

    # take all points above a prob threshhold
        obs = df.top_prob >= thresh

    # predict labels from fitted knn
        df.loc[obs,'label'] =             knn.predict(df.loc[obs, df.columns.drop(['top_prob','label'])])
    except:
        df['label'] = 0
        return df.label
    
#----------------------- TO-DO -----------------------------
# allow batch prediction
# -- 1. assign points below thresh to outlier class
# -- 2. take top n% of obs by cluster prob and predict label
# -- 3. refit knn on assigned points
# -- 4. repeat steps 2 & 3 for remaining percentage bins

# allow for custom distance metrics and weight in hdbscan call
    return df.label

def read_clust_tests_encoded(test_scores):
    test_scores = test_scores if type(test_scores) is dict else unpickle(test_scores)
    enc_clusts = []
    for k, tests in test_scores.items():
        accs = []
        for clf in tests:
            accs.append((clf, tests[clf]['accuracy_test']))
        acc = max(accs, key = lambda x: x[1])
        enc_clusts.append(('{}_{}'.format(k,acc[0]),acc[1]))
    
    if len(enc_clusts) == 0:
        method, accuracy = [],[]
    else:
        method, accuracy = zip(*enc_clusts)
    return pd.DataFrame({'method':method, 'accuracy':accuracy})

def read_clust_test_dicts(val_tests, thresh = .025):
    val_tests = val_tests if type(val_tests) is dict else unpickle(val_tests)
    fair_clusts = []
    for k,d in val_tests.items():
        for f,tests in d.items():
            if 'win_div' not in tests: continue
            if np.mean(tests['win_div']) <= thresh:
                accs = []
                for clf in tests:
                    if clf != 'win_div':
                        accs.append((clf, tests[clf]['accuracy_test']))
                acc = max(accs, key = lambda x: x[1])
                fair_clusts.append(('{}_{}_{}'.format(k,f,acc[0]),acc[1], np.mean(tests['win_div'])))
                
    if len(fair_clusts) == 0:
        method, accuracy, win_div = [],[],[]
    else:
        method, accuracy, win_div = zip(*fair_clusts)
    return pd.DataFrame({'method':method, 'accuracy':accuracy, 'win_div':win_div})
