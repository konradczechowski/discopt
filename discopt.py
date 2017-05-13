import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
import copy
import xgboost as xgb
import scipy
from scipy import stats

def cut_space(space, scores, ignore=0.5):
    # overwrites values in space!
    # cuts from each dimension part not used in relatively good parameters combinations
    # much place for improvement here
    names = space.keys()
    good_scores = scores.sort_values('obj')[:int(scores.shape[0] * (1-ignore))]
    for name in names:#dist in space_scipy.items():
        dist = space[name]
        min_val = good_scores[name].min()
        max_val = good_scores[name].max()
        # check type of variable
        if isinstance(dist.dist,stats._continuous_distns.uniform_gen):
            new_dist = scipy_uniform(min_val, max_val)
        elif isinstance(dist.dist,stats._discrete_distns.randint_gen):
            new_dist = stats.randint(min_val, max_val + 1)
        else:
            raise ValueError('type of distribution is unknown: {}'.format(type(dist)))
        # overwrite distribution
        space[name] = new_dist


def sample_from_space(space,size=1):
    sample = pd.DataFrame(columns = space.keys(), index=range(size))
    for key, distr in space.items():
        sample[key] = distr.rvs(size=size)
    return sample


def sample_positive_class(clf, space, params_order, scores, verbose=False):
    sample_size = 100
    for i in range(4):
        sample = sample_from_space(space=space,size=sample_size)
        pred = clf.predict_proba(sample[params_order])[:,1]
        positive = pred>0.5
        if verbose: print('positive freq', positive.mean())
        if positive.sum() == 0:
            sample_size *= 2
            if verbose: print('not found positive example, increasing sample size to {}'.format(sample_size))
        else:
            ret = sample.loc[positive,:].iloc[0,:]
            return ret
    if verbose: print('not found positive sample, the best value is chosen, space will be cut')
    cut_space(space, scores)
    return sample.iloc[pred.argmax(),:]


def find_candidate(scores, space, params_order, new=0.2, best=0.5, worse=0.5, clf = ExtraTreesClassifier(n_estimators=100, n_jobs=4)):
    
    # prepare train sample sorted by objective, get all observations which are new or best
    scores = scores.copy()
    n = scores.shape[0]
    nnew = int(new*n)
    if nnew >0:
        new_params = scores.iloc[-nnew:,:].index
    else:
        new_params = scores.index[:0] # empty index
    scores_sorted = scores.sort_values(by='obj', ascending=False)
    nbest = max(2, int(best*n)) # at least 2 examples are needed for training classifier
    best_params = scores_sorted.iloc[-nbest:,:].index
    train = scores_sorted.loc[scores_sorted.index.isin(list(best_params)+list(new_params)),:].copy()
    
    assert train.shape[0] >= 2
    print('best objective yet {}'.format(train.obj.iloc[-1]))
    #print 'worse', train.obj.iloc[0],'middle', train.obj.iloc[train.shape[0]/2], 'best', train.obj.iloc[-1]
    del train['obj']
    target = pd.Series(1,index=train.index)
    target[:int(target.size * worse)] = 0
    assert target.nunique() > 1
    clf.fit(train[params_order], target)
    return sample_positive_class(clf, space,params_order, scores)


def search_min(obj_func, space, nrep=100,ninit=5, new=1., best=1., worse=0.7, n_jobs=4, verbose=False):
    space = copy.deepcopy(space)
    params_order = sorted(space.keys())
    # space may be changed inside of
    print('running random initial parameters combinations')
    scores = sample_from_space(space=space, size=ninit)
    scores['obj'] = 0
    assert scores.iloc[:,-1].name == 'obj'
    # init with random trials
    for ix in range(scores.shape[0]):
        scores.iloc[ix,-1] = obj_func(dict(scores.iloc[ix,:-1]))
    # search
    for i in range(nrep):
        if verbose: print(i)
        candidate = find_candidate(scores, space, params_order, new=new, best=best, worse=worse, 
                                   clf=ExtraTreesClassifier(n_estimators=100, n_jobs=n_jobs))
        candidate['obj'] = obj_func(dict(candidate))
        if verbose: print('(negative) new_obj', -candidate['obj'], dict(candidate), '\n')
        scores = scores.append(candidate,ignore_index=True)
    return scores


def scipy_uniform(min, max):
    return scipy.stats.uniform(min, max - min)

# the subsample, and colsamples will get overwriten
DEFAULT_SPACE_SCIPY = {
    'max_depth': stats.randint(1, 21),
    'lr_trees_ratio': scipy_uniform(2,10),
    'n_estimators': stats.randint(50, 301),
    'log_gamma': scipy_uniform(np.log(0.01), np.log(10)),
    'log_reg_lambda': scipy_uniform(np.log(0.01), np.log(10)),
    'subsample': 'data_dependant',#scipy_uniform(0.2, 1),
    'colsample_bylevel': 'data_dependant',#scipy_uniform(0.2, 1),
    'colsample_bytree': 'data_dependant',#scipy_uniform(0.2, 1),
}


def xgb_parse_params(kwargs):
    # parse transformed xgb arguments to usual xgb arguments
    kwargs = copy.deepcopy(kwargs)
    if 'max_depth' in kwargs:
        kwargs['max_depth'] = int(kwargs['max_depth'])

    if ('lr_trees_ratio' in kwargs) and ('n_estimators' in kwargs):
        kwargs['learning_rate'] = kwargs['lr_trees_ratio'] / kwargs['n_estimators']
        del kwargs['lr_trees_ratio']

    if 'n_estimators' in kwargs:
        kwargs['n_estimators'] = int(kwargs['n_estimators'])

    for name in kwargs:
        if name.startswith('log_'):
            new_name = name[4:]
            kwargs[new_name] = np.exp(kwargs[name])
            del kwargs[name]
    return kwargs


def min_sample_required(train_data_shape):
    eps = 1e-3
    nrow, ncol = train_data_shape
    inv_col = 1. / ncol
    cs_min = inv_col ** 0.5 + eps
    inv_row = 1. / nrow
    ss_min = inv_row + eps
    return cs_min, ss_min


def prepare_xgb_space(train_data_shape, min_sampling_based_on_shape=True):
    space = copy.deepcopy(DEFAULT_SPACE_SCIPY)
    if min_sampling_based_on_shape:
        # Add sampling parameters to space.
        # Assure that number of sampled columns and rows for tree will be >= 1
        # (otherwise XGB rises error)
        cs_min, ss_min = min_sample_required(train_data_shape)
        space['colsample_bylevel'] = scipy_uniform(cs_min, 1)
        space['colsample_bytree'] = scipy_uniform(cs_min, 1)
        ss_min = max(0.2, ss_min) # TODO shouldn't row subsample data dependency be removed at all?
        space['subsample'] = scipy_uniform(ss_min, 1)
    return space


def prepare_xgb_obj_func(train_valid_func, n_jobs):
    # create wrapper for train_valid_func
    def obj_func(transformed_xgb_params):
        xgb_params = xgb_parse_params(transformed_xgb_params)
        clf = xgb.sklearn.XGBClassifier(nthread=n_jobs,**xgb_params)
        obj_value = train_valid_func(clf)
        return obj_value
    return obj_func


def prepare_params_order(space):
    return sorted(space.keys())


def get_best_params(scores):
    best_params = copy.deepcopy(scores.iloc[scores.obj.argmin()])
    del best_params['obj']
    return xgb_parse_params(dict(best_params))


def search_params_for_xgb(train_valid_func, train_data_shape, ntrials=200, n_jobs=1, verbose=False, **discopt_kwargs):
    space = prepare_xgb_space(train_data_shape)
    obj_func = prepare_xgb_obj_func(train_valid_func, n_jobs)

    scores = search_min(obj_func, space,
                        nrep=ntrials, n_jobs=n_jobs, verbose=verbose,
                        **discopt_kwargs
                        )
    best_params = get_best_params(scores)

    return best_params, scores
