import xgboost as xgb
import numpy as np
import sklearn.datasets


def clf_acc(clf, xtr, ytr, xte, yte):
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    return (yte == pred).mean()


def xgb_parse_params(kwargs):
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


def unpack_values(d):
    for elem, val in d.items():
        d[elem] = val[0]
    return d


def read_digits():
    digits = sklearn.datasets.load_digits()
    x = digits.data
    y = digits.target
    return x, y


def read_digits_tr_test_split(ntrain=100, seed=0):
    if seed is not None:
        exit_seed = np.random.randint(1000000000)
        np.random.seed(seed)

    x, y = read_digits()
    order = np.random.permutation(y.size)
    tr = order[:ntrain]
    te = order[ntrain:]
    ytr = y[tr]
    xtr = x[tr, :]
    yte = y[te]
    xte = x[te, :]

    if seed is not None:
        np.random.seed(exit_seed)
    return xtr, ytr, xte, yte


n_jobs = 1
ntrain = 100
xtr, ytr, xval, yval = read_digits_tr_test_split(ntrain=ntrain)
def obj_func(kwargs, verbose=1, n_jobs=n_jobs):
    # global xtr, ytr, xval, yval
    kwargs = xgb_parse_params(kwargs)
    print 'parsed params', kwargs
    obj = -clf_acc(xgb.sklearn.XGBClassifier(nthread=n_jobs, **kwargs), xtr, ytr, xval, yval)
    if verbose > 0:
        print obj, kwargs
    return obj


def main(job_id, params):
    print 'job_id', job_id
    params = unpack_values(params)
    print params
    obj = obj_func(params)
    return obj
