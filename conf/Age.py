ss = 0.5
mc = 3
md = 7
gm = 2
# n_trees = 37

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    # "eval_metric": "merror",
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 1,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
    # 'seed':seed,
}
