ss = 0.5
mc = 0.8
md = 7
gm = 1
# n_trees = 25

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    # "eval_metric": "merror",
    # "num_class":num_class,
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
