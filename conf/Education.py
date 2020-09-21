ss = 0.9
mc = 2
md = 8
gm = 2

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    # "eval_metric": "merror",
    # "num_class": num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 0.8,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
    # 'seed':seed,
}
