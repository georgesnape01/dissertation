def ModelSelection_Summary(model):

# pr curve

    scores = model[0]
    preds = model[1]
    actuals = model[2]

    print('Average accuracy score: {0}'.format(np.average(scores)))

    prec, recall, _ = metrics.precision_recall_curve(actuals, preds)
    print('AUPRC score: {0}\n'.format(metrics.auc(recall, prec)))

    # plot pr curve
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(recall, prec, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

# confusion matrix

    # probabilities to binary
    threshold = 0.5
    binary_preds = [1 if pred >= threshold else 0 for pred in preds]

    # plot confusion matrix
    conf_matrix = confusion_matrix(actuals, binary_preds)

    plt.subplot(121)
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Purples')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def XGBoost_ModelSelection(df, target, selector, param_grid, k=5):

    # dataset
    sample = df.dropna(subset = [target])
    EndDesc_cols = [col for col in df.columns if 'EndDesc' in col] #EndDescShort_cols = [col for col in df.columns if 'EndDescShort' in col] # none
    X = sample.drop(['ReliableChangeDesc', 'ReliableRecovery', 'Recovery'] + EndDesc_cols, axis = 1)
    y = sample[target]
    cols = X.columns

    # machine learning algorithm
    classifier = XGBClassifier()

    # feature selection method
    if selector == 'SelectFromModel':
        selector = SelectFromModel(classifier)
    elif selector == 'RFE':
        selector = RFE(classifier)
    else:
        # fill this #
        raise ValueError('Unsupported selector type')

    # pipeline
    pipeline = Pipeline([("FS", selector), ("classifier", classifier)])

    # initialise lists
    scores, preds, actuals = [], [], []

    # cross validation and hyperparameter tuning
    outer_cv = StratifiedKFold(n_splits = k, shuffle = True)
    inner_cv = StratifiedKFold(n_splits = k, shuffle = True) # both set to k for now
    for train_index, test_index in outer_cv.split(X, y):

        # outer CV train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # hyper-parameter tuning
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy', verbose=0, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("Inner CV accuracy: {}".format(grid_search.best_score_)) # validation sets

        # optimal model
        estimator = grid_search.best_estimator_

        # count features selected
        support = estimator.named_steps['FS'].get_support()
        num_feat = np.sum(support)
        print("Number of selected features {0}".format(num_feat))

        # features selected
        col_index = np.where(support)[0]
        col_names = [cols[col] for col in col_index]
        print("Selected features {0}".format(col_names))

        # hyperparameters selected
        print("Max depth {0}".format(estimator.named_steps["classifier"].max_depth))
        print("Number of trees {0}".format(estimator.named_steps["classifier"].n_estimators))
        print("Learning rate {0}".format(estimator.named_steps["classifier"].learning_rate))
        print("Minimum child weight {0}".format(estimator.named_steps["classifier"].min_child_weight))
        print("Subsample {0}".format(estimator.named_steps["classifier"].subsample))
        print("Colsample bytree {0}".format(estimator.named_steps["classifier"].colsample_bytree))
        print("Gamma {0}".format(estimator.named_steps["classifier"].gamma))
        print("Lambda {0}".format(estimator.named_steps["classifier"].reg_lambda))
        print("Alpha {0}".format(estimator.named_steps["classifier"].reg_alpha))

        # evaluating optimised model on test
        predictions = estimator.predict(X_test)
        score = metrics.accuracy_score(y_test, predictions)
        scores.append(score)
        print('Outer CV accuracy: {}'.format(score)) # test sets

        print("--------------------------------------------------")

        probs = estimator.predict_proba(X_test)[:, 1]
        preds.extend(probs)
        actuals.extend(y_test)

    return scores, preds, actuals
