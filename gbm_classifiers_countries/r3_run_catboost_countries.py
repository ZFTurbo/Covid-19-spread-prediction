# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from gbm_classifiers_countries.a0_read_data import *


def get_importance(gbm, data, features):
    importance = gbm.get_feature_importance(data, thread_count=-1, fstr_type='FeatureImportance')
    imp = dict()
    for i, f in enumerate(features):
        imp[f] = importance[i]
    res = sort_dict_by_values(imp)
    return res


def create_catboost_model(train, features, params, day):
    import catboost as catb
    import matplotlib.pyplot as plt
    print('Catboost version: {}'.format(catb.__version__))
    target_name = params['target']
    start_time = time.time()
    if USE_LOG:
        train[target_name] = np.log10(train[target_name] + 1)
    if USE_DIFF:
        if USE_LOG:
            if USE_DIV:
                train[target_name] /= (np.log10(train['case_day_minus_0'] + 1) + 1)
            else:
                train[target_name] -= np.log10(train['case_day_minus_0'] + 1)
        else:
            if USE_DIV:
                train[target_name] /= (train['case_day_minus_0'] + 1)
            else:
                train[target_name] -= (train['case_day_minus_0'] + 1)

    train[target_name] = train[target_name]

    required_iterations = 1
    seed = 1921
    overall_train_predictions = np.zeros((len(train), ), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        num_folds = random.randint(4, 5)
        learning_rate = random.choice([0.01, 0.03, 0.05])
        depth = random.choice([4, 5, 6])
        subsample = random.choice([0.8, 0.9, 0.95])
        colsample_bylevel = random.choice([0.8, 0.9, 0.95])

        ret = get_kfold_split_v2(num_folds, train, seed + iter1)
        full_single_preds = np.zeros((len(train), ), dtype=np.float32)
        fold_num = 0
        for train_index, valid_index in ret:
            fold_num += 1
            print('Start fold {}'.format(fold_num))
            X_train = train.loc[train_index].copy()
            X_valid = train.loc[valid_index].copy()
            y_train = X_train[target_name]
            y_valid = X_valid[target_name]

            print('Train data:', X_train.shape)
            print('Valid data:', X_valid.shape)

            # v1 (don't support GPU)
            if 1:
                early_stop = 100
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=10000,
                    learning_rate=learning_rate,
                    depth=depth,
                    bootstrap_type='Bayesian',
                    task_type='GPU',
                    devices='0',
                    # subsample=subsample,
                    # colsample_bylevel=colsample_bylevel,
                    metric_period=1,
                    od_type='Iter',
                    od_wait=early_stop,
                    random_seed=17,
                    l2_leaf_reg=3,
                    allow_writing_files=False
                )

            if 0:
                early_stop = 100
                model = CatBoostClassifier(
                    loss_function="MultiClass",
                    eval_metric="MultiClass",
                    # task_type='CPU',
                    task_type='GPU',
                    devices='1:2:3',
                    iterations=10000,
                    early_stopping_rounds=early_stop,
                    learning_rate=learning_rate,
                    depth=depth,
                    random_seed=17,
                    l2_leaf_reg=10,
                    allow_writing_files=False
                )

            if 1:
                cat_features_names = [
                    'country'
                ]
                cat_features = []
                for cfn in cat_features_names:
                    cat_features.append(features.index(cfn))

                dtrain = Pool(X_train[features].values, label=y_train, cat_features=cat_features)
                dvalid = Pool(X_valid[features].values, label=y_valid, cat_features=cat_features)
            else:
                dtrain = Pool(X_train[features].values, label=y_train)
                dvalid = Pool(X_valid[features].values, label=y_valid)

            gbm = model.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=100)
            model_list.append(gbm)

            imp = get_importance(gbm, dvalid, features)
            print('Importance: {}'.format(imp[:20]))
            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            print('Best iter: {}'.format(gbm.get_best_iteration()))
            pred = gbm.predict(X_valid[features].values)
            print(pred.shape)
            full_single_preds[valid_index] += pred.copy()
            try:
                score = contest_metric(y_valid, pred)
                print('Fold {} score: {:.6f}'.format(fold_num, score))
            except Exception as e:
                print('Error:', e)

        print(len(train[target_name].values), len(full_single_preds))

        train_tmp = train.copy()
        train_tmp['pred'] = full_single_preds
        train_tmp = decrease_table_for_last_date(train_tmp)

        score = contest_metric(train_tmp[target_name].values, train_tmp['pred'].values)
        overall_train_predictions += full_single_preds
        print('Score iter {}: {:.6f} Time: {:.2f} sec'.format(iter1, score, time.time() - start_time))

    overall_train_predictions /= required_iterations
    for el in overall_importance:
        overall_importance[el] /= required_iterations
    imp = sort_dict_by_values(overall_importance)
    names = []
    values = []
    print('Total importance count: {}'.format(len(imp)))
    output_features = 100
    for i in range(min(output_features, len(imp))):
        print('{}: {:.6f}'.format(imp[i][0], imp[i][1]))
        names.append(imp[i][0])
        values.append(imp[i][1])

    if USE_DIFF:
        if USE_LOG:
            if USE_DIV:
                train[target_name] *= (np.log10(train['case_day_minus_0'] + 1) + 1)
                overall_train_predictions *= (np.log10(train['case_day_minus_0'] + 1) + 1)
            else:
                train[target_name] += np.log10(train['case_day_minus_0'] + 1)
                overall_train_predictions += np.log10(train['case_day_minus_0'] + 1)
        else:
            if USE_DIV:
                train[target_name] *= (train['case_day_minus_0'] + 1)
                overall_train_predictions *= (train['case_day_minus_0'] + 1)
            else:
                train[target_name] += (train['case_day_minus_0'] + 1)
                overall_train_predictions += (train['case_day_minus_0'] + 1)

    if USE_LOG:
        train[target_name] = np.power(10, train[target_name]) - 1
        overall_train_predictions = np.power(10, overall_train_predictions) - 1

    overall_train_predictions[overall_train_predictions < 0] = 0
    train_tmp = train.copy()
    train_tmp['pred'] = overall_train_predictions

    # We now that value must be equal or higher
    train_tmp['pred'] = np.maximum(train_tmp['pred'], train_tmp['case_day_minus_0'])

    train_tmp = decrease_table_for_last_date(train_tmp)
    score = contest_metric(train[target_name].values, overall_train_predictions)
    print('Total score day {} full: {:.6f}'.format(day, score))
    score = contest_metric(train_tmp[target_name].values, train_tmp['pred'].values)
    print('Total score day {} last date only: {:.6f}'.format(day, score))

    return overall_train_predictions, score, model_list, imp


def predict_with_catboost_model(test, features, model_list):
    full_preds = []
    for m in model_list:
        preds = m.predict(test[features].values)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)

    if USE_DIFF:
        if USE_LOG:
            if USE_DIV:
                preds *= (np.log10(test['case_day_minus_0'] + 1) + 1)
            else:
                preds += np.log10(test['case_day_minus_0'] + 1)
        else:
            if USE_DIV:
                preds *= (test['case_day_minus_0'] + 1)
            else:
                preds += (test['case_day_minus_0'] + 1)

    if USE_LOG:
        preds = np.power(10, preds) - 1

    preds[preds < 0] = 0

    return preds


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'CatB'
    params = get_params()
    target = params['target']
    id = params['id']
    metric = params['metric']
    limit_date = DAYS_TO_PREDICT

    all_scores = dict()
    alldays_preds_train = dict()
    alldays_preds_test = dict()
    for type in ['confirmed', 'deaths']:
        print('Go for type: {}'.format(type))
        alldays_preds_train[type] = []
        alldays_preds_test[type] = []
        for day in range(1, limit_date+1):
            train, test, features = read_input_data(day, type, step_back_days=STEP_BACK)
            print('Features: [{}] {}'.format(len(features), features))
            print('Test date: {}'.format(sorted(test['date'].unique())))

            if 0:
                train[features].to_csv(CACHE_PATH + 'train_debug_{}_{}.csv'.format(type, day))
                test[features].to_csv(CACHE_PATH + 'test_debug_{}_{}.csv'.format(type, day))

            overall_train_predictions, score, model_list, importance = create_catboost_model(train, features, params, day)
            prefix = '{}_day_{}_{}_{:.6f}'.format(gbm_type, day, len(model_list), score)
            save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')

            all_scores[(type, day)] = score
            train['pred'] = overall_train_predictions
            train['pred'] = np.maximum(train['pred'], train['case_day_minus_0'])
            train[['name1', 'name2', 'date', 'target', 'pred']].to_csv(SUBM_PATH_DETAILED + prefix + '_train.csv', index=False, float_format='%.8f')
            train_tmp = decrease_table_for_last_date(train)
            alldays_preds_train[type].append(train_tmp[['name1', 'name2', 'date', 'target', 'pred']].copy())

            overall_test_predictions = predict_with_catboost_model(test, features, model_list)
            test['pred'] = overall_test_predictions

            # We now that value must be equal or higher
            count_less = (test['pred'] < test['case_day_minus_0']).astype(np.int32).sum()
            if count_less > 0:
                print('Values less than needed for test: {} ({:.4f} %)'.format(count_less, 100 * count_less / len(test)))
            test['pred'] = np.maximum(test['pred'], test['case_day_minus_0'])

            test['shift_day'] = day
            test[['name1', 'name2', 'date', 'pred']].to_csv(SUBM_PATH_DETAILED + prefix + '_test.csv', index=False, float_format='%.8f')
            alldays_preds_test[type].append(test[['name1', 'name2', 'date', 'shift_day', 'pred']].copy())

        train = pd.concat(alldays_preds_train[type], axis=0)
        score = contest_metric(train['target'].values, train['pred'].values)
        all_scores[(type, 'full')] = score
        print('Total score {} for all days: {:.6f}'.format(type, score))
        prefix = '{}_{}_all_days_{}_{:.6f}'.format(gbm_type, type, len(model_list), score)
        train.to_csv(SUBM_PATH + '{}_train.csv'.format(prefix), index=False)
        test = pd.concat(alldays_preds_test[type], axis=0)
        test.to_csv(SUBM_PATH + '{}_test.csv'.format(prefix), index=False)

        # Needed for ensemble
        prefix_2 = '{}_{}_LOG_{}_DIFF_{}_DIV_{}_countries'.format(gbm_type, type, USE_LOG, USE_DIFF, USE_DIV)
        train.to_csv(SUBM_PATH + '{}_train.csv'.format(prefix_2), index=False)
        test.to_csv(SUBM_PATH + '{}_test.csv'.format(prefix_2), index=False)

    for type in ['confirmed', 'deaths']:
        for day in range(1, limit_date + 1):
            print('Type: {} Day: {} Score: {:.6f}'.format(type, day, all_scores[(type, day)]))
    for type in ['confirmed', 'deaths']:
        print('Total score {} for all days: {:.6f}'.format(type, all_scores[(type, 'full')]))

    print("Elapsed time overall: {:.2f} seconds".format((time.time() - start_time)))