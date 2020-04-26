# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from gbm_classifiers_countries.a0_read_data import *


SUBM_PATH_DETAILED = SUBM_PATH + 'detailed/'
if not os.path.isdir(SUBM_PATH_DETAILED):
    os.mkdir(SUBM_PATH_DETAILED)


random.seed(2020)
np.random.seed(2020)


def print_importance(features, gbm, prnt=True):
    max_report = 100
    importance_arr = sorted(list(zip(features, gbm.feature_importance())), key=lambda x: x[1], reverse=True)
    s1 = 'Importance TOP {}: '.format(max_report)
    for d in importance_arr[:max_report]:
        s1 += str(d) + ', '
    if prnt:
        print(s1)
    return importance_arr


def create_lightgbm_model(train, features, params, day):
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    target_name = params['target']

    print('LightGBM version: {}'.format(lgb.__version__))
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

    if 0:
        unique_target = np.array(sorted(train[target_name].unique()))
        print('Target length: {}: {}'.format(len(unique_target), unique_target))

    required_iterations = 3
    overall_train_predictions = np.zeros((len(train),), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        # Debug
        num_folds = random.randint(4, 6)
        random_state = 10
        learning_rate = random.choice([0.01, 0.02, 0.03])
        num_leaves = random.choice([32, 48, 64])
        feature_fraction = random.choice([0.8, 0.85, 0.9, 0.95])
        bagging_fraction = random.choice([0.8, 0.85, 0.9, 0.95])

        # objective = random.choice(["rmse", "huber", 'fair', 'poisson'])
        # metric_function = random.choice(["rmse", "quantile", "huber", 'fair', 'poisson'])
        objective = random.choice(["rmse"])
        metric_function = random.choice(["rmse", "quantile"])

        boosting_type = 'gbdt'
        # boosting_type = 'dart'
        min_data_in_leaf = 1
        # max_bin = 511
        bagging_freq = 5
        drop_rate = 0.05
        skip_drop = 0.5
        max_drop = 1

        if 1:
            params_lgb = {
                'task': 'train',
                'boosting_type': boosting_type,
                'objective': objective,
                'metric': {metric_function},
                # 'objective': 'multiclass',
                # 'num_class': 5,
                # 'metric': {'multi_logloss'},
                'device': 'cpu',
                'gpu_device_id': 1,
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'min_data_in_leaf': min_data_in_leaf,
                'bagging_freq': bagging_freq,
                # 'max_bin': max_bin,
                'drop_rate': drop_rate,
                'boost_from_average': True,
                'skip_drop': skip_drop,
                'max_drop': max_drop,
                # 'lambda_l1': 5,
                # 'lambda_l2': 5,
                'feature_fraction_seed': random_state + iter1,
                'bagging_seed': random_state + iter1,
                'data_random_seed': random_state + iter1,
                'verbose': 0,
                'num_threads': 6,
            }
        log_str = 'LightGBM iter {}. PARAMS: {}'.format(iter1, sorted(params_lgb.items()))
        print(log_str)
        num_boost_round = 10000
        early_stopping_rounds = 100

        ret = get_kfold_split_v2(num_folds, train, 821 + iter1)
        full_single_preds = np.zeros((len(train),), dtype=np.float32)
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

            lgb_train = lgb.Dataset(X_train[features].values, y_train)
            lgb_eval = lgb.Dataset(X_valid[features].values, y_valid, reference=lgb_train)

            gbm = lgb.train(params_lgb, lgb_train, num_boost_round=num_boost_round,
                            early_stopping_rounds=early_stopping_rounds, valid_sets=[lgb_eval], verbose_eval=0)

            imp = print_importance(features, gbm, True)
            model_list.append(gbm)

            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            pred = gbm.predict(X_valid[features].values, num_iteration=gbm.best_iteration)
            full_single_preds[valid_index] += pred
            score = contest_metric(y_valid, pred)
            print('Fold {} score: {:.6f}'.format(fold_num, score))

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

    if 0:
        fig, ax = plt.subplots(figsize=(10, 25))
        ax.barh(list(range(min(output_features, len(imp)))), values, 0.4, color='green', align='center')
        ax.set_yticks(list(range(min(output_features, len(imp)))))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        plt.subplots_adjust(left=0.47)
        plt.savefig('debug.png')

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


def predict_with_lightgbm_model(test, features, model_list):
    dtest = test[features].values
    full_preds = []
    total = 0
    for m in model_list:
        total += 1
        print('Process test model: {}'.format(total))
        preds = m.predict(dtest, num_iteration=m.best_iteration)
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

    return preds


if __name__ == '__main__':
    start_time = time.time()
    gbm_type = 'LGBM'
    params = get_params()
    target = params['target']
    id = params['id']
    metric = params['metric']
    LIMIT_DATE = 8
    STEP_BACK = None
    # STEP_BACK = 7

    all_scores = dict()
    alldays_preds_train = dict()
    alldays_preds_test = dict()
    for type in ['confirmed', 'deaths']:
        print('Go for type: {}'.format(type))
        alldays_preds_train[type] = []
        alldays_preds_test[type] = []
        for day in range(1, LIMIT_DATE+1):
            train, test, features = read_input_data(day, type, step_back_days=STEP_BACK)
            print('Features: [{}] {}'.format(len(features), features))

            if 1:
                overall_train_predictions, score, model_list, importance = create_lightgbm_model(train, features, params, day)
                prefix = '{}_day_{}_{}_{}_{:.6f}'.format(gbm_type, day, len(model_list), metric, score)
                save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')
            else:
                prefix = 'XGB_5_auc_0.891512'
                score, model_list, importance, overall_train_predictions = load_from_file(MODELS_PATH + prefix + '.pklz')

            all_scores[(type, day)] = score
            train['pred'] = overall_train_predictions
            train['pred'] = np.maximum(train['pred'], train['case_day_minus_0'])
            train[['name1', 'name2', 'date', 'target', 'pred']].to_csv(SUBM_PATH_DETAILED + prefix + '_train.csv', index=False, float_format='%.8f')
            train_tmp = decrease_table_for_last_date(train)
            alldays_preds_train[type].append(train_tmp[['name1', 'name2', 'date', 'target', 'pred']].copy())

            overall_test_predictions = predict_with_lightgbm_model(test, features, model_list)
            test['pred'] = overall_test_predictions

            # We now that value must be equal or higher
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
        for day in range(1, LIMIT_DATE + 1):
            print('Type: {} Day: {} Score: {:.6f}'.format(type, day, all_scores[(type, day)]))
    for type in ['confirmed', 'deaths']:
        print('Total score {} for all days: {:.6f}'.format(type, all_scores[(type, 'full')]))

    print("Elapsed time overall: {:.2f} seconds".format((time.time() - start_time)))
