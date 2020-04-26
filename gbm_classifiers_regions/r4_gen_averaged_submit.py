# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime, timedelta
from gbm_classifiers_countries.a0_read_data import contest_metric


def ensemble(subms, score):
    preds_confirmed = []
    preds_deaths = []
    weights = []
    shifts = None
    dates = None
    names = None
    for path1, path2, weight in subms:
        weights.append(weight)

        s = pd.read_csv(path1)
        s.sort_values(['name1', 'name2', 'date', 'shift_day'], inplace=True)
        preds_confirmed.append(s['pred'].values)

        s = pd.read_csv(path2)
        s.sort_values(['name1', 'name2', 'date', 'shift_day'], inplace=True)
        preds_deaths.append(s['pred'].values)

        if shifts is None:
            shifts = s['shift_day'].values
            dates = s['date'].values
            names = s['name2'].values
        else:
            if tuple(shifts) != tuple(s['shift_day'].values) or \
                tuple(dates) != tuple(s['date'].values) or \
                tuple(names) != tuple(s['name2'].values):
                print('Error!')
                exit()

    print(len(preds_confirmed), len(preds_deaths))
    preds_confirmed = np.array(preds_confirmed).mean(axis=0)
    preds_deaths = np.array(preds_deaths).mean(axis=0)
    print(preds_confirmed.shape, preds_deaths.shape)

    out = open(SUBM_PATH + 'subm_raw_rus_regions.csv', 'w')
    out.write('date,region,prediction_confirmed,prediction_deaths\n')
    for i in range(len(preds_confirmed)):
        prediction_confirmed, prediction_deaths, date, shift, name = preds_confirmed[i], preds_deaths[i], dates[i], int(shifts[i]), names[i]
        datetime_object = datetime.strptime(date, '%Y.%m.%d')
        datetime_object += timedelta(days=shift)
        if name == 'XXX':
            continue
        out.write("{},{},{},{}\n".format(datetime_object.strftime("%Y-%m-%d"), name, prediction_confirmed, prediction_deaths))
    out.close()

    sample = pd.read_csv(INPUT_PATH + 'sample_submission_dDoEbyO.csv')
    s1 = pd.read_csv(SUBM_PATH + 'subm_raw_rus_regions.csv')
    unique_dates = sorted(s1['date'].unique())
    start_date, end_date = unique_dates[0], unique_dates[-1]

    s2 = sample[['date', 'region']].merge(s1, on=['date', 'region'], how='left')
    s2.loc[s2['prediction_confirmed'].isna(), 'prediction_confirmed'] = 0
    s2.loc[s2['prediction_deaths'].isna(), 'prediction_deaths'] = 0
    s2['prediction_confirmed'] = np.round(s2['prediction_confirmed'].values).astype(np.int32)
    s2['prediction_deaths'] = np.round(s2['prediction_deaths'].values).astype(np.int32)
    s2.to_csv(SUBM_PATH + 'subm_{:.6f}_{}_{}_rus_region.csv'.format(score, start_date, end_date), index=False)


def get_validation_score(subms):
    preds_confirmed = []
    target_confirmed = None
    preds_deaths = []
    target_deaths = None
    weights = []
    shifts = None
    dates = None
    names = None
    for path1, path2, weight in subms:
        print('Go for {}'.format(os.path.basename(path1)[:-9]))
        path1 = path1[:-9] + '_train.csv'
        path2 = path2[:-9] + '_train.csv'
        weights.append(weight)

        s1 = pd.read_csv(path1)
        s1.sort_values(['name1', 'name2', 'date'], inplace=True)
        preds_confirmed.append(s1['pred'].values)
        unique_dates = sorted(s1['date'].unique())
        # print('Unique dates confirmed: {}'.format(unique_dates))
        score1 = contest_metric(s1['target'], s1['pred'])
        if target_confirmed is None:
            target_confirmed = s1['target'].values

        s2 = pd.read_csv(path2)
        s2.sort_values(['name1', 'name2', 'date'], inplace=True)
        preds_deaths.append(s2['pred'].values)
        unique_dates = sorted(s2['date'].unique())
        # print('Unique dates deaths: {}'.format(unique_dates))
        score2 = contest_metric(s2['target'], s2['pred'])
        if target_deaths is None:
            target_deaths = s2['target'].values

        # print(len(s1['pred'].values), len(s2['pred'].values))
        s = pd.concat((s1, s2), axis=0)
        # print(len(s))
        score_total = contest_metric(s['target'], s['pred'])
        print('Score confirmed: {:.6f} Score deaths: {:.6f} Score overall: {:.6f}'.format(score1, score2, score_total))

    preds_confirmed = np.array(preds_confirmed).mean(axis=0)
    preds_deaths = np.array(preds_deaths).mean(axis=0)
    score1 = contest_metric(target_confirmed, preds_confirmed)
    score2 = contest_metric(target_deaths, preds_deaths)
    s = np.concatenate((preds_confirmed, preds_deaths), axis=0)
    target = np.concatenate((target_confirmed, target_deaths))
    score_total = contest_metric(target, s)
    print('Score confirmed: {:.6f} Score deaths: {:.6f} Score overall: {:.6f}'.format(score1, score2, score_total))
    return score_total


if __name__ == '__main__':
    if 1:
        subms = [
            [
                SUBM_PATH + 'XGB_confirmed_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                SUBM_PATH + 'XGB_deaths_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                1,
            ],
        ]
    if 0:
        subms = [
            [
                SUBM_PATH + 'XGB_confirmed_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                SUBM_PATH + 'XGB_deaths_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                1,
            ],
            [
                SUBM_PATH + 'LGBM_confirmed_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                SUBM_PATH + 'LGBM_deaths_LOG_1_DIFF_1_DIV_1_rus_regions_test.csv',
                1,
            ],
        ]
    score = get_validation_score(subms)
    ensemble(subms, score)
