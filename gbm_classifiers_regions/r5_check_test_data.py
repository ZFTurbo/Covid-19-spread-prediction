# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime, timedelta
from gbm_classifiers_countries.a0_read_data import contest_metric
from prepare_features_rus.r3_convert_timeseries_rus import get_russian_regions_names


def replace_name(data):
    reg_names = get_russian_regions_names()
    replaced_name = []
    for nm in data:
        # print(nm)
        if nm in reg_names:
            replaced_name.append(reg_names[nm])
        else:
            replaced_name.append(nm)
    return replaced_name


def get_real_score(input_subm):
    s = pd.read_csv(input_subm)

    conf = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_confirmed.csv')
    conf['region'] = conf['name']
    dates = conf['date']
    dates = [d.replace('.', '-') for d in dates]
    conf['date'] = dates
    print(conf['date'], conf['region'])
    conf['region'] = replace_name(conf['name'].values)

    death = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_deaths.csv')
    death['region'] = death['name']
    dates = death['date']
    dates = [d.replace('.', '-') for d in dates]
    death['date'] = dates
    print(death['date'], death['region'])
    death['region'] = replace_name(death['name'].values)

    print(len(s))
    s = s.merge(conf[['region', 'date', 'cases']], on=['date', 'region'], how='left')
    s['confirmed'] = s['cases']
    s.drop('cases', axis=1, inplace=True)
    s = s.merge(death[['region', 'date', 'cases']], on=['date', 'region'], how='left')
    s['deaths'] = s['cases']
    s.drop('cases', axis=1, inplace=True)

    unique_dates = sorted(s['date'].unique())
    for u in unique_dates:
        part = s[s['date'] == u]
        # part.to_csv(CACHE_PATH + 'debug.csv', index=False)
        score1 = contest_metric(part['confirmed'], part['prediction_confirmed'])
        print('Type: confirmed Day: {} Score: {:.6f}'.format(u, score1))
    for u in unique_dates:
        part = s[s['date'] == u]
        score1 = contest_metric(part['deaths'], part['prediction_deaths'])
        print('Type: deaths Day: {} Score: {:.6f}'.format(u, score1))

    score1 = contest_metric(s['confirmed'], s['prediction_confirmed'])
    score2 = contest_metric(s['deaths'], s['prediction_deaths'])
    preds = np.concatenate((s['prediction_confirmed'], s['prediction_deaths']))
    real = np.concatenate((s['confirmed'], s['deaths']))
    score_total = contest_metric(real, preds)

    print('Score confirmed: {:.6f} Score deaths: {:.6f} Score overall: {:.6f}'.format(score1, score2, score_total))
    return score_total


if __name__ == '__main__':
    input_subm = SUBM_PATH + 'subm_raw_rus_regions.csv'
    score = get_real_score(input_subm)