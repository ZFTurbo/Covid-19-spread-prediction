# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime, timedelta
from gbm_classifiers_countries.a0_read_data import contest_metric


def merge_subms():
    sample = pd.read_csv(INPUT_PATH + 'sample_submission_JgJvhOF.csv')
    scountry = pd.read_csv(SUBM_PATH + 'subm_countries.csv')
    srus = pd.read_csv(SUBM_PATH + 'subm_raw_rus_regions.csv')
    s1 = pd.concat((scountry, srus), axis=0)
    unique_dates = sorted(list(s1['date'].unique()))
    start_date, end_date = unique_dates[0], unique_dates[-1]
    print(start_date, end_date)

    s2 = sample[['date', 'region']].merge(s1, on=['date', 'region'], how='left')
    s2.loc[s2['prediction_confirmed'].isna(), 'prediction_confirmed'] = 0
    s2.loc[s2['prediction_deaths'].isna(), 'prediction_deaths'] = 0
    s2['prediction_confirmed'] = np.round(s2['prediction_confirmed'].values).astype(np.int32)
    s2['prediction_deaths'] = np.round(s2['prediction_deaths'].values).astype(np.int32)
    out_path = SUBM_PATH + 'subm_final_{}_{}.csv'.format(start_date, end_date)
    s2.to_csv(SUBM_PATH + 'subm_final_{}_{}.csv'.format(start_date, end_date), index=False)
    return out_path


def update_with_existed_data(path):
    from prepare_features_rus.r3_convert_timeseries_rus import get_russian_regions_names

    s = pd.read_csv(path)

    # Countries part
    if 1:
        fc = pd.read_csv(FEATURES_PATH + 'time_table_flat_confirmed.csv')
        fd = pd.read_csv(FEATURES_PATH + 'time_table_flat_deaths.csv')
        uni_names = fc['name'].unique()
        part_fc = dict()
        part_fd = dict()
        for u in uni_names:
            part_fc[u] = fc[fc['name'] == u].copy()
            part_fd[u] = fd[fd['name'] == u].copy()

        for index, row in s.iterrows():
            # date, region, prediction_confirmed, prediction_deaths
            region = row['region']
            date = row['date']
            date = date.replace('-', '.')
            if region not in part_fc:
                continue
            part1 = part_fc[region][part_fc[region]['date'] == date]
            part2 = part_fd[region][part_fd[region]['date'] == date]
            if len(part1) > 0 and len(part2) > 0:
                print(region, date)
                s.loc[index, 'prediction_confirmed'] = part1['cases'].values[0]
                s.loc[index, 'prediction_deaths'] = part2['cases'].values[0]

    # Russian regions part
    if 1:
        fc = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_confirmed.csv')
        fc1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_latest_confirmed.csv')
        fc = pd.concat((fc, fc1))

        fd = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_deaths.csv')
        fd1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_latest_deaths.csv')
        fd = pd.concat((fd, fd1))

        uni_names = fc['name'].unique()
        iso_names = get_russian_regions_names()
        part_fc = dict()
        part_fd = dict()
        for u in uni_names:
            part_fc[iso_names[u]] = fc[fc['name'] == u].copy()
            part_fd[iso_names[u]] = fd[fd['name'] == u].copy()

        for index, row in s.iterrows():
            # date, region, prediction_confirmed, prediction_deaths
            region = row['region']
            date = row['date']
            date = date.replace('-', '.')
            if region not in part_fc:
                continue
            part1 = part_fc[region][part_fc[region]['date'] == date]
            part2 = part_fd[region][part_fd[region]['date'] == date]
            if len(part1) > 0 and len(part2) > 0:
                print(region, date)
                s.loc[index, 'prediction_confirmed'] = part1['cases'].values[0]
                s.loc[index, 'prediction_deaths'] = part2['cases'].values[0]

    # Get latest non-zero date
    part = s[s['region'] == 'RUS']
    part = part.sort_values('date')[::-1]
    part_dates = part['date'].values
    part_cases = part['prediction_confirmed'].values
    latest_date = ''
    future_dates = set()
    for i in range(len(part_dates)):
        if part_cases[i] > 0:
            latest_date = part_dates[i]
            break
        future_dates |= set([part_dates[i]])
    print('Latest date: {}'.format(latest_date))

    # Find latest value for each region
    latest_values = dict()
    part = s[s['date'] == latest_date]
    for index, row in part.iterrows():
        latest_values[(row['region'])] = (row['prediction_confirmed'], row['prediction_deaths'])

    for index, row in s.iterrows():
        print(index)
        if row['date'] not in future_dates:
            continue
        if row['prediction_confirmed'] > 0 or row['prediction_deaths'] > 0:
            print('Error!')
            exit()
        s.loc[index, ['prediction_confirmed', 'prediction_deaths']] = [latest_values[row['region']][0] + 1, latest_values[row['region']][1] + 1]

    s.to_csv(path[:-4] + '_upd.csv', index=False)
    print('Final result is in: {}'.format(path[:-4] + '_upd.csv'))


if __name__ == '__main__':
    out_path = merge_subms()
    update_with_existed_data(out_path)