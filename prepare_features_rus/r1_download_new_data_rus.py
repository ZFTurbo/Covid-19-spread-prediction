# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *


def update_current_data():
    store_file1 = INPUT_PATH + 'time_series_covid19_confirmed_RU.csv'
    store_file2 = INPUT_PATH + 'time_series_covid19_deaths_RU.csv'

    try:
        os.remove(store_file1)
        os.remove(store_file2)
    except:
        print('Already removed')

    download_url('https://raw.githubusercontent.com/grwlf/COVID-19_plus_Russia/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_RU.csv', store_file1)
    download_url('https://raw.githubusercontent.com/grwlf/COVID-19_plus_Russia/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_RU.csv', store_file2)

    for store_file in [store_file1, store_file2]:
        s = pd.read_csv(store_file)
        print('Latest date available in {}: {}'.format(os.path.basename(store_file), s.columns.values[-1]))


def update_yandex_metric():
    store_file1 = INPUT_PATH + 'additional/mobility-yandex.csv'
    try:
        os.remove(store_file1)
    except:
        print('Already removed')

    download_url('https://raw.githubusercontent.com/tyz910/sberbank-covid19/master/data/mobility-yandex.csv', store_file1)

    for store_file in [store_file1]:
        s = pd.read_csv(store_file)
        print('Latest date available in {}: {}'.format(os.path.basename(store_file), sorted(s['date'].unique())[-1]))


def check_data_noise(type):
    if type == 'confirmed':
        in_path = INPUT_PATH + 'time_series_covid19_confirmed_RU.csv'
    else:
        in_path = INPUT_PATH + 'time_series_covid19_deaths_RU.csv'
    tbl = pd.read_csv(in_path)

    tbl['Combined_Key'].fillna('', inplace=True)
    tbl['name'] = tbl['Combined_Key']

    dates = list(tbl.columns.values)
    remove_feats = ['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key', 'name', 'Population']
    for r in remove_feats:
        try:
            dates.remove(r)
        except:
            pass

    unique_country = set()
    unique_dates = set()
    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['name']
            unique_country |= set([cntry])
            unique_dates |= set([d])
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            try:
                summary[(cntry, d)] += cases
            except:
                print(row)
                print(cases)
                exit()

    print('Possible noise in training data: {}'.format(os.path.basename(in_path)))
    unique_country = sorted(list(unique_country))
    unique_dates = sorted(list(unique_dates))
    for c in unique_country:
        for i, d in enumerate(unique_dates[:-1]):
            # print(c, d, summary[(c, d)])
            prev, next = unique_dates[i], unique_dates[i+1]
            if summary[(c, next)] < summary[(c, prev)]:
                print(c, d, summary[(c, next)], summary[(c, prev)])


if __name__ == '__main__':
    update_current_data()
    update_yandex_metric()
    check_data_noise('confirmed')
    check_data_noise('deaths')
