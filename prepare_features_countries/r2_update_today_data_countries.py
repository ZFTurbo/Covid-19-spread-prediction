# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
import requests
import zipfile


def update_current_data():
    store_file1 = INPUT_PATH + 'cases_country_latest.csv'
    store_file2 = INPUT_PATH + 'cases_all_latest.csv'
    store_file3 = INPUT_PATH + 'cases_state_latest.csv'

    try:
        os.remove(store_file1)
        os.remove(store_file2)
        os.remove(store_file3)
    except:
        print('Already removed')

    download_url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv', store_file1)
    download_url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases.csv', store_file2)
    download_url('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_state.csv', store_file3)


def check_if_updated():
    path1 = INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    path2 = INPUT_PATH + 'cases_country_latest.csv'
    s1 = pd.read_csv(path1)
    latest_date1 = s1.columns.values[-1]
    s2 = pd.read_csv(path2)
    latest_date2 = sorted(list(s2['Last_Update'].unique()))[-1]
    print('Latest date in {} file: {}'.format(os.path.basename(path1), latest_date1))
    print('Latest date in {} file: {}'.format(os.path.basename(path2), latest_date2))

    # print(s1[latest_date1])
    # print(s2['Confirmed'])
    s1 = s1[s1['Province/State'].isna()]
    # print(len(s1))

    updated = 0
    for index, row in s1.iterrows():
        nm = row['Country/Region']
        case1 = row[latest_date1]
        case2 = s2[s2['Country_Region'] == nm]['Confirmed'].values[0]
        # print(nm, case1, case2)
        if case2 > case1:
            updated += 1

    print('Updated countries: {} from {} ({:.2f} %)'.format(updated, len(s1), 100 * updated / len(s1)))


if __name__ == '__main__':
    update_current_data()
    check_if_updated()