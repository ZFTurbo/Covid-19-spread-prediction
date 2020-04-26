# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
import requests
import zipfile


def get_first_case_date(type):
    if type == 'confirmed':
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    else:
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    con = pd.read_csv(INPUT_PATH + 'countries.csv', dtype={'iso_alpha3': str}, keep_default_na=False)

    print(tbl.columns.values)
    print(con.columns.values)

    not_found = 0
    found = 0
    unique_countries = tbl['Country/Region'].unique()
    for u in unique_countries:
        if u not in list(con['ccse_name'].values):
            not_found += 1
            print('Not Found: {}'.format(u))
        else:
            found += 1
    print(not_found)
    print(found)

    dates = list(tbl.columns.values)
    dates.remove('Province/State')
    dates.remove('Country/Region')
    dates.remove('Lat')
    dates.remove('Long')

    ccce_names = get_ccce_code_dict()
    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['Country/Region']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            summary[(cntry, d)] += cases

    summary_sorted = dict()
    dates_sorted = set()
    all_countries = set()
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        dates_sorted |= set([dt])
        all_countries |= set([cntry])
        summary_sorted[(cntry, dt)] = summary[el]
    dates_sorted = sorted(list(dates_sorted))
    all_countries = sorted(list(all_countries))

    print(dates_sorted)

    first_case = dict()
    for a in all_countries:
        first_case[a] = '2020.01.21'
        for i, dt in enumerate(dates_sorted[:-1]):
            el1 = (a, dates_sorted[i])
            el2 = (a, dates_sorted[i+1])
            if summary_sorted[el1] == 0 and summary_sorted[el2] > 0:
                first_case[a] = dates_sorted[i+1]

    out_path = FEATURES_PATH + 'first_date_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,name2,date\n')
    for cntry in sorted(list(first_case.keys())):
        dt = first_case[cntry]
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(c, cntry, dt))
    out.close()


if __name__ == '__main__':
    get_first_case_date('confirmed')
    get_first_case_date('deaths')
