# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
import requests
import zipfile


def convert_timeseries_last_date():
    s = pd.read_csv(INPUT_PATH + 'cases_country_latest.csv')
    ccce_names = get_ccce_code_dict()
    arr_confirmed = []
    arr_deaths = []
    for index, row in s.iterrows():
        name, name2, date, cases, deaths = row['Country_Region'], row['ISO3'], row['Last_Update'], row['Confirmed'], row['Deaths']
        date = (date.split(' ')[0]).replace('-', '.')
        if name in ccce_names:
            c = ccce_names[name]
        else:
            c = 'XXX'
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        name2 = c

        arr_confirmed.append((name2, name, date, cases))
        arr_deaths.append((name2, name, date, deaths))

    out_path = FEATURES_PATH + 'time_table_flat_latest_{}.csv'.format('confirmed')
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for i in range(len(arr_confirmed)):
        name, name2, date, cases = arr_confirmed[i]
        out.write("{},{},{},{}\n".format(name, name2, date, cases))
    out.close()

    out_path = FEATURES_PATH + 'time_table_flat_latest_{}.csv'.format('deaths')
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for i in range(len(arr_deaths)):
        name, name2, date, cases = arr_deaths[i]
        out.write("{},{},{},{}\n".format(name, name2, date, cases))
    out.close()


def convert_timeseries_confirmed(type):
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

    print(summary)
    out_path = FEATURES_PATH + 'time_table_flat_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,name2,date,cases\n')
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        if cntry in ccce_names:
            c = ccce_names[cntry]
        else:
            c = 'XXX'
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{},{}\n".format(c, cntry, dt, summary[el]))
    out.close()

    s = pd.read_csv(out_path)
    if USE_LATEST_DATA_COUNTRY:
        latest = pd.read_csv(FEATURES_PATH + 'time_table_flat_latest_{}.csv'.format(type))
        s = pd.concat((s, latest))
    s.sort_values(['name', 'date'], inplace=True)
    s.to_csv(out_path, index=False)


def create_time_features_confirmed(plus_day, type):
    FEAT_SIZE = 10

    ccce_names = get_ccce_code_dict()
    s = pd.read_csv(FEATURES_PATH + 'time_table_flat_{}.csv'.format(type))
    unique_dates = sorted(s['date'].unique())[::-1]
    unique_countries = sorted(s['name2'].unique())
    print(unique_dates)
    print(unique_countries)
    val_matrix = np.zeros((len(unique_countries), len(unique_dates)), dtype=np.int32)
    print(val_matrix.shape)

    out = open(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type, plus_day), 'w')
    if plus_day > 0:
        out.write('target,')
    out.write('name1,name2,date')
    for i in range(FEAT_SIZE):
        out.write(',case_day_minus_{}'.format(i))
    out.write('\n')

    for index, row in s.iterrows():
        name, name2, date, cases = row['name'], row['name2'], row['date'], row['cases']
        i0 = unique_countries.index(name2)
        i1 = unique_dates.index(date)
        val_matrix[i0, i1] = cases

    print(val_matrix)

    for i in range(len(unique_countries)):
        name1 = unique_countries[i]
        if name1 in ccce_names:
            name2 = ccce_names[name1]
        else:
            name2 = 'XXX'
        for j in range(plus_day, len(unique_dates) - FEAT_SIZE):
            if plus_day > 0:
                target = val_matrix[i, j - plus_day]
                out.write('{},'.format(target))
            out.write('{},{},{}'.format(name1, name2, unique_dates[j]))
            for k in range(j, j + FEAT_SIZE):
                out.write(',{}'.format(val_matrix[i, k]))
            out.write('\n')
    out.close()


if __name__ == '__main__':
    convert_timeseries_last_date()
    convert_timeseries_confirmed('confirmed')
    convert_timeseries_confirmed('deaths')
    for i in range(0, DAYS_TO_PREDICT + 1):
        create_time_features_confirmed(i, 'confirmed')
        create_time_features_confirmed(i, 'deaths')
