# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *


def convert_timeseries_countries(type):
    if type == 'confirmed':
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    else:
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    tbl['Province/State'].fillna('', inplace=True)
    tbl['Country/Region'].fillna('', inplace=True)
    tbl['name'] = tbl[['Province/State', 'Country/Region']].agg('_'.join, axis=1)

    dates = list(tbl.columns.values)
    dates.remove('Province/State')
    dates.remove('Country/Region')
    dates.remove('Lat')
    dates.remove('Long')
    dates.remove('name')

    ccce_names = get_ccce_code_dict()
    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['name']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            try:
                summary[(cntry, d)] += cases
            except:
                print(row)
                print(cases)
                exit()

    print(summary)
    out_path = FEATURES_PATH + 'time_table_flat_for_rus_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,date,cases\n')
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(cntry, dt, summary[el]))
    out.close()

    s = pd.read_csv(out_path)
    s.sort_values(['name', 'date'], inplace=True)
    s.to_csv(out_path, index=False)


def convert_timeseries_us(type):
    if type == 'confirmed':
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    else:
        tbl = pd.read_csv(INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

    tbl['Combined_Key'].fillna('', inplace=True)
    tbl['name'] = tbl['Combined_Key']

    dates = list(tbl.columns.values)
    remove_feats = ['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key', 'name', 'Population']
    for r in remove_feats:
        try:
            dates.remove(r)
        except:
            print('Cant remove')

    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['name']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            try:
                summary[(cntry, d)] += cases
            except:
                print(row)
                print(cases)
                exit()

    print(summary)
    out_path = FEATURES_PATH + 'time_table_flat_us_for_rus_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,date,cases\n')
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(cntry, dt, summary[el]))
    out.close()

    s = pd.read_csv(out_path)
    s.sort_values(['name', 'date'], inplace=True)
    s.to_csv(out_path, index=False)


def convert_timeseries_rus(type):
    if type == 'confirmed':
        tbl = pd.read_csv(INPUT_PATH + 'time_series_covid19_confirmed_RU.csv')
    else:
        tbl = pd.read_csv(INPUT_PATH + 'time_series_covid19_deaths_RU.csv')

    tbl['Combined_Key'].fillna('', inplace=True)
    tbl['name'] = tbl['Combined_Key']

    dates = list(tbl.columns.values)
    remove_feats = ['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_','Combined_Key', 'name', 'Population']
    for r in remove_feats:
        try:
            dates.remove(r)
        except:
            print('Cant remove')

    summary = dict()
    for d in dates:
        for index, row in tbl.iterrows():
            cases = row[d]
            cntry = row['name']
            if (cntry, d) not in summary:
                summary[(cntry, d)] = 0
            try:
                summary[(cntry, d)] += cases
            except:
                print(row)
                print(cases)
                exit()

    print(summary)
    out_path = FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,date,cases\n')
    for el in sorted(list(summary.keys())):
        cntry, date = el
        arr = date.split('/')
        m, d, y = arr
        dt = "20{}.{:02d}.{:02d}".format(y, int(m), int(d))
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(cntry, dt, summary[el]))
    out.close()

    s = pd.read_csv(out_path)
    s.sort_values(['name', 'date'], inplace=True)
    s.to_csv(out_path, index=False)


def create_time_features_rus(plus_day, type):
    FEAT_SIZE = 10

    iso_names = get_russian_regions_names()
    s1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_{}.csv'.format(type))
    if USE_LATEST_DATA_RUS:
        s2 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format(type))
        s1 = pd.concat((s1, s2), axis=0)
    # s3 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_{}.csv'.format(type))
    # s4 = pd.read_csv(FEATURES_PATH + 'time_table_flat_us_for_rus_{}.csv'.format(type))
    # s5 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format(type))
    # s = pd.concat((s1, s2, s3, s4, s5), axis=0)
    s = s1

    unique_dates = sorted(s['date'].unique())[::-1]
    unique_countries = sorted(s['name'].unique())
    print('Unique dates: {}'.format(unique_dates))
    print('Unique regions: {}'.format(len(unique_countries)))
    val_matrix = np.zeros((len(unique_countries), len(unique_dates)), dtype=np.int32)
    val_matrix[...] = -1
    print(val_matrix.shape)

    out = open(FEATURES_PATH + 'features_rus_predict_{}_day_{}.csv'.format(type, plus_day), 'w')
    if plus_day > 0:
        out.write('target,')
    out.write('name1,name2,date')
    for i in range(FEAT_SIZE):
        out.write(',case_day_minus_{}'.format(i))
    out.write('\n')

    for index, row in s.iterrows():
        name, date, cases = row['name'], row['date'], row['cases']
        i0 = unique_countries.index(name)
        i1 = unique_dates.index(date)
        val_matrix[i0, i1] = cases

    print(val_matrix)

    for i in range(len(unique_countries)):
        name1 = unique_countries[i]
        if name1 in iso_names:
            name2 = iso_names[name1]
        else:
            name2 = 'XXX'
        for j in range(plus_day, len(unique_dates) - FEAT_SIZE):
            if plus_day > 0:
                target = val_matrix[i, j - plus_day]
                if target == -1:
                    continue
                if val_matrix[i, j] == -1:
                    continue
                out.write('{},'.format(target))
            if val_matrix[i, j] == -1:
                continue
            out.write('{},{},{}'.format(name1, name2, unique_dates[j]))
            for k in range(j, j + FEAT_SIZE):
                out.write(',{}'.format(val_matrix[i, k]))
            out.write('\n')
    out.close()


def convert_timeseries_only_rus_regions_last_date_manual():
    iso_names = get_russian_regions_names_v2()
    iso_names2 = get_russian_regions_names()
    in1 = open(INPUT_PATH + 'rus_lastdate.txt')
    out1 = open(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format('confirmed'), 'w')
    out2 = open(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format('deaths'), 'w')
    out1.write('name,date,cases\n')
    out2.write('name,date,cases\n')
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    while 1:
        line = in1.readline().strip()
        if line == '':
            break
        arr = line.split('\t')
        name, confirmed, deaths = arr[0].strip(), int(arr[1]), int(arr[2])
        name2 = iso_names[name]
        for el in iso_names2:
            if iso_names2[el] == name2:
                name2 = el
                break
        out1.write('{},{},{}\n'.format(name2, dt, confirmed))
        out2.write('{},{},{}\n'.format(name2, dt, deaths))
        print(arr)
    out1.close()
    out2.close()


def convert_timeseries_all_regions_last_date():
    s = pd.read_csv(INPUT_PATH + 'cases_all_latest.csv')
    s['Combined_Key'].fillna('', inplace=True)
    s['name1'] = s['Combined_Key']

    s['Province_State'].fillna('', inplace=True)
    s['Country_Region'].fillna('', inplace=True)
    s['name2'] = s[['Province_State', 'Country_Region']].agg('_'.join, axis=1)

    s1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_{}.csv'.format('confirmed'))
    s2 = pd.read_csv(FEATURES_PATH + 'time_table_flat_us_for_rus_{}.csv'.format('confirmed'))
    tmp_table = pd.concat((s1, s2), axis=0)
    needed_names = set(tmp_table['name'].unique())
    print('Needed names length: {}'.format(len(needed_names)))

    arr_confirmed = []
    arr_deaths = []
    nor_found = 0
    for index, row in s.iterrows():
        name1, name2, prov, name3, name4, date, cases, deaths = row['name1'], row['name2'], row['Province_State'], row['Country_Region'], row['ISO3'], row['Last_Update'], row['Confirmed'], row['Deaths']
        date = (date.split(' ')[0]).replace('-', '.')
        name1 = name1.replace(',', '_')
        name1 = name1.replace(' ', '_')
        name2 = name2.replace(',', '_')
        name2 = name2.replace(' ', '_')
        if name1 in needed_names:
            name = name1
        elif name2 in needed_names:
            name = name2
        else:
            nor_found += 1
            print('Name is not found: {} {}'.format(name1, name2))
            continue

        arr_confirmed.append((name, date, cases))
        arr_deaths.append((name, date, deaths))

    print('Not found: {}'.format(nor_found))

    out_path = FEATURES_PATH + 'time_table_flat_latest_all_regions_for_rus_{}.csv'.format('confirmed')
    out = open(out_path, 'w')
    out.write('name,date,cases\n')
    for i in range(len(arr_confirmed)):
        name, date, cases = arr_confirmed[i]
        out.write("{},{},{}\n".format(name, date, cases))
    out.close()

    out_path = FEATURES_PATH + 'time_table_flat_latest_all_regions_for_rus_{}.csv'.format('deaths')
    out = open(out_path, 'w')
    out.write('name,date,cases\n')
    for i in range(len(arr_deaths)):
        name, date, cases = arr_deaths[i]
        out.write("{},{},{}\n".format(name, date, cases))
    out.close()


if __name__ == '__main__':
    # convert_timeseries_countries('confirmed')
    # convert_timeseries_countries('deaths')
    # convert_timeseries_us('confirmed')
    # convert_timeseries_us('deaths')
    # convert_timeseries_all_regions_last_date()
    convert_timeseries_rus('confirmed')
    convert_timeseries_rus('deaths')
    for i in range(0, DAYS_TO_PREDICT + 1):
        create_time_features_rus(i, 'confirmed')
        create_time_features_rus(i, 'deaths')
