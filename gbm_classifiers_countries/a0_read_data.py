# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from hashlib import sha224
from datetime import datetime, timedelta


def get_kfold_split_v2(folds_number, train, random_state):
    train_index = list(range(len(train)))
    folds = KFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index)):
        ret.append([trn_idx, val_idx])
    return ret


def encode_country(values, names):
    names = sorted(list(set(names)))
    c = []
    for v in values:
        c.append(names.index(v))
    return c


def contest_metric(true, pred):
    s = (pred + 1) / (true + 1)
    error = np.absolute(np.log10(s)).mean()
    return error


def decrease_table_for_last_date(table):
    unique_dates = sorted(list(table['date'].unique()))
    last_date = unique_dates[-1]
    table = table[table['date'] == last_date]
    table = table[table['name2'] != 'XXX']
    return table


def gen_additional_features(table):
    for i in range(LIMIT_DAYS):
        f = 'case_day_minus_{}'.format(i)
        table[f + '_log10'] = np.log10(table[f].values + 1)

    for i in range(LIMIT_DAYS - 1):
        f1 = 'case_day_minus_{}'.format(i)
        f2 = 'case_day_minus_{}'.format(i + 1)
        table['diff_div_{}'.format(i)] = (table[f1].values + 1) / (table[f2].values + 1)
        table['diff_div_{}_opp'.format(i)] = (table[f2].values + 1) / (table[f1].values + 1)

    for i in range(LIMIT_DAYS - 1):
        f1 = 'case_day_minus_{}'.format(i)
        f2 = 'case_day_minus_{}'.format(i + 1)
        table['diff_log_div_{}'.format(i)] = (np.log10(table[f1].values + 1) + 1) / (np.log10(table[f2].values + 1) + 1)
        table['diff_log_div_{}_opp'.format(i)] = (np.log10(table[f2].values + 1) + 1) / (np.log10(table[f1].values + 1) + 1)

    return table


def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


def gen_interpolation_features(table, day, type):
    from scipy import interpolate
    from scipy.optimize import curve_fit

    features = []
    for i in range(LIMIT_DAYS):
        features.append('case_day_minus_{}'.format(i))
    matrix_init = table[features].values[:, ::-1].astype(np.float32)
    matrix_log = np.log10(matrix_init + 1)

    for number, matrix in enumerate([matrix_init, matrix_log]):
        cache_path = CACHE_PATH + 'day_{}_type_{}_num_{}_hash_{}.pkl'.format(day, type, number, str(sha224(matrix.data.tobytes()).hexdigest()))
        if not os.path.isfile(cache_path):
            ival = dict()
            ival[0] = []
            ival[1] = []
            ival[2] = []
            ival[3] = []
            points = list(range(matrix.shape[1]))
            endpoint = matrix.shape[1] + day - 1
            for i in range(matrix.shape[0]):
                row = matrix[i]
                if row.sum() == 0:
                    ival[0].append(0)
                    ival[1].append(0)
                    ival[2].append(0)
                    ival[3].append(0)
                else:
                    f1 = interpolate.interp1d(points, row, kind='slinear', fill_value='extrapolate')
                    ival[0].append(f1(endpoint))
                    f2 = interpolate.interp1d(points, row, kind='quadratic', fill_value='extrapolate')
                    ival[1].append(f2(endpoint))
                    f3 = interpolate.interp1d(points, row, kind='cubic', fill_value='extrapolate')
                    ival[2].append(f3(endpoint))

                    fitting_parameters, covariance = curve_fit(exponential_fit, points, row, maxfev=5000)
                    a, b, c = fitting_parameters
                    f4 = exponential_fit(endpoint, a, b, c)
                    # print(row, f1(endpoint), f2(endpoint), f3(endpoint), f4)
                    ival[3].append(f4)
            save_in_file_fast(ival, cache_path)
        else:
            ival = load_from_file_fast(cache_path)

        if number == 0:
            table['interpol_1'] = ival[0]
            table['interpol_2'] = ival[1]
            table['interpol_3'] = ival[2]
            table['interpol_4'] = ival[3]
        else:
            table['interpol_log_1'] = ival[0]
            table['interpol_log_2'] = ival[1]
            table['interpol_log_3'] = ival[2]
            table['interpol_log_4'] = ival[3]

    return table


def add_country_features(table):
    s = pd.read_csv(INPUT_PATH + 'countries.csv')
    s['name2'] = s['iso_alpha3']
    s = s[['name2', 'density', 'fertility_rate', 'land_area', 'median_age', 'migrants', 'population', 'urban_pop_rate', 'world_share']]
    table = table.merge(s, on='name2', how='left')
    return table


def add_special_additional_features(table):
    s = pd.read_csv(INPUT_PATH + 'additional/2.12_Health_systems_converted.csv')
    s['name2'] = s['iso_alpha3']
    s = s[s['name2'] != 'XXX']
    s.drop('iso_alpha3', axis=1, inplace=True)
    s.drop('Country_Region', axis=1, inplace=True)
    s.drop('Province_State', axis=1, inplace=True)
    s.drop('World_Bank_Name', axis=1, inplace=True)
    # print(len(s), len(s['name2'].unique()))
    table = table.merge(s, on='name2', how='left')

    s = pd.read_csv(INPUT_PATH + 'additional/share-of-adults-who-smoke_converted.csv')
    s['name2'] = s['iso_alpha3']
    s = s[s['name2'] != 'XXX']
    s.drop('iso_alpha3', axis=1, inplace=True)
    table = table.merge(s, on='name2', how='left')

    s = pd.read_csv(INPUT_PATH + 'additional/WorldPopulationByAge2020_converted.csv')
    s['name2'] = s['iso_alpha3']
    s = s[s['name2'] != 'XXX']
    s.drop('iso_alpha3', axis=1, inplace=True)
    table = table.merge(s, on='name2', how='left')

    table.fillna(-1, inplace=True)
    return table


def add_weekday(table, day):
    weekday = []
    dates = table['date'].values
    for d in dates:
        datetime_object = datetime.strptime(d, '%Y.%m.%d')
        datetime_object += timedelta(days=day)
        w = datetime_object.weekday()
        weekday.append(w)
    table['weekday'] = weekday
    return table


def remove_latest_days(table, days):
    dates = sorted(table['date'].unique())
    dates_valid = dates[:-days]
    table = table[table['date'].isin(dates_valid)]
    return table


def days_from_first_case(table, type):
    for type in ['confirmed', 'deaths']:
        first_case = pd.read_csv(FEATURES_PATH + 'first_date_{}.csv'.format(type))
        first_case = first_case[['name', 'name2', 'date']].values
        fc = dict()
        for i in range(first_case.shape[0]):
            fc[first_case[i, 1]] = first_case[i, 2]

        delta = []
        for index, row in table.iterrows():
            dt1 = datetime.strptime(row['date'], '%Y.%m.%d')
            dt2 = datetime.strptime(fc[row['name1']], '%Y.%m.%d')
            diff = dt1 - dt2
            delta.append(diff.days)

        table['days_from_first_case_{}'.format(type)] = delta
    return table


def read_input_data(day, type, step_back_days=None):
    train = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type, day))
    test = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type, 0))
    print(len(train), len(test))

    # Add data from parallel features
    if 1:
        # print(len(train), len(test))
        if type == 'confirmed':
            type_opposite = 'deaths'
            train_add = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type_opposite, day))
            test_add = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type_opposite, day))
        elif type == 'deaths':
            type_opposite = 'confirmed'
            train_add = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type_opposite, day))
            test_add = pd.read_csv(FEATURES_PATH + 'features_predict_{}_day_{}.csv'.format(type_opposite, day))
        else:
            train_add = None
            test_add = None
            type_opposite = None
            print('Error')
            exit()

        feats = ['name1', 'name2', 'date']
        for table in [train_add, test_add]:
            for i in range(2):
                f_name = 'case_{}_day_minus_{}'.format(type_opposite, i)
                table.rename(columns={'case_day_minus_{}'.format(i): f_name}, inplace=True)
                if f_name not in feats:
                    feats.append(f_name)

        # print(train_add.columns.values)

        train = train.merge(train_add[feats], on=['name1', 'name2', 'date'], how='left')
        test = test.merge(test_add[feats], on=['name1', 'name2', 'date'], how='left')
        # print(len(train), len(test))
        # print(train.columns.values)

    if step_back_days is not None:
        train = remove_latest_days(train, step_back_days)
        test = remove_latest_days(test, step_back_days)

    if 1:
        # Remove zero target (must be faster training)
        l1 = len(train)
        train = train[train['target'] > 0]
        l2 = len(train)
        train.reset_index(drop=True, inplace=True)
        print('Removed zero target. Reduction {} -> {}'.format(l1, l2))

    all_names = list(train['name1']) + list(test['name1'])
    train['country'] = encode_country(train['name1'], all_names)
    test['country'] = encode_country(test['name1'], all_names)

    # Remove unneeded data from test
    test = decrease_table_for_last_date(test)
    print(len(train), len(test))

    train = gen_additional_features(train)
    test = gen_additional_features(test)

    if USE_INTERPOLATION_FEATURES:
        train = gen_interpolation_features(train, day, type)
        test = gen_interpolation_features(test, day, type)

    train = add_country_features(train)
    test = add_country_features(test)

    train = add_special_additional_features(train)
    test = add_special_additional_features(test)

    if USE_WEEKDAY_FEATURES:
        train = add_weekday(train, day)
        test = add_weekday(test, day)

    train = days_from_first_case(train, type)
    test = days_from_first_case(test, type)

    features = list(test.columns.values)
    features.remove('name1')
    features.remove('name2')
    features.remove('date')
    print(len(train), len(test))
    # test.to_csv(CACHE_PATH + 'debug_test.csv', index=False)
    return train, test, features


def get_params():
    params = {
        'target': 'target',
        'id' : 'id',
        'metric': 'mean_squared_error',
    }
    params['metric_function'] = mean_squared_error
    return params
