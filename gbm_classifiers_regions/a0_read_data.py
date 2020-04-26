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

                    try:
                        fitting_parameters, covariance = curve_fit(exponential_fit, points, row, maxfev=5000)
                        a, b, c = fitting_parameters
                        f4 = exponential_fit(endpoint, a, b, c)
                    except:
                        print(points, row)
                        f4 = -1
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


def add_special_additional_features(table):
    s = pd.read_csv(INPUT_PATH + 'additional/population_rus.csv')
    s['name1'] = s['name']
    table = table.merge(s[['name1', 'population' ,'population_urban', 'population_rural']], on='name1', how='left')
    table['population'] = table['population'].fillna(-1)
    table['population_urban'] = table['population_urban'].fillna(-1)
    table['population_rural'] = table['population_rural'].fillna(-1)

    return table


def add_weekday(table, day):
    from datetime import datetime, timedelta

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
        first_case = pd.read_csv(FEATURES_PATH + 'first_date_for_rus_{}.csv'.format(type))
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
        # print(len(delta))
    return table


def gen_simple_linear_features(table, day, type):
    features = []
    for i in range(LIMIT_DAYS):
        features.append('case_day_minus_{}'.format(i))
    matrix_init = table[features].values[:, ::-1].astype(np.float32)
    matrix_log = np.log10(matrix_init + 1)

    for number, matrix in enumerate([matrix_init, matrix_log]):
        cache_path = CACHE_PATH + 'day_{}_type_{}_num_{}_simple_feats_hash_{}.pkl'.format(day, type, number, str(sha224(matrix.data.tobytes()).hexdigest()))
        if not os.path.isfile(cache_path) or 1:
            ival = []
            endpoint = matrix.shape[1] + day - 1
            for i in range(matrix.shape[0]):
                gen_feats = [0] * (LIMIT_DAYS - 1)
                row = matrix[i]
                if row.sum() != 0:
                    for j in range(LIMIT_DAYS - 1):
                        point1 = row[j]
                        point2 = row[LIMIT_DAYS - 1]
                        delta1 = LIMIT_DAYS - 1 - j
                        linear_pred = point2 + day * (point2 - point1) / delta1
                        gen_feats[j] = linear_pred
                ival.append(gen_feats)
                # print(row, gen_feats, day, endpoint, LIMIT_DAYS - 1)
            ival = np.array(ival, dtype=np.float32)
            # save_in_file_fast(ival, cache_path)
        else:
            ival = load_from_file_fast(cache_path)

        for i in range(LIMIT_DAYS - 1):
            if number == 0:
                table['linear_extra_{}'.format(i)] = ival[:, i]
            else:
                table['linear_log_extra_{}'.format(i)] = ival[:, i]

    return table


def add_yandex_mobility_data(mobility, table, yandex_shift, day, type):
    NEEDED_PERIOD = 10
    yandex = dict()
    unique_dates = set()
    unique_regions = set()
    for index, row in mobility.iterrows():
        date = row['date'].replace('-', '.')
        unique_dates |= set([date])
        country = row['country']
        unique_regions |= set([country])
        yandex[(country, date)] = row['isolation']

    # print(sorted(list(unique_dates)))
    # print(len(unique_regions), sorted(list(unique_regions)))

    yandex_matrix = []
    for index, row in table.iterrows():
        date = row['date']
        country = row['name2']
        # print(date, country)
        datetime_object = datetime.strptime(date, '%Y.%m.%d')
        lst = []
        for i in range(NEEDED_PERIOD):
            delta = datetime_object - timedelta(days=yandex_shift)
            shifted_date = delta.strftime('%Y.%m.%d')
            if (country, shifted_date) in yandex:
                value = yandex[(country, shifted_date)]
            else:
                value = -1
            # print(date, shifted_date, value)
            lst.append(value)
        yandex_matrix.append(lst)

    yandex_matrix = np.array(yandex_matrix)
    # print(yandex_matrix.shape)
    for i in range(NEEDED_PERIOD):
        table['yandex_isolation_{}'.format(i)] = yandex_matrix[:, i]
    return table


def add_area_and_density(table):
    s = pd.read_csv(INPUT_PATH + 'additional/data_rus_regions_upd.csv')
    s['density'] = s['population_2020'].values / s['area'].values
    table = table.merge(s[['name2', 'area', 'density']], on='name2', how='left')
    return table


def read_input_data(day, type, step_back_days=None):
    train = pd.read_csv(FEATURES_PATH + 'features_rus_predict_{}_day_{}.csv'.format(type, day))
    test = pd.read_csv(FEATURES_PATH + 'features_rus_predict_{}_day_{}.csv'.format(type, 0))
    print('Initial train: {} Initial test: {}'.format(len(train), len(test)))

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

    if 1:
        # Remove all additional data
        l1 = len(train)
        train = train[train['name2'] != 'XXX']
        l2 = len(train)
        train.reset_index(drop=True, inplace=True)
        print('Removed XXX target. Reduction {} -> {}'.format(l1, l2))

    all_names = list(train['name2']) + list(test['name2'])
    train['country'] = encode_country(train['name2'], all_names)
    test['country'] = encode_country(test['name2'], all_names)

    # Remove unneeded data from test
    test = decrease_table_for_last_date(test)
    print('Updated train: {} Updated test: {}'.format(len(train), len(test)))

    train = add_area_and_density(train)
    test = add_area_and_density(test)

    if 1:
        # Yandex mobility data
        # https://github.com/tyz910/sberbank-covid19/blob/master/data/mobility-yandex.csv
        yandex_mobility_path = INPUT_PATH + 'additional/mobility-yandex.csv'
        if os.path.isfile(yandex_mobility_path):
            mobility_table = pd.read_csv(yandex_mobility_path)
            dt1 = sorted(test['date'].unique())[-1]
            dt2 = sorted(mobility_table['date'].unique())[-1]
            print(dt1, dt2)
            dt1 = datetime.strptime(dt1, '%Y.%m.%d')
            dt2 = datetime.strptime(dt2, '%Y-%m-%d')
            yandex_shift = dt1 - dt2
            yandex_shift = yandex_shift.days
            if yandex_shift < 1:
                yandex_shift = 1
            print('Yandex shift days: {}'.format(yandex_shift))

            train = add_yandex_mobility_data(mobility_table, train, yandex_shift, day, type)
            test = add_yandex_mobility_data(mobility_table, test, yandex_shift, day, type)
        else:
            print('No yandex mobility data. Check: {}'.format(yandex_mobility_path))

    train = gen_additional_features(train)
    test = gen_additional_features(test)

    if 1:
        train = gen_simple_linear_features(train, day, type)
        test = gen_simple_linear_features(test, day, type)

    train = gen_interpolation_features(train, day, type)
    test = gen_interpolation_features(test, day, type)

    train = add_special_additional_features(train)
    test = add_special_additional_features(test)

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
