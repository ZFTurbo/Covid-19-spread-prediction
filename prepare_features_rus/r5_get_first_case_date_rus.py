# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from prepare_features_rus.r3_convert_timeseries_rus import get_russian_regions_names


def get_first_case_date(type):
    s1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_{}.csv'.format(type))
    s2 = pd.read_csv(FEATURES_PATH + 'time_table_flat_us_for_rus_{}.csv'.format(type))
    s3 = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_{}.csv'.format(type))
    s4 = pd.read_csv(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format(type))
    s5 = pd.read_csv(FEATURES_PATH + 'time_table_flat_latest_all_regions_for_rus_{}.csv'.format(type))
    s = pd.concat((s1, s2, s3, s4, s5), axis=0)

    first_case = dict()
    unique_names = s['name'].unique()
    for name in unique_names:
        part = s[s['name'] == name]
        part = part.sort_values('date')
        dates = part['date'].values
        cases = part['cases'].values
        first_case[name] = '2020.01.21'
        for i in range(len(dates) - 1):
            if cases[i] == 0 and cases[i + 1] > 0:
                first_case[name] = dates[i + 1]
                break
        print(name, len(part), first_case[name])

    short_name = get_russian_regions_names()
    out_path = FEATURES_PATH + 'first_date_for_rus_{}.csv'.format(type)
    out = open(out_path, 'w')
    out.write('name,name2,date\n')
    for cntry in sorted(list(first_case.keys())):
        dt = first_case[cntry]
        if cntry in short_name:
            c = short_name[cntry]
        else:
            c = 'XXX'
        cntry = cntry.replace(',', '_')
        cntry = cntry.replace(' ', '_')
        out.write("{},{},{}\n".format(c, cntry, dt))
    out.close()


if __name__ == '__main__':
    get_first_case_date('confirmed')
    get_first_case_date('deaths')
