# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
import requests
import zipfile


def update_current_data():
    store_file1 = INPUT_PATH + 'data_rus_latest.html'

    try:
        os.remove(store_file1)
    except:
        print('Already removed')

    download_url('https://xn--80aesfpebagmfblc0a.xn--p1ai/information/', store_file1)

    html = open(store_file1, 'r', encoding='utf8').read()
    search1 = '<cv-spread-overview :spread-data=\''
    search2 = '\' region-data-url="'
    r1 = html.find(search1)
    r2 = html.find(search2)
    if r1 == -1 or r2 == -1:
        print('Parse file failed!')
        exit()
    part = html[r1+len(search1):r2]
    data = json.loads(part)

    parsed_data = []
    for d in data:
        l = [d['title'], d['sick'], d['healed'], d['died']]
        parsed_data.append(l)

    iso_names = get_russian_regions_names_v2()
    iso_names2 = get_russian_regions_names()
    out1 = open(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format('confirmed'), 'w')
    out2 = open(FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format('deaths'), 'w')
    out1.write('name,name2,date,cases\n')
    out2.write('name,name2,date,cases\n')
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    for arr in parsed_data:
        name, confirmed, deaths = arr[0].strip(), int(arr[1]), int(arr[3])
        name2 = iso_names[name]
        for el in iso_names2:
            if iso_names2[el] == name2:
                name2 = el
                break
        out1.write('{},{},{},{}\n'.format(name2, iso_names2[el], dt, confirmed))
        out2.write('{},{},{},{}\n'.format(name2, iso_names2[el], dt, deaths))
        print(arr)
    out1.close()
    out2.close()


def update_current_data_full():
    store_file1 = INPUT_PATH + 'data_rus_latest.html'

    try:
        os.remove(store_file1)
    except:
        print('Already removed')

    download_url('https://xn--80aesfpebagmfblc0a.xn--p1ai/information/', store_file1)

    html = open(store_file1, 'r', encoding='utf8').read()
    search1 = '<cv-spread-overview :spread-data=\''
    search2 = '\' region-data-url="'
    r1 = html.find(search1)
    r2 = html.find(search2)
    if r1 == -1 or r2 == -1:
        print('Parse file failed!')
        exit()
    part = html[r1+len(search1):r2]
    data = json.loads(part)

    parsed_data = []
    for d in data:
        l = [d['title'], d['sick'], d['healed'], d['died'], d['sick_incr'], d['healed_incr'], d['died_incr']]
        parsed_data.append(l)

    iso_names = get_russian_regions_names_v2()
    iso_names2 = get_russian_regions_names()
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    out1 = open(FEATURES_PATH + 'time_table_full_{}.csv'.format(dt), 'w')
    out1.write('name,name2,date,confirmed,deaths,new,active,healed\n')
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    for arr in parsed_data:
        name, confirmed, deaths, new, healed = arr[0].strip(), int(arr[1]), int(arr[3]), int(arr[4]), int(arr[2])
        name2 = iso_names[name]
        for el in iso_names2:
            if iso_names2[el] == name2:
                name2 = el
                break
        out1.write('{},{},{},{},{},{},{},{}\n'.format(name2, iso_names2[el], dt, confirmed, deaths, new, confirmed - deaths - healed, healed))
        print(arr)
    out1.close()


def check_if_updated():
    path1 = INPUT_PATH + 'time_series_covid19_confirmed_RU.csv'
    path2 = FEATURES_PATH + 'time_table_flat_for_rus_latest_{}.csv'.format('confirmed')
    s1 = pd.read_csv(path1)
    latest_date1 = s1.columns.values[-1]
    s2 = pd.read_csv(path2)
    latest_date2 = sorted(list(s2['date'].unique()))[-1]

    part1 = s1[s1['Combined_Key'] == 'Moscow,Russia'][latest_date1].values[0]
    part2 = s2[s2['name'] == 'Moscow_Russia']['cases'].values[0]
    print('Value for Moscow in {} and last date {}: {}'.format(os.path.basename(path1), latest_date1, part1))
    print('Value for Moscow in {} and last date {}: {}'.format(os.path.basename(path2), latest_date2, part2))
    if part2 > part1:
        print('Looks OK')
    else:
        print('Failed! Looks like you need to wait')


if __name__ == '__main__':
    update_current_data()
    update_current_data_full()
    check_if_updated()
