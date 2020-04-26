# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
from prepare_features_rus.r3_convert_timeseries_rus import get_russian_regions_names


def get_inverted_names(d):
    res = dict()
    for el in d:
        res[d[el]] = el
    return res


def convert_population_rus():
    regions = pd.read_csv(INPUT_PATH + 'russia_regions.csv')
    reg_names = get_inverted_names(get_russian_regions_names())
    out = open(INPUT_PATH + 'additional/population_rus.csv', 'w')
    out.write('name,iso_name,population,population_urban,population_rural\n')
    for index, row in regions.iterrows():
        name = row['iso_code']
        print(name, reg_names[name])
        out.write('{},{},{},{},{}\n'.format(reg_names[name], name, row['population'], row['population_urban'], row['population_rural']))
    out.close()


if __name__ == '__main__':
    convert_population_rus()
