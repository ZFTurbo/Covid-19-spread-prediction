# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score
import random
import shutil
import operator
import platform
import json
import base64
import typing as t
import zlib
import requests
import zipfile
from a0_settings import *


random.seed(2016)
np.random.seed(2016)

def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_rgb(im, name='image'):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('acc', 'val_acc')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def read_single_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def get_country_list():
    tbl = pd.read_csv(INPUT_PATH + 'countries.csv', dtype={'iso_alpha3': str}, keep_default_na=False)
    countries = sorted(tbl['iso_alpha3'].values)
    return countries


def get_ccce_code_dict():
    tbl = pd.read_csv(INPUT_PATH + 'countries.csv', dtype={'iso_alpha3': str}, keep_default_na=False)
    res = dict()
    for index, row in tbl.iterrows():
        cntry = row['ccse_name'].replace(',', '_')
        cntry = cntry.replace(' ', '_')
        res[row['ccse_name']] = row['iso_alpha3']
        res[cntry] = row['iso_alpha3']
    # print(res)
    return res


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def get_russian_regions_names_v2():
    short_name = {
        'Алтайский край': 'RU-ALT',
        'Амурская область': 'RU-AMU',
        'Архангельская область': 'RU-ARK',
        'Астраханская область': 'RU-AST',
        'Белгородская область': 'RU-BEL',
        'Брянская область': 'RU-BRY',
        'Чеченская Республика': 'RU-CE',
        'Челябинская область': 'RU-CHE',
        'Чукотский автономный округ': 'RU-CHU',
        'Хабаровский край': 'RU-KHA',
        'Ханты-Мансийский АО': 'RU-KHM',
        'Республика Ингушетия': 'RU-IN',
        'Иркутская область': 'RU-IRK',
        'Ивановская область': 'RU-IVA',
        'Еврейская автономная область': 'RU-YEV',
        'Калининградская область': 'RU-KGD',
        'Калужская область': 'RU-KLU',
        'Камчатский край': 'RU-KAM',
        'Кемеровская область': 'RU-KEM',
        'Кировская область': 'RU-KIR',
        'Республика Коми': 'RU-KO',
        'Костромская область': 'RU-KOS',
        'Краснодарский край': 'RU-KDA',
        'Красноярский край': 'RU-KYA',
        'Курганская область': 'RU-KGN',
        'Курская область': 'RU-KRS',
        'Ленинградская область': 'RU-LEN',
        'Липецкая область': 'RU-LIP',
        'Магаданская область': 'RU-MAG',
        'Москва': 'RU-MOW',
        'Московская область': 'RU-MOS',
        'Мурманская область': 'RU-MUR',
        'Ненецкий автономный округ': 'RU-NEN',
        'Нижегородская область': 'RU-NIZ',
        'Новгородская область': 'RU-NGR',
        'Новосибирская область': 'RU-NVS',
        'Омская область': 'RU-OMS',
        'Орловская область': 'RU-ORL',
        'Оренбургская область': 'RU-ORE',
        'Пензенская область': 'RU-PNZ',
        'Пермский край': 'RU-PER',
        'Приморский край': 'RU-PRI',
        'Псковская область': 'RU-PSK',
        'Республика Алтай': 'RU-AL',
        'Республика Адыгея': 'RU-AD',
        'Республика Башкортостан': 'RU-BA',
        'Республика Бурятия': 'RU-BU',
        'Республика Чувашия': 'RU-CU',
        'Республика Крым': 'UA-43',
        'Республика Дагестан': 'RU-DA',
        'Республика Хакасия': 'RU-KK',
        'Кабардино-Балкарская Республика': 'RU-KB',
        'Республика Калмыкия': 'RU-KL',
        'Карачаево-Черкесская Республика': 'RU-KC',
        'Республика Карелия': 'RU-KR',
        'Республика Марий Эл': 'RU-ME',
        'Республика Мордовия': 'RU-MO',
        'Республика Северная Осетия — Алания': 'RU-SE',
        'Республика Татарстан': 'RU-TA',
        'Республика Тыва': 'RU-TY',
        'Удмуртская Республика': 'RU-UD',
        'Ростовская область': 'RU-ROS',
        'Рязанская область': 'RU-RYA',
        'Республика Саха (Якутия)': 'RU-SA',
        'Санкт-Петербург': 'RU-SPE',
        'Сахалинская область': 'RU-SAK',
        'Самарская область': 'RU-SAM',
        'Саратовская область': 'RU-SAR',
        'Севастополь': 'UA-40',
        'Смоленская область': 'RU-SMO',
        'Ставропольский край': 'RU-STA',
        'Свердловская область': 'RU-SVE',
        'Тамбовская область': 'RU-TAM',
        'Томская область': 'RU-TOM',
        'Тульская область': 'RU-TUL',
        'Тюменская область': 'RU-TYU',
        'Тверская область': 'RU-TVE',
        'Ульяновская область': 'RU-ULY',
        'Владимирская область': 'RU-VLA',
        'Волгоградская область': 'RU-VGG',
        'Вологодская область': 'RU-VLG',
        'Воронежская область': 'RU-VOR',
        'Ямало-Ненецкий автономный округ': 'RU-YAN',
        'Ярославская область': 'RU-YAR',
        'Забайкальский край': 'RU-ZAB',
    }
    return short_name


def get_russian_regions_names():
    if 0:
        s = pd.read_csv(INPUT_PATH + 'russia_regions.csv')
        s1 = pd.read_csv(FEATURES_PATH + 'time_table_flat_rus_regions_for_rus_{}.csv'.format('confirmed'))
        print(s['iso_code'].values)
        print(s['name'].values)
        print(sorted(s1['name'].unique()))
    short_name = {
        'Altayskiy_kray_Russia': 'RU-ALT',
        'Amursk_oblast_Russia': 'RU-AMU',
        'Arkhangelsk_oblast_Russia': 'RU-ARK',
        'Astrahan_oblast_Russia': 'RU-AST',
        'Belgorod_oblast_Russia': 'RU-BEL',
        'Briansk_oblast_Russia': 'RU-BRY',
        'Chechen_republic_Russia': 'RU-CE',
        'Cheliabinsk_oblast_Russia': 'RU-CHE',
        'Chukotskiy_autonomous_oblast_Russia': 'RU-CHU',
        'Habarovskiy_kray_Russia': 'RU-KHA',
        'Hanty-Mansiyskiy_AO_Russia': 'RU-KHM',
        'Ingushetia_republic_Russia': 'RU-IN',
        'Irkutsk_oblast_Russia': 'RU-IRK',
        'Ivanovo_oblast_Russia': 'RU-IVA',
        'Jewish_Autonomous_oblast_Russia': 'RU-YEV',
        'Kaliningrad_oblast_Russia': 'RU-KGD',
        'Kaluga_oblast_Russia': 'RU-KLU',
        'Kamchatskiy_kray_Russia': 'RU-KAM',
        'Kemerovo_oblast_Russia': 'RU-KEM',
        'Kirov_oblast_Russia': 'RU-KIR',
        'Komi_republic_Russia': 'RU-KO',
        'Kostroma_oblast_Russia': 'RU-KOS',
        'Krasnodarskiy_kray_Russia': 'RU-KDA',
        'Krasnoyarskiy_kray_Russia': 'RU-KYA',
        'Kurgan_oblast_Russia': 'RU-KGN',
        'Kursk_oblast_Russia': 'RU-KRS',
        'Leningradskaya_oblast_Russia': 'RU-LEN',
        'Lipetsk_oblast_Russia': 'RU-LIP',
        'Magadan_oblast_Russia': 'RU-MAG',
        'Moscow_Russia': 'RU-MOW',
        'Moscow_oblast_Russia': 'RU-MOS',
        'Murmansk_oblast_Russia': 'RU-MUR',
        'Nenetskiy_autonomous_oblast_Russia': 'RU-NEN',
        'Nizhegorodskaya_oblast_Russia': 'RU-NIZ',
        'Novgorod_oblast_Russia': 'RU-NGR',
        'Novosibirsk_oblast_Russia': 'RU-NVS',
        'Omsk_oblast_Russia': 'RU-OMS',
        'Orel_oblast_Russia': 'RU-ORL',
        'Orenburg_oblast_Russia': 'RU-ORE',
        'Pensa_oblast_Russia': 'RU-PNZ',
        'Perm_oblast_Russia': 'RU-PER',
        'Primorskiy_kray_Russia': 'RU-PRI',
        'Pskov_oblast_Russia': 'RU-PSK',
        'Altay_republic_Russia': 'RU-AL',
        'Republic_of_Adygeia_Russia': 'RU-AD',
        'Republic_of_Bashkortostan_Russia': 'RU-BA',
        'Republic_of_Buriatia_Russia': 'RU-BU',
        'Republic_of_Chuvashia_Russia': 'RU-CU',
        'Republic_of_Crimea_Russia': 'UA-43',
        'Republic_of_Dagestan_Russia': 'RU-DA',
        'Republic_of_Hakassia_Russia': 'RU-KK',
        'Republic_of_Kabardino-Balkaria_Russia': 'RU-KB',
        'Republic_of_Kalmykia_Russia': 'RU-KL',
        'Republic_of_Karachaevo-Cherkessia_Russia': 'RU-KC',
        'Republic_of_Karelia_Russia': 'RU-KR',
        'Republic_of_Mariy_El_Russia': 'RU-ME',
        'Republic_of_Mordovia_Russia': 'RU-MO',
        'Republic_of_North_Osetia_-_Alania_Russia': 'RU-SE',
        'Republic_of_Tatarstan_Russia': 'RU-TA',
        'Republic_of_Tyva_Russia': 'RU-TY',
        'Republic_of_Udmurtia_Russia': 'RU-UD',
        'Rostov_oblast_Russia': 'RU-ROS',
        'Ryazan_oblast_Russia': 'RU-RYA',
        'Saha_republic_Russia': 'RU-SA',
        'Saint_Petersburg_Russia': 'RU-SPE',
        'Sakhalin_oblast_Russia': 'RU-SAK',
        'Samara_oblast_Russia': 'RU-SAM',
        'Saratov_oblast_Russia': 'RU-SAR',
        'Sevastopol_Russia': 'UA-40',
        'Smolensk_oblast_Russia': 'RU-SMO',
        'Stavropolskiy_kray_Russia': 'RU-STA',
        'Sverdlov_oblast_Russia': 'RU-SVE',
        'Tambov_oblast_Russia': 'RU-TAM',
        'Tomsk_oblast_Russia': 'RU-TOM',
        'Tula_oblast_Russia': 'RU-TUL',
        'Tumen_oblast_Russia': 'RU-TYU',
        'Tver_oblast_Russia': 'RU-TVE',
        'Ulianovsk_oblast_Russia': 'RU-ULY',
        'Vladimir_oblast_Russia': 'RU-VLA',
        'Volgograd_oblast_Russia': 'RU-VGG',
        'Vologda_oblast_Russia': 'RU-VLG',
        'Voronezh_oblast_Russia': 'RU-VOR',
        'Yamalo-Nenetskiy_AO_Russia': 'RU-YAN',
        'Yaroslavl_oblast_Russia': 'RU-YAR',
        'Zabaykalskiy_kray_Russia': 'RU-ZAB',
    }
    return short_name


def get_russian_regions_names_all_variants():
    short_name = {
        'Алтайский край': 'RU-ALT',
        'Амурская область': 'RU-AMU',
        'Архангельская область': 'RU-ARK',
        'Астраханская область': 'RU-AST',
        'Белгородская область': 'RU-BEL',
        'Брянская область': 'RU-BRY',
        'Чеченская Республика': 'RU-CE',
        'Челябинская область': 'RU-CHE',
        'Чукотский автономный округ': 'RU-CHU',
        'Чукотский АО': 'RU-CHU',
        'Хабаровский край': 'RU-KHA',
        'Ханты-Мансийский АО': 'RU-KHM',
        'Ханты-Мансийский АО — Югра': 'RU-KHM',
        'Республика Ингушетия': 'RU-IN',
        'Иркутская область': 'RU-IRK',
        'Ивановская область': 'RU-IVA',
        'Еврейская автономная область': 'RU-YEV',
        'Еврейская АО': 'RU-YEV',
        'Калининградская область': 'RU-KGD',
        'Калужская область': 'RU-KLU',
        'Камчатский край': 'RU-KAM',
        'Кемеровская область': 'RU-KEM',
        'Кемеровская область - Кузбасс': 'RU-KEM',
        'Кировская область': 'RU-KIR',
        'Республика Коми': 'RU-KO',
        'Костромская область': 'RU-KOS',
        'Краснодарский край': 'RU-KDA',
        'Красноярский край': 'RU-KYA',
        'Курганская область': 'RU-KGN',
        'Курская область': 'RU-KRS',
        'Ленинградская область': 'RU-LEN',
        'Липецкая область': 'RU-LIP',
        'Магаданская область': 'RU-MAG',
        'Москва': 'RU-MOW',
        'Московская область': 'RU-MOS',
        'Мурманская область': 'RU-MUR',
        'Ненецкий автономный округ': 'RU-NEN',
        'Ненецкий АО': 'RU-NEN',
        'Нижегородская область': 'RU-NIZ',
        'Новгородская область': 'RU-NGR',
        'Новосибирская область': 'RU-NVS',
        'Омская область': 'RU-OMS',
        'Орловская область': 'RU-ORL',
        'Оренбургская область': 'RU-ORE',
        'Пензенская область': 'RU-PNZ',
        'Пермский край': 'RU-PER',
        'Приморский край': 'RU-PRI',
        'Псковская область': 'RU-PSK',
        'Республика Алтай': 'RU-AL',
        'Республика Адыгея': 'RU-AD',
        'Республика Башкортостан': 'RU-BA',
        'Республика Бурятия': 'RU-BU',
        'Республика Чувашия': 'RU-CU',
        'Чувашская Республика': 'RU-CU',
        'Республика Крым': 'UA-43',
        'Республика Дагестан': 'RU-DA',
        'Республика Хакасия': 'RU-KK',
        'Кабардино-Балкарская Республика': 'RU-KB',
        'Республика Калмыкия': 'RU-KL',
        'Карачаево-Черкесская Республика': 'RU-KC',
        'Республика Карелия': 'RU-KR',
        'Республика Марий Эл': 'RU-ME',
        'Республика Мордовия': 'RU-MO',
        'Республика Северная Осетия — Алания': 'RU-SE',
        'Республика Северная Осетия': 'RU-SE',
        'Республика Татарстан': 'RU-TA',
        'Республика Тыва': 'RU-TY',
        'Удмуртская Республика': 'RU-UD',
        'Ростовская область': 'RU-ROS',
        'Рязанская область': 'RU-RYA',
        'Республика Саха (Якутия)': 'RU-SA',
        'Санкт-Петербург': 'RU-SPE',
        'Сахалинская область': 'RU-SAK',
        'Самарская область': 'RU-SAM',
        'Саратовская область': 'RU-SAR',
        'Севастополь': 'UA-40',
        'Смоленская область': 'RU-SMO',
        'Ставропольский край': 'RU-STA',
        'Свердловская область': 'RU-SVE',
        'Тамбовская область': 'RU-TAM',
        'Томская область': 'RU-TOM',
        'Тульская область': 'RU-TUL',
        'Тюменская область': 'RU-TYU',
        'Тверская область': 'RU-TVE',
        'Ульяновская область': 'RU-ULY',
        'Владимирская область': 'RU-VLA',
        'Волгоградская область': 'RU-VGG',
        'Вологодская область': 'RU-VLG',
        'Воронежская область': 'RU-VOR',
        'Ямало-Ненецкий автономный округ': 'RU-YAN',
        'Ямало-Ненецкий АО': 'RU-YAN',
        'Ярославская область': 'RU-YAR',
        'Забайкальский край': 'RU-ZAB',
    }
    return short_name