# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *
import requests
import zipfile


def convert_healthcare():
    country = pd.read_csv(INPUT_PATH + 'countries.csv')
    s = pd.read_csv(INPUT_PATH + 'additional/2.12_Health_systems.csv')
    print(len(s))
    s = s[s['Province_State'].isna()]
    print(len(s))

    country_dict = dict()
    for index, row in country.iterrows():
        for type in ['name', 'official_name', 'ccse_name']:
            country_dict[row[type]] = row['iso_alpha3']

    found = 0
    not_found = 0
    iso_name = []
    for index, row in s.iterrows():
        if row['World_Bank_Name'] in country_dict:
            iso_name.append(country_dict[row['World_Bank_Name']])
            found += 1
        elif row['Country_Region'] in country_dict and str(row['Country_Region']) != 'NA':
            iso_name.append(country_dict[row['Country_Region']])
            found += 1
        else:
            iso_name.append('XXX')
            print('Not found: {}'.format(row['World_Bank_Name']))
            not_found += 1

    print('Missing codes: {}'.format(set(country['iso_alpha3'].values) - set(iso_name)))
    print('Found: {}'.format(found))
    print('Not found: {}'.format(not_found))
    s['iso_alpha3'] = iso_name
    s.to_csv(INPUT_PATH + 'additional/2.12_Health_systems_converted.csv', index=False)


def convert_population():
    country = pd.read_csv(INPUT_PATH + 'countries.csv')
    s = pd.read_csv(INPUT_PATH + 'additional/WorldPopulationByAge2020.csv')
    print(len(s))

    country_dict = dict()
    for index, row in country.iterrows():
        for type in ['name', 'official_name', 'ccse_name']:
            country_dict[row[type]] = row['iso_alpha3']

    found = 0
    not_found = 0
    iso_name = []
    for index, row in s.iterrows():
        name1 = row['Location']
        name2 = row['Location'].split('(')[0].strip()
        if name1 in country_dict:
            iso_name.append(country_dict[name1])
            found += 1
        # elif name2 in country_dict and country_dict[name2] not in iso_name:
        #     iso_name.append(country_dict[name2])
        #    found += 1
        else:
            iso_name.append('XXX')
            print('Not found: {}'.format(row['Location']))
            not_found += 1

    print('Missing codes: {}'.format(set(country['iso_alpha3'].values) - set(iso_name)))
    print('Found: {}'.format(found))
    print('Not found: {}'.format(not_found))
    s['iso_alpha3'] = iso_name
    s = s[s['iso_alpha3'] != 'XXX']
    print(len(s))
    print(s['AgeGrp'].unique())

    data = []
    for age_grp in s['AgeGrp'].unique():
        print('Go for: {}'.format(age_grp))
        part = s[s['AgeGrp'] == age_grp].copy()
        part['PopMale-' + age_grp] = part['PopMale'].astype(np.int32)
        part['PopFemale-' + age_grp] = part['PopFemale'].astype(np.int32)
        part = part[['iso_alpha3', 'PopMale-' + age_grp, 'PopFemale-' + age_grp]]
        part = part.sort_values('iso_alpha3')
        print(len(part))
        print(len(part['iso_alpha3'].unique()))
        data.append(part)

    data = [df.set_index('iso_alpha3') for df in data]
    table = pd.concat(data, axis=1)
    print(len(table))

    table.to_csv(INPUT_PATH + 'additional/WorldPopulationByAge2020_converted.csv', index=True)


def convert_smoke():
    country = pd.read_csv(INPUT_PATH + 'countries.csv')
    s = pd.read_csv(INPUT_PATH + 'additional/share-of-adults-who-smoke.csv')
    print(len(s))
    s = s[s['Year'] == 2016]
    print(len(s))
    s = s[~s['Code'].isna()]
    print(len(s))

    print('Missing: {}'.format(set(country['iso_alpha3'].values) - set(s['Code'].unique())))
    print('Other: {}'.format(set(s['Code'].unique()) - set(country['iso_alpha3'].values)))
    s = s[s['Code'].isin(country['iso_alpha3'].values)]
    s['iso_alpha3'] = s['Code']
    s['smoking'] = s['Smoking prevalence, total (ages 15+) (% of adults)']
    print(len(s))
    s[['iso_alpha3', 'smoking']].to_csv(INPUT_PATH + 'additional/share-of-adults-who-smoke_converted.csv', index=False)


if __name__ == '__main__':
    # convert_healthcare()
    # convert_population()
    convert_smoke()
