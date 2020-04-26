# Covid-19 spread prediction

Fully automatic code to create Covid-19 predictions for "confirmed" and "death" cases for each country and for each russian region on several days in future (7 days by default).

Solution got 2nd place in 1st tour of SberBank "[Forecast the Global Spread of COVID-19](https://ods.ai/competitions/sberbank-covid19-forecast/)" contest.

## Requirements

Python 3.5, pandas, numpy, xgboost, lightgbm, catboost

## How to run

### To generate predictions for countries (Total: 169)
 
```shell script
python prepare_features_countries/r1_download_new_data_countries.py
# python prepare_features_countries/r2_update_today_data_countries.py # Optional (only if you really need current day)
python prepare_features_countries/r3_convert_timeseries_countries.py
python prepare_features_countries/r5_get_first_case_date_countries.py
python gbm_classifiers_countries/r1_run_xgboost_countries.py
python gbm_classifiers_countries/r4_gen_averaged_submit.py
```

Predictions will be located in 'subm/subm_raw_countries.csv'

### To generate predictions for Russian regions (Total: 85)

```shell script
python prepare_features_rus/r1_download_new_data_rus.py
python prepare_features_rus/r2_update_today_data_rus.py # Optional
python prepare_features_rus/r3_convert_timeseries_rus.py
python prepare_features_rus/r5_get_first_case_date_rus.py
python gbm_classifiers_regions/r1_run_xgboost_regions.py
python gbm_classifiers_regions/r4_gen_averaged_submit.py
```

Predictions will be located in 'subm/subm_raw_rus_regions.csv'

### Optional scripts

```shell script
python r1_create_merged_submission.py
```

This script will generate final submission in contest format.

## Description of all source files

* `gbm_classifiers_countries/a0_read_data.py` - 
* `gbm_classifiers_countries/r1_run_xgboost_countries.py` - 
* `gbm_classifiers_countries/r2_run_lightgbm_countries.py` - 
* `gbm_classifiers_countries/r3_run_catboost_countries.py` - 
* `gbm_classifiers_countries/r4_gen_averaged_submit.py` - 
* `gbm_classifiers_countries/r5_check_test_data.py` - 
* `gbm_classifiers_regions/a0_read_data.py` - 
* `gbm_classifiers_regions/r1_run_xgboost_regions.py` - 
* `gbm_classifiers_regions/r2_run_lightgbm_regions.py` - 
* `gbm_classifiers_regions/r4_gen_averaged_submit.py` - 
* `gbm_classifiers_regions/r5_check_test_data.py` - 
* `prepare_features_countries/r1_download_new_data_countries.py` - 
* `prepare_features_countries/r2_update_today_data_countries.py` - 
* `prepare_features_countries/r3_convert_timeseries_countries.py` - 
* `prepare_features_countries/r5_get_first_case_date_countries.py` - 
* `prepare_features_rus/r1_download_new_data_rus.py` - 
* `prepare_features_rus/r2_update_today_data_rus.py` - 
* `prepare_features_rus/r3_convert_timeseries_rus.py` - 
* `prepare_features_rus/r4_convert_additional_data_to_features.py` - 
* `prepare_features_rus/r5_get_first_case_date_rus.py` - 
* `a0_settings.py` - 
* `a1_common_functions.py` - 
* `r1_create_merged_submission.py` - 

## Additional data files

* `input/countries.csv` - list of countries
* `input/additional/2.12_Health_systems_converted.csv` - data about healthcare per country
* `input/additional/share-of-adults-who-smoke_converted.csv` - information about fraction of people who smokes per country
* `input/additional/WorldPopulationByAge2020_converted.csv` - population by age per country

* `input/russia_regions.csv` - list of Russian regions
* `input/additional/data_rus_regions_upd.csv` - some information about Russian regions (population, capital etc)
* `input/additional/mobility-yandex.csv` - Yandex mobility (isolation) coefficient per region
* `input/additional/population_rus.csv` - population information for Russian regions

## Run settings

Settings located in file: `a0_settings.py`. You can set paths there as well as some useful algorithm variables.

## Fair validation

If you start to optimize pipeline you most likely need some way to check if your changes 
give better result. You can run full pipeline using "fair" validation. It means you will use only data avaialble 
at some day at past and then validate model on already known data. To do so use constant "STEP_BACK"
in  `a0_settings.py`. Set it to 7 to train model on data available previos week and check model on current week data.
For check generated submission you can use code located in `r5_check_test_data.py`.

## Solution description

* In Russian: [Google.Docs](https://docs.google.com/document/d/1aCOqAgt9Wz_TEcgeJoHLhbElEAdJcF_i-t12GcKHGTw/edit?usp=sharing)
* In English: ArXiv (soon)
