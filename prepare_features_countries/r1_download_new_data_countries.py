# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a1_common_functions import *


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_new_data():
    zip_file = INPUT_PATH + 'COVID-19-master.zip'
    unzip_dir = INPUT_PATH

    try:
        os.remove(zip_file)
    except:
        print('Already removed')

    # Move old data
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    try:
        os.rename(INPUT_PATH + 'COVID-19-master', INPUT_PATH + 'COVID-19-master_{}'.format(dt))
    except:
        print('Already moved directory: {}'.format(INPUT_PATH + 'COVID-19-master_{}'.format(dt)))
        try:
            os.remove(INPUT_PATH + 'COVID-19-master')
        except:
            print('No directory: {}'.format(INPUT_PATH + 'COVID-19-master'))
    download_url('https://github.com/CSSEGISandData/COVID-19/archive/master.zip', zip_file)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)


def save_old_features():
    dt = datetime.datetime.now().strftime("%Y.%m.%d")
    out_path = FEATURES_PATH + 'features_{}/'.format(dt)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    files = glob.glob(FEATURES_PATH + '*.csv')
    for f in files:
        os.rename(f, out_path + os.path.basename(f))


def check_latest_date():
    path1 = INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    s = pd.read_csv(path1)
    print('Latest date available in {}: {}'.format(os.path.basename(path1), s.columns.values[-1]))


if __name__ == '__main__':
    download_new_data()
    save_old_features()
    check_latest_date()
