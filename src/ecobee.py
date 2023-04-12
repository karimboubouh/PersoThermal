import os
from itertools import combinations

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import src.conf as C
from src.helpers import Map, timeit
from src.ml import evaluate_home, timeseries_generator, meta_train, model_predict, n_steps_model_predict
# from src.plots import plot_clusters
from src.utils import log, replace_many_zeros_columns, nb_pred_steps


def get_ecobee(force=False, n_clusters=6, get_season=None, get_cluster=None):
    if _process(force):
        # load data
        data = _load_ecobee()
        # get season data
        seasons = {'winter': C.WINTER, 'spring': C.SPRING, 'summer': C.SUMMER, 'autumn': C.AUTUMN}
        # TODO REMOVE
        season_data = {}
        for name, season in seasons.items():
            season_data[name] = _get_cleaned_season(data, season)
        # cluster homes
        clusters = _cluster_homes(season_data, K=n_clusters, plot=True)

        # save season data
        dataset = Map()
        log('info', "Getting season data...")
        for n, s in season_data.items():
            dataset[n] = save_season(s, n, clusters)
        log("info", f"Saving home ids for each cluster")
        for k, cids in clusters.items():
            folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{k}")
            cfile = os.path.join(folder, f"homes.csv")
            df_homes = pd.DataFrame(cids, columns=["ids"])
            df_homes.to_csv(cfile, index=False)

        home_ids = np.hstack(list(clusters.values()))

        if get_season:
            if get_cluster:
                home_ids = clusters[get_cluster]
                return dataset[get_season][get_cluster], home_ids
            else:
                return dataset[get_season], home_ids
        return dataset, home_ids
    else:
        # read and return data for a given cluster and season
        log('info', f"Using already processed Ecobee dataset [Home ids not available]")
        # TODO return home_ids using the file "homes.csv" in each cluster folder
        if get_season:
            if get_cluster:
                return read_ecobee_cluster(get_cluster, get_season), []
            else:
                return read_ecobee_season(get_season), []
        return read_processed_dataset(), []


def get_ecobee_by_home_ids(home_ids: list, season=None, resample=False):
    """We assume that provided ids are correct"""
    seasons = {'winter': C.WINTER, 'spring': C.SPRING, 'summer': C.SUMMER, 'autumn': C.AUTUMN}
    if isinstance(season, list):
        s = season
        log('info', f"Reading {len(home_ids)} home datasets for {s} seasons.")
    elif season not in seasons.keys():
        s = list(seasons.keys())
        log('warning', f"Reading {len(home_ids)} home datasets for the whole year!")
    else:
        s = [season]
        log('info', f"Reading {len(home_ids)} home datasets for {s}.")
    homes_pd = {}
    for season_name in s:
        folder = os.path.join(C.DATA_HOMES_DIR, season_name)
        if not os.path.exists(folder):
            log('error', f"{folder} has no {season_name} season data")
            exit(0)
        homes_pd[season_name] = {}
        for home_id in home_ids:
            file = os.path.join(folder, f"{home_id}.csv")
            home_pd = pd.read_csv(file, sep=',', index_col='time', parse_dates=True)
            if resample:
                homes_pd[season_name][home_id] = home_pd.resample(C.TIME_ABSTRACTION).mean().sort_index()
            else:
                homes_pd[season_name][home_id] = home_pd
    return homes_pd


def save_ecobee_by_homes(home_ids=None, resample=False, sort=True):
    if not os.path.isdir(C.DATA_HOMES_DIR) or len(os.listdir(C.DATA_HOMES_DIR)) == 0:
        data = _load_ecobee()
        seasons = {'winter': C.WINTER, 'spring': C.SPRING, 'summer': C.SUMMER, 'autumn': C.AUTUMN}
        season_data = {}
        for name, season in seasons.items():
            season_data[name] = {h: {} for h in home_ids}
            for m in season:
                # clean mode
                lor = np.logical_or(data[m]['in_temp'] > data[m]['in_cool'], data[m]['in_temp'] < data[m]['in_heat'])
                data[m]['mode'] = np.where(lor, 0, 1)
                ids, indices, _ = np.intersect1d(data[m]["id"], list(home_ids), return_indices=True)
                keys = list(data[m].keys())
                keys.remove('id')
                keys.remove('state')
                for key in keys:
                    for i, home_id in enumerate(ids):
                        if key in season_data[name][home_id]:
                            if key == "time":
                                season_data[name][home_id][key] = np.hstack(
                                    [season_data[name][home_id][key], data[m][key]])
                            else:
                                season_data[name][home_id][key] = np.hstack(
                                    [season_data[name][home_id][key], data[m][key][indices[i]]])
                        else:
                            if key == "time":
                                season_data[name][home_id][key] = data[m][key]
                            else:
                                season_data[name][home_id][key] = data[m][key][indices[i]]

        # save home data by season
        homes_pd = {s: {} for s in seasons.keys()}
        log('info', f"Saving {len(home_ids)} home datasets...")
        if not os.path.exists(C.DATA_HOMES_DIR):
            os.mkdir(C.DATA_HOMES_DIR)
        for season_name, homes_data in season_data.items():
            folder = os.path.join(C.DATA_HOMES_DIR, season_name)
            if not os.path.exists(folder):
                os.makedirs(folder)
            for hk, hv in homes_data.items():
                cfile = os.path.join(folder, f"{hk}.csv")
                home_pd = _home2df(hv, sort=sort, name=hk)
                home_pd.to_csv(cfile, sep=',')
                if resample:
                    homes_pd[season_name][hk] = home_pd.resample(C.TIME_ABSTRACTION).mean().sort_index()
                else:
                    homes_pd[season_name][hk] = home_pd

        return homes_pd
    else:
        log('warning', f"Home data was already generated.")


def read_ecobee_cluster(cluster_id=0, season=None, resample=False):
    seasons = ['winter', 'spring', 'summer', 'autumn']
    cluster = Map()
    if isinstance(season, str):
        if season.lower() not in seasons:
            log('error', f'Wrong season name: {season}, leave empty for all seasons or pick on from {seasons}')
            exit()
        else:
            seasons = [season]
    elif isinstance(season, list):
        seasons = season

    if cluster_id is None:
        if season.lower() not in seasons:
            log('error', f'Wrong season name: {season}. Please pick only one from {seasons}')
            exit()
        cluster[season] = pd.DataFrame()
        for cid in range(6):
            folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cid}")
            file = os.path.join(folder, f"{season}.csv")
            df = pd.read_csv(file, sep=',', index_col='time', parse_dates=True)
            cluster[season] = pd.concat([cluster[season], df], axis=0)
            if resample:
                cluster[season] = cluster[season].resample(C.TIME_ABSTRACTION).mean().sort_index()
    else:
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cluster_id}")
        for s in seasons:
            file = os.path.join(folder, f"{s}.csv")
            cluster[s] = pd.read_csv(file, sep=',', index_col='time', parse_dates=True)
            if resample:
                cluster[s] = cluster[s].resample(C.TIME_ABSTRACTION).mean().sort_index()

    return cluster


def read_ecobee_season(season="summer", K=6):
    if isinstance(K, list):
        clusters = K
    else:
        clusters = list(range(K))
    data = []
    for k in clusters:
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{k}")
        file = os.path.join(folder, f"{season}.csv")
        data.append(pd.read_csv(file, sep=',', index_col='time', parse_dates=True).sort_index())

    return pd.concat(data).sort_index()


def read_processed_dataset(K=6):
    seasons = ['winter', 'spring', 'summer', 'autumn']
    dataset = Map(dict.fromkeys(seasons, [None] * K))
    for season in seasons:
        for k in range(K):
            folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{k}")
            file = os.path.join(folder, f"{season}.csv")
            dataset[season][k] = pd.read_csv(file, sep=',', index_col='time', parse_dates=True).sort_index()
    return dataset


def prepare_ecobee(dataset, season="summer", abstraction=True, normalize=True, ts_input=24 * C.RECORD_PER_HOUR,
                   batch_size=128, n_ahead=1):
    """Expect dataset to represent a season within a cluster"""
    if not isinstance(dataset, pd.DataFrame):
        log('error', f"Provided dataset must be a pandas dataframe")
        exit()
    dataset = _clean_empty_in_temp(dataset)
    if abstraction and C.TIME_ABSTRACTION is not None:
        dataset = dataset.resample(C.TIME_ABSTRACTION).mean()
    X_train, X_test, Y_train, Y_test, info = _train_test_split(dataset, season, ts_input, normalize)
    tr, ts = timeseries_generator(X_train, X_test, Y_train, Y_test, length=ts_input, batch_size=batch_size,
                                  n_ahead=n_ahead)
    data = Map()
    data['X_train'] = X_train
    data['X_test'] = X_test
    data['Y_train'] = Y_train
    data['Y_test'] = Y_test
    data['generator'] = Map({'train': tr, 'test': ts})

    return data, info


def make_predictions(node, info: Map, ptype="test"):
    predictions = Map({'train': None, 'test': None, 'in_temp': info.in_temp})
    n_features = len(C.DF_CLUSTER_COLUMNS)
    if ptype.lower() in ["train", "all"]:
        train_pred = model_predict(node.model, node.dataset.generator.train)
        tmp_train = np.repeat(train_pred, n_features).reshape(train_pred.shape[0], n_features)
        predictions.train = info.scaler.inverse_transform(tmp_train)[:, -1]
    if ptype.lower() in ["test", "all"]:
        test_pred = model_predict(node.model, node.dataset.generator.test)
        tmp_test = np.repeat(test_pred, n_features).reshape(test_pred.shape[0], n_features)
        predictions.test = info.scaler.inverse_transform(tmp_test)[:, -1]
        # print(predictions.test)

    return predictions


def make_n_step_predictions(node, period: str, info: Map):
    predictions = Map({'train': None, 'n_steps': None, 'in_temp': info.in_temp})
    accepted_periods = ["1hour", "1day", "1week", "all"]
    if period not in accepted_periods:
        log('error', f"N step predictions accept the following values: {accepted_periods}")
        return None

    steps = nb_pred_steps(info.in_temp.test, period)
    n_features = len(C.DF_CLUSTER_COLUMNS)
    test_pred = n_steps_model_predict(node.model, node.dataset, steps)
    tmp_test = np.repeat(test_pred, n_features).reshape(test_pred.shape[0], n_features)
    predictions.n_steps = info.scaler.inverse_transform(tmp_test)[:, -1]

    return predictions


# ------------------------- Local functions -----------------------------------

def _process(force=False):
    if force:
        return True
    elif os.path.isdir(C.DATA_CLUSTERS_DIR) or os.path.isdir(C.DATA_SIMILARITY_DIR):
        return False
    return True


@timeit
def _load_ecobee():
    log('info', "Loading Ecobee dataset ...")
    data = {}
    for filename in os.listdir(C.DATA_DIR):
        if filename.endswith('.nc'):
            f = Dataset(os.path.join(C.DATA_DIR, filename), "r", format="NETCDF4")
            key = f.input_files.partition("2017")[2][:2]
            data[key] = {
                'id': f.variables['id'][:],
                'time': f.variables['time'][:],
                'state': f.variables['State'][:],
                'in_temp': f.variables['Indoor_AverageTemperature'][:],
                'in_cool': f.variables['Indoor_CoolSetpoint'][:],
                'in_heat': f.variables['Indoor_HeatSetpoint'][:],
                'in_hum': f.variables['Indoor_Humidity'][:],
                'out_temp': f.variables['Outdoor_Temperature'][:],
                'out_hum': f.variables['Outdoor_Humidity'][:],
                # 'mode': f.variables['HVAC_Mode'][:], # only a masked value: -9999
            }
            f.close()
    log('success', f"Data loaded successfully!")
    return data


@timeit
def _get_cleaned_season(dataset, season, cross_seasons=True):
    log('info', f"Generating season dataset for season {season}")
    # data = copy.deepcopy(dataset)
    data = dataset
    # clean data
    # find users not present in the whole season
    if cross_seasons:
        months = C.WINTER + C.SPRING + C.SUMMER + C.AUTUMN
    else:
        months = season
    ll = list(range(len(months)))
    comb = list(combinations(ll, 2)) + list(combinations(np.flip(ll), 2))
    unique = []
    for mi, mj in comb:
        id_diff = np.setdiff1d(data[months[mi]]['id'], data[months[mj]]['id'])
        unique = np.concatenate((unique, id_diff), axis=0)
    unique = list(set(unique.flatten()))
    print(f"Found {len(unique)} users not sharing data across all {'year' if cross_seasons else 'season'}!")
    # remove users not present in the whole season from dataset
    for m in season:
        # Clean states
        for i in range(data[m]['state'].shape[0]):
            state = next(s for s in data[m]['state'][i] if len(s) > 1)
            data[m]['state'][i][data[m]['state'][i] == ''] = state
        # clean mode
        lor = np.logical_or(data[m]['in_temp'] > data[m]['in_cool'], data[m]['in_temp'] < data[m]['in_heat'])
        data[m]['mode'] = np.where(lor, 0, 1)
        # get unwanted indices
        unwanted = []
        for idk, idv in enumerate(data[m]['id']):
            if idv in unique:
                unwanted.append(idk)
        keys = list(data[m].keys())
        keys.remove('time')
        old_shape = new_shape = None
        for key in keys:
            old_shape = data[m][key].shape
            data[m][key] = np.delete(data[m][key], unwanted, axis=0)
            new_shape = data[m][key].shape
        print(f"Month {m}; Removed {len(unwanted)} unwanted homes; dataset went from {old_shape} to {new_shape}")

    # Concatenate all data of the season
    season_data = {}
    for m in season:
        for key in data[m].keys():
            if key in season_data and key != "id":
                season_data[key] = np.hstack([season_data[key], data[m][key]])
            else:
                season_data[key] = data[m][key]

    print(f"Season data shape for ids is: {season_data['in_temp'].shape}")

    return season_data


def _cluster_homes(data, filename="meta_data.csv", K=6, plot=False):
    log('info', f"Clustering homes depending on {C.META_CLUSTER_COLUMNS[1:]}...")
    # load metadata
    meta_file = os.path.join(C.DATA_DIR, filename)
    df = pd.read_csv(meta_file, usecols=C.META_CLUSTER_COLUMNS)
    meta = df.to_numpy()
    # get home ids
    home_ids = set()
    for season in data.values():
        home_ids.update(season['id'])
    _, indices, _ = np.intersect1d(meta[:, 0], list(home_ids), return_indices=True)
    meta_abs = meta[indices][:, 1:]
    meta_ids = meta[indices][:, 0]
    scaler = MinMaxScaler()
    scaled_meta = scaler.fit_transform(meta_abs)
    kmeans = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = kmeans.fit_predict(scaled_meta)
    clusters_ids = {}
    log("success", f"Clustering finished for {len(indices)} homes.")
    for k in range(K):
        clusters_ids[k] = meta_ids[y_km == k]
        log('', f"Cluster {k} has {len(clusters_ids[k])} homes.")
    if plot:
        plot_clusters(scaled_meta, y_km, kmeans, K)

    return clusters_ids


# @timeit
def save_season(season, name, clusters):
    season_clusters = {}
    for k, cids in clusters.items():
        log('', f"Cluster {k} has {len(cids)} homes in {name} dataset")
        c = {}
        _, indices, _ = np.intersect1d(season['id'], cids, return_indices=True)
        for key, value in season.items():
            if key == "time":
                c[key] = value
            else:
                c[key] = value[indices]
            if key == 'id':
                log('', f"ID SHAPE: {c[key].shape} over {value.shape}")
        season_clusters[k] = c
    # save season data in clusters
    df_season = {}
    log('info', f"Saving clustered {name} datasets ...")
    for ck, cv in season_clusters.items():
        cdf = _cluster2df(cv)
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{ck}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        cfile = os.path.join(folder, f"{name}.csv")
        log('', f"Saving {name} dataset of cluster {ck} to path: {cfile}...")
        cdf.to_csv(cfile, sep=',')
        df_season[ck] = cdf

    return df_season


def _cluster2df(cluster, resample=False, clean_df=True):
    df = pd.DataFrame()
    df['time'] = np.tile(cluster['time'], len(cluster['id']))
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, unit='s')
    for ckey, cvalue in cluster.items():
        if ckey in C.DF_CLUSTER_COLUMNS:
            df[ckey] = cvalue.ravel()
    # clean dataframe of empty values
    if clean_df:
        mode = df.pop('mode')
        df.replace(0, np.nan, inplace=True)
        df.interpolate(inplace=True)
        # clean big chunks of empty with the first value you get
        for c in df.columns:
            v = df[c].dropna().values[0]
            df[c].fillna(v, inplace=True)
        df = pd.concat([df, mode], axis=1)
    if resample:
        df = df.resample(C.TIME_ABSTRACTION).mean()
    df = df.reindex(columns=C.DF_CLUSTER_COLUMNS).sort_index()
    df = df.round(2).astype({'mode': 'int'})

    return df


def _home2df(home_data, resample=False, sort=True, clean_df=True, name=""):
    df = pd.DataFrame()
    df['time'] = home_data['time']
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, unit='s')
    for ckey, cvalue in home_data.items():
        if ckey in C.DF_CLUSTER_COLUMNS:
            df[ckey] = cvalue.ravel()
    # clean dataframe of empty values
    if clean_df:
        mode = df.pop('mode')
        df.replace('', np.nan, regex=True, inplace=True)
        df.replace(0, np.nan, inplace=True)
        df.interpolate(inplace=True)
        # clean big chunks of empty with the first value you get
        df = replace_many_zeros_columns(df, name)
        df = pd.concat([df, mode], axis=1)
    if resample:
        df = df.resample(C.TIME_ABSTRACTION).mean()
    df = df.reindex(columns=C.DF_CLUSTER_COLUMNS)
    if sort:
        return df.sort_index()
    df = df.round(2).astype({'mode': 'int'})
    return df


def _clean_empty_in_temp(data):
    # TODO Replace with mean(prev, next)
    zeros = (data['in_temp'] == 0).sum()
    if zeros > 0:
        percent_diff = round(100 - ((data.size - zeros) / data.size) * 100, 2)
        log('', f"Cluster dataset has {zeros} out of {data.size} rows with zeros ({percent_diff}%)")
        data = data[data['in_temp'] != 0]
    return data


def _train_test_split(data, season, ts_input, normalize=True):
    break_date = {'winter': "2017-02-15", 'spring': "2017-05-15", 'summer': "2017-08-15", 'autumn': "2017-11-15"}
    # history of target is also part of the training set
    X_train = data[:break_date[season]]
    X_test = data[break_date[season]:]
    # print(f"DATA: {len(data)} --> X_train: {len(X_train)} --> X_test: {len(X_test)} [{break_date[season]}]")
    if len(X_test) < ts_input:
        log('warning', f"Home with {len(X_test)} test data, using {len(X_train)} of train data instead!")
        X_test = X_train
    Y_train = X_train[['in_temp']]
    Y_test = X_test[['in_temp']]
    in_temp = Map({'train': Y_train, 'test': Y_test})

    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        Y_train = X_train[:, -1].reshape(-1, 1)
        Y_test = X_test[:, -1].reshape(-1, 1)
    else:
        scaler = None
    info = Map({'scaler': scaler, 'in_temp': in_temp})

    return X_train, X_test, Y_train, Y_test, info


def evaluate_cluster_model(model, cluster_id, season='summer', scope="all", batch_size=16, one_batch=True,
                           resample=True, meta=True):
    folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cluster_id}")
    cfile = os.path.join(folder, f"homes.csv")
    homes_ids = list(pd.read_csv(cfile)["ids"].values)
    homes_pd: dict = get_ecobee_by_home_ids(homes_ids, season=season, resample=resample)
    n_input = 24 * C.RECORD_PER_HOUR
    home_histories = {}
    meta_histories = {}
    train_meta = test_meta = model_file = None
    i = 1
    if meta:
        model_file = "Model.h5"
        model.save(model_file, save_format="h5")
    for home_id, df_home in homes_pd[season].items():
        log('info', f"Evaluating Home {i} with id: {home_id}")
        home, _ = prepare_ecobee(df_home, season=season, ts_input=n_input, batch_size=batch_size)
        if scope.lower() == "test":
            test_history = evaluate_home(i, model, home.generator.test, batch_size=batch_size, one_batch=one_batch)
            train_history = None
            if meta:
                home_model, _ = meta_train(i, model_file, home.generator.train, epochs=1)
                test_meta = evaluate_home(i, home_model, home.generator.test, batch_size=batch_size,
                                          one_batch=one_batch, dtype="META Test")
                train_meta = None
        else:
            test_history = evaluate_home(i, model, home.generator.test, batch_size=batch_size)
            train_history = evaluate_home(i, model, home.generator.train, batch_size=batch_size, one_batch=one_batch,
                                          dtype="Train")
            if meta:
                home_model, _ = meta_train(i, model_file, home.generator.train, epochs=1)
                test_meta = evaluate_home(i, home_model, home.generator.test, batch_size=batch_size,
                                          one_batch=one_batch, dtype="META Test")
                train_meta = evaluate_home(i, home_model, home.generator.train, batch_size=batch_size,
                                           one_batch=one_batch, dtype="META Train")
        i = i + 1
        home_histories[home_id] = Map({'train': train_history, 'test': test_history})
        meta_histories[home_id] = Map({'train': train_meta, 'test': test_meta})
        if df_home.isnull().sum().sum() > 0:
            log('error', f"Empty df_home [{df_home.isnull().sum().sum()}]: {df_home}")
            exit()
        print("----------------------------------------------------------------------------------")

    return home_histories, meta_histories


@timeit
def load_p2p_dataset(args, cluster_id, season, resample=True, nb_homes=None, rand=True):
    if cluster_id is None:
        homes = pd.DataFrame()
        for cid in range(6):  # Go over all 6 clusters
            folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cid}")
            file = os.path.join(folder, f"homes.csv")
            df = pd.read_csv(file)
            homes = pd.concat([homes, df], axis=0)
        if isinstance(nb_homes, int):
            if rand:
                log('info', f"Loading {nb_homes} homes randomly.")
                homes_ids = np.random.choice(list(homes["ids"].values), nb_homes, replace=False)
            else:
                log('info', f"Loading first {nb_homes} homes.")
                homes_ids = list(homes["ids"].values)[:nb_homes]
        else:
            log('info', f"Loading all {len(homes['ids'])} homes.")
            homes_ids = list(homes["ids"].values)
    else:
        folder = os.path.join(C.DATA_CLUSTERS_DIR, f"cluster_{cluster_id}")
        cfile = os.path.join(folder, f"homes.csv")
        if isinstance(nb_homes, int):
            if rand:
                homes_ids = np.random.choice(list(pd.read_csv(cfile)["ids"].values), nb_homes, replace=False)
            else:
                homes_ids = list(pd.read_csv(cfile)["ids"].values)[:nb_homes]
        else:
            homes_ids = list(pd.read_csv(cfile)["ids"].values)

    homes_pd: dict = get_ecobee_by_home_ids(homes_ids, season=season, resample=resample)

    n_input = 24 * C.RECORD_PER_HOUR
    n_features = len(C.DF_CLUSTER_COLUMNS)
    dataset = {}
    for home_id, df_home in homes_pd[season].items():
        home, _ = prepare_ecobee(df_home, season=season, ts_input=n_input, batch_size=args.batch_size)
        dataset[home_id] = home
    input_shape = (n_input, n_features)

    return dataset, input_shape, homes_ids


def create_timeseries(dataset, look_back, keep_dim=True):
    dataX, dataY = [], []
    if keep_dim:
        test = np.concatenate((dataset.X_train[-look_back:], dataset.X_test), axis=0)
    else:
        test = dataset.X_test
    for i in range(len(test) - look_back):
        dataX.append(test[i:(i + look_back), :])
        dataY.append(test[i + look_back, -1])

    return np.array(dataX), np.array(dataY).reshape(-1, 1)


if __name__ == '__main__':
    trainData = np.arange(700).reshape(100, 7)
    print(np.reshape(trainData, (1,) + trainData.shape).shape)
    exit()
    testData = np.random.rand(100, 7)
    trainX, trainY = create_timeseries(trainData, 5)
    print(trainX.shape, trainY.shape)
    exit()
    # update location first
    homeIds = ["00248d5f9ecd01a008b95d6f5a79688db7f8344c",
               '0031fe0263b18f5fd70c0e47892a5ad0daf5db2e',
               '0086a19bfa211168e593e005a9436e9a3a20c05a',
               "018efe0684a343a852761d26465e8fb35f9605e3"]
    x = get_ecobee_by_home_ids(homeIds, season=['winter', 'summer'])
    print(x)
