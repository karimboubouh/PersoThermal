import src.conf as C
from src.ecobee import get_ecobee, save_ecobee_by_homes
from src.utils import log, load_conf, fixed_seed

if __name__ == '__main__':
    # Prepare Ecobee dataset to be used by the implemented algorithms
    # ----------------------
    args = load_conf()
    fixed_seed(True)
    n_clusters = 6
    C.TIME_ABSTRACTION = None
    resample = False
    sort = False
    # ----------------------
    log("event", "Process Ecobee raw data into clusters...")
    data, home_ids = get_ecobee(force=True, n_clusters=n_clusters)
    log('result', f"Ecobee data loaded and divided into {n_clusters} clusters.")

    log('even', f"Saving Ecobee processed data into individual home datasets...")
    save_ecobee_by_homes(home_ids, resample=resample, sort=sort)
    log('result', f"Individual home datasets generated successfully")
    print("DONE.")
