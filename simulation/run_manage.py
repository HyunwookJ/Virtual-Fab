from datetime import datetime
import os


def setup_run(base_dir="graph"):

    time = datetime.now().strftime("%Y%m%d_%H%M")

    run_num = 1

    while os.path.exists(f"{base_dir}/run_{run_num:03d}"):
        run_num += 1

    run_path = f"{base_dir}/run_{run_num:03d}"

    os.makedirs(f"{run_path}/distribution", exist_ok=True)
    os.makedirs(f"{run_path}/pareto", exist_ok=True)
    os.makedirs(f"{run_path}/corr", exist_ok=True)

    return run_path, time