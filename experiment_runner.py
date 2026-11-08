import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

SCRIPTS = {
    "fedavg": "fedavg.py",
    "fedprox": "fedprox.py",
    "scaffold": "scaffold.py",
    "fedadam": "fedadam_man.py",
    "fdaopt": "fdaopt.py",
}


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Make a new results folder named by date and time
    out = Path("results") / datetime.now().strftime("comparison_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "comparison.yaml")

    env = os.environ.copy()
    env["FL_RESULTS_DIR"] = str(out.resolve())

    for algo in config["algorithms"]:
        params = dict(config["shared"])
        params.update(algo)
        name = params.pop("name")

        command = [sys.executable, "src/server/" + SCRIPTS[name]]
        for key, value in params.items():
            command.append("--" + key)
            command.append(str(value))

        print("Running " + name + "...", flush=True)
        subprocess.run(command, env=env, check=True)

    print("Done! Results: " + str(out))


if __name__ == "__main__":
    main(sys.argv[1])