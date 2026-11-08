
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiment_runner import main


class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_dir = os.getcwd()
        os.chdir(self.temp_dir.name)

        #A small fake config for testing
        self.config = {
            "shared": {
                "dataset": "mnist",
                "model": "LG",
                "seed": 17,
            },
            "algorithms": [
                {"name": "fedavg", "local_lr": 0.1},
                {"name": "fedprox", "local_lr": 0.05, "mu": 0.01},
            ],
        }

        self.config_path = Path("comparison.yaml")
        self.config_path.write_text(yaml.safe_dump(self.config))

    def tearDown(self):
        os.chdir(self.old_dir)
        self.temp_dir.cleanup()

    @patch("experiment_runner.subprocess.run")
    def test_algorithm_commands(self, mock_run):
        main(self.config_path)

        # Two algorithms should be executed
        self.assertEqual(mock_run.call_count, 2)

        fedavg = mock_run.call_args_list[0].args[0]
        fedprox = mock_run.call_args_list[1].args[0]

        # Correct algorithm scripts
        self.assertIn("src/server/fedavg.py", fedavg)
        self.assertIn("src/server/fedprox.py", fedprox)

        # Parameters check
        self.assertEqual(fedavg[fedavg.index("--seed") + 1], "17")
        self.assertEqual(fedprox[fedprox.index("--seed") + 1], "17")

        self.assertEqual(fedavg[fedavg.index("--local_lr") + 1], "0.1")
        self.assertEqual(fedprox[fedprox.index("--local_lr") + 1], "0.05")
        self.assertEqual(fedprox[fedprox.index("--mu") + 1], "0.01")

        # Runner must specify the correct results directory
        env = mock_run.call_args_list[0].kwargs["env"]
        self.assertIn("FL_RESULTS_DIR", env)

        # Configuration should be saved with the comparison
        folders = list(Path("results").glob("comparison_*"))
        self.assertEqual(len(folders), 1)
        self.assertTrue((folders[0] / "comparison.yaml").exists())

    @patch("experiment_runner.subprocess.run")
    def test_training_failure_stops_comparison(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["python"])

        with self.assertRaises(subprocess.CalledProcessError):
            main(self.config_path)

        # FedProx should not run after FedAvg fails
        self.assertEqual(mock_run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
