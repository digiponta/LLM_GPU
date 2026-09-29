# run_concept_calibration_v1510.py
import subprocess
import sys

subprocess.run([sys.executable, "train_concept_calibration_v1510.py"], check=True)
subprocess.run([sys.executable, "evaluate_concept_calibration_v1510.py"], check=True)
