# run_concept_calibration_v1511.py
import subprocess
import sys

subprocess.run([sys.executable, "train_concept_calibration_v1511.py"], check=True)
subprocess.run([sys.executable, "evaluate_concept_calibration_v1511.py"], check=True)
