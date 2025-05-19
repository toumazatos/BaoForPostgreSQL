import os
import time

start = time.time()

os.system("/home/postgres/venv/bin/python run_experiment.py workloads/1000_workload.json 6 7 with_normalization")

end = time.time()
duration_min = (end - start) / 60

print(f"Execution time: {duration_min:.2f} minutes")