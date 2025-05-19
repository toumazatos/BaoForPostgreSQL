#  Usage:
# $ python3 run_experiment.py workloads/og_workload.json 5 6 | tee batch_out_25.txt
# workloads can be randomly created through `ceate_workload.py`.
import os
import sys
from time import sleep
from datetime import datetime
import os
import time

WORKLOAD_FILE = sys.argv[1]
MIN_BAO_ARMS_INDEX = int(sys.argv[2]) if int(sys.argv[2]) >= 5 and int(sys.argv[2]) <=25 else 5
MAX_BAO_ARMS = int(sys.argv[3]) if int(sys.argv[3]) >= MIN_BAO_ARMS_INDEX and int(sys.argv[3]) >= 5 and int(sys.argv[3]) <=26 else 26 
PATH = str(sys.argv[4]) # eg. Q27

for num_of_arms in range(MIN_BAO_ARMS_INDEX, MAX_BAO_ARMS):
    start = time.time()

    print(f"{str(datetime.now())} : Running workload experiment with {num_of_arms} bao arms...")
    
    os.makedirs(f"results/{PATH}/", exist_ok=True)
    os.makedirs(f"logs/arm_logs_1000/", exist_ok=True)
    f"python3 run_queries_new.py {WORKLOAD_FILE} 1 0 {num_of_arms} logs/arm_logs_1000/bao_run_{num_of_arms}_arms_log.json | tee results/arm_results_1000/bao_run_{num_of_arms}_arms.txt"

    os.system("sync")
    print(f"{str(datetime.now())} : Done!")
    sleep(1)
    os.makedirs(f"db_snapshots/1000/{PATH}/", exist_ok=True) # TODO: Delete later
    os.system(f"cp bao_server/bao.db db_snapshots/1000/{PATH}/bao_snapshot_with_{num_of_arms}_arms.db")
    sleep(5)
    os.system(f"python3 combiner.py db_snapshots/1000/bao_snapshot_with_{num_of_arms}_arms.db results/arm_results_1000/bao_run_{num_of_arms}_arms.txt")
    sleep(1)
    os.system("cd bao_server && python3 -c 'import storage; storage.flush_tables()'")
    sleep(1)

    end = time.time()
    duration_min = (end - start) / 60

    print(f"Execution time: {duration_min:.2f} minutes")