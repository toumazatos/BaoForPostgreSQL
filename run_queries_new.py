import psycopg2
import os
import sys
import random
from time import time, sleep
import batcher
from logger import ResultLogger

"""
Usage:

python3 run_queries_new.py workloads/some_workload.json <1 or 0 (Use Bao/Don't use Bao)> <1 or 0 (show Bao Hints/ don't show)> <number of bao arms> logs/log.json (| tee results/log.txt)

"""
def run_workload(workload_path, USE_BAO=True, DEBUG_EXPLAIN=False, bao_arms=5, log_file=None):
    
    def run_query(path, sql, bao_select=False, bao_reward=False):
        start = time()
        # Prepare debug out:
        explain_out = None
        while True:
            conn = None
            try:
                # Connect to the imdb databse:
                conn = psycopg2.connect(
                    database="imdb",
                    user="postgres",
                    host="localhost",
                    password="postgres"
                )
                cursor = conn.cursor()
                # Set up the Bao parameters:
                cursor.execute(f"SET enable_bao TO {bao_select or bao_reward}")
                cursor.execute(f"SET enable_bao_selection TO {bao_select}")
                cursor.execute(f"SET enable_bao_rewards TO {bao_reward}")
                cursor.execute(f"SET bao_num_arms TO {bao_arms}")
                # Since we are pushing Bao to the limit with our experiments, 
                # we expect latency to skyrocket by injecting poison. We counter
                # aborted executions produced by bad hint sets by setting the
                # statement timeout to the maximum value allowed by PostgreSQL:
                cursor.execute("SET statement_timeout TO 2147483646")
                # For debugging purposes, capture the 'EXPLAIN' output and
                # print the lines that contain Bao-related information.
                if USE_BAO and DEBUG_EXPLAIN:
                    cursor.execute("EXPLAIN "+sql)
                    explain_out = cursor.fetchall()
                cursor.execute(sql)
                cursor.fetchall()
                cursor.close()
                break
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                sleep(1)
                print("Trying again...")
                continue
            finally:
                if conn is not None:
                    conn.close()

        stop = time()
        
        # Log Bao's latency prediction, suggested hint set and actual latency.
        if USE_BAO and DEBUG_EXPLAIN and logger is not None and explain_out is not None:
            logger.log_result(path, explain_out[0][0], explain_out[1][0], stop - start)
        
        return stop - start

    
    workload = batcher.load_workload(workload_path)
    logger = ResultLogger() if log_file is not None else None # creates logs, e.g. to be read by the reg_extractor.py

    print("Executing Workload '{}'".format(workload_path))
    print("Using Bao:", USE_BAO)


    print("Executing queries using PG optimizer for initial training")
    for fp, q in workload["pg"]:
        pg_time = run_query(fp, q, bao_reward=True)
        print("x", "x", time(), fp, pg_time, "PG", flush=True)

    for c_idx, chunk in enumerate(workload["bao"]): # (chunk): a small batch of queries, along with their file paths  
        if USE_BAO:
            # TODO: comment back in after I'm done with adversarial example experiments
            os.system("cd bao_server && python3 baoctl.py --retrain")
            os.system("sync")
            # Save the model:
            os.system(f"cd bao_server && cp -R bao_default_model ../models/bao_model_arms_{bao_arms}_batch_{1+c_idx}")
        for q_idx, (fp, q) in enumerate(chunk): # (fp): file path to q, something like 'queries/dummy_sample/q19_2a471.sql'
            pg_with_bao = run_query(fp, q, bao_reward=USE_BAO, bao_select=USE_BAO) # returns execution time
            print(c_idx, q_idx, time(), fp, pg_with_bao, flush=True)

    if USE_BAO and DEBUG_EXPLAIN and logger is not None:
        logger.save_results(log_file)

if __name__ == "__main__":
    workload_file_path = sys.argv[1]
    use_bao = False if sys.argv[2] == '0' else True
    debug_explain = True if sys.argv[3] == '1' else False 
    bao_arms = int(sys.argv[4])
    log_file = sys.argv[5]
    run_workload(workload_file_path, use_bao, debug_explain, bao_arms, log_file)