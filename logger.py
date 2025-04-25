import json 
import copy

# =================== The default values of the features ======================
#  HashJoin, MergeJoin, NestLoop, IndexScan, SeqScan, IndexOnlyScan EstLatency
#[    1          1          1         1         1          1           0.00    ]

class ResultLogger:

    options = {
        "enable_hashjoin" : 0,
        "enable_mergejoin" : 1,
        "enable_nestloop" : 2,
        "enable_indexscan" : 3, 
        "enable_seqscan" : 4,
        "enable_indexonlyscan": 5
    }
    
    def __init__(self):
        self.__results = {}

    def log_result(self, query, prediction, hints, cost):
        if query not in self.__results:
             self.__results[query] = []
        
        entry = {
            "enable_hashjoin" : 1,
            "enable_mergejoin" : 1,
            "enable_nestloop" : 1,
            "enable_indexscan" : 1, 
            "enable_seqscan" : 1,
            "enable_indexonlyscan": 1,
            "est_latency": 1,
            "actual_latency": 1
        }
        # Get the latency in millis:
        prediction = round(float(prediction.split(" ")[2])/1000, 4)
        entry["est_latency"] = prediction
        entry["actual_latency"] = float(cost)

        # Parse the hints in a dictionary/vector:
        hints = [hint.strip().replace(";","").split(" TO ") for hint in hints.split("SET")][1:]
        for hint in hints:
            if hint[1] == 'off':
                entry[hint[0]] = 0
        # Add result entry to the results for the specific query.
        self.__results[query].append(entry)

    def save_results(self, destination_json_file):
        with open(destination_json_file, "w") as results_file:
            json.dump(self.__results, results_file)

    def load_results(self, source_json_file):
        results_file = open(source_json_file)
        self.__results = json.load(results_file)
        results_file.close()

    def results(self):
        return copy.deepcopy(self.__results)

    def fetch_result_for(self, key):
        return copy.deepcopy(self.__results)[key]
    
    def print_result_for(self, key):
        results = self.__results[key]
        for res in results:
            print(res)
