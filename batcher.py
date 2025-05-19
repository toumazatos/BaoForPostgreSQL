import random
import json

def read_queries(query_paths):
    queries = []
    for fp in query_paths:
        with open(fp) as f:
            query = f.read()
        queries.append((fp, query))
    return queries

def create_sequence(queries, seq_length, specific_query, specific_query_n):
    random.seed(42)
    # A python list of k tuples in the form of:
    #     ("../sql_file_path.sql", "SELECT contents FROM sql_file;")
    # from the query sample pool provided. 
    """target_query = [element for element in queries if specific_query in element[0]] 
    if target_query: # if the specified query exists in the query set
        print("check")
        #
        #specific_queries = target_query * specific_query_n
        #query_sequence = random.choices(queries, k=seq_length-specific_query_n)
        #query_sequence += specific_queries
        #random.seed(42)
        #random.shuffle(query_sequence)
        #return query_sequence
        #
        # for storing query_plans
        print(len(queries))
        query_sequence = random.choices(queries, k=40)
        query_sequence += 6 * queries
        return query_sequence
    
    else:"""
    query_sequence = random.choices(queries, k=seq_length)
    
    return query_sequence

def create_batches(query_sequence, n):
    def chunks(lst, n):
        # The 'chunks' function cuts a python list in n parts
        # and returns an iterator.
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # The 'chunks' function cuts a python list in n parts and
    # returns an iterator. In the next line, pg_chunks stores the first
    # chunk. *bao_chunks stores the rest of the chunks in a list.
    pg_chunks, *bao_chunks = list(chunks(query_sequence, n))
    return pg_chunks, bao_chunks

def generate_workload(query_paths, sequence_length, batch_length, remove_query, SAVE=False, out_file="", specific_query=None, specific_query_n=10):
    qs = read_queries(query_paths)
    seq = create_sequence(qs, sequence_length, specific_query, specific_query_n)
    pg_batch, bao_batches = create_batches(seq, batch_length)
    workload = {
        "pg" : pg_batch,
        "bao": bao_batches
    }
    if remove_query == 1:
        # generating workload without the specified query
        bao_batches_cleared = []
        for batch in bao_batches:
            bao_batches_cleared.append([item for item in batch if specific_query not in item[0]])
        workload_removed_query = {
            "pg" : pg_batch,
            "bao": bao_batches_cleared
        }
    if SAVE:
        if remove_query != 1:
            save_workload(out_file, workload)
        else:
            save_workload(out_file.split(".")[0]+"_before.json", workload) # save workload before deletion 
            save_workload(out_file.split(".")[0]+"_after.json", workload_removed_query) # save file after deletion
    return workload
    
def save_workload(out_file, workload):
    with open(out_file, "w") as workload_file:
        json.dump(workload, workload_file)

def load_workload(workload_file_path):
    workload_file = open(workload_file_path)
    workload = json.load(workload_file)
    workload_file.close()
    return workload