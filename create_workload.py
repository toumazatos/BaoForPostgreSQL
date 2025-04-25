import batcher
import sys

# Usage:
# $ python3 create_workload.py file_to_save_workload.json seq_length batch_size queries/sample1/* q8_6a505 specific_instances
# specific_query: eg. q8_6a505, specific_instances: eg. 45

target_file = sys.argv[1]
sequence_length = int(sys.argv[2])
batch_length = int(sys.argv[3])
remove_query = int(sys.argv[4])
query_paths = sys.argv[5:-2]
specific_query = sys.argv[-2]
specific_query_instances = int(sys.argv[-1])

batcher.generate_workload(query_paths, sequence_length, batch_length, remove_query, SAVE=True, out_file=target_file, specific_query=specific_query, specific_query_n=specific_query_instances )