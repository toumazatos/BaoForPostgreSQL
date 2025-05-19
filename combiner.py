import sqlite3
import sys

def _db_conn(db_path):
    conn = sqlite3.connect(db_path)
    return conn

def _add_names(db_path, names):
    with _db_conn(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""ALTER TABLE experience ADD COLUMN query_name;""")
        conn.commit()
        for i, name in enumerate(names):
            cursor.execute("UPDATE experience SET query_name = '{}' WHERE id = {};".format(name, i+1))
            conn.commit()
        cursor.close()

def _get_query_paths(results_file_path):
    query_sequence = []
    with open(results_file_path, 'r') as f:
        for line in f:
            if line.startswith("x") or line[0].isnumeric():
                contents = line.split(" ")
                path_contents = contents[3].split("/")
                query = path_contents[-1]
                query_sequence.append(query)
    return query_sequence


def combine_results(database_path, results_path):
    query_names = _get_query_paths(results_path)
    _add_names(database_path, query_names)

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("Usage: $ python3 combiner.py <DB_FILE_PATH> <RESULTS_FILE_PATH>")
    DATABASE_PATH = sys.argv[1]
    RESULTS_PATH = sys.argv[2]
    combine_results(DATABASE_PATH, RESULTS_PATH)