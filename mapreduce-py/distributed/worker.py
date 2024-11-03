import socket
import json
import os
import traceback

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

WORKER_SOCKET_PATH = "/tmp/worker.sock"
TOP_N = 3  # 推荐结果的数量

def create_unix_socket(path):
    if os.path.exists(path):
        os.remove(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(5)
    return server

def cosine_similarities(vector, matrix):
    vector = vector.reshape(1, -1)
    return cosine_similarity(vector, matrix).flatten()

def map_function(data_chunk, query_vector):
    ids, vectors = zip(*data_chunk)
    similarities = cosine_similarities(query_vector, np.array(vectors))
    return list(zip(ids, similarities))

def reduce_function(results):
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results[:TOP_N]

def handle_coordinator_request(connection):
    try:
        received_data = receive_full_data(connection)
        request = json.loads(received_data)
        task_type = request['task']
        data = request['data']

        if task_type == 'map':
            query_vector = np.array(request['query_vector'])
            result = map_function(data, query_vector)
        elif task_type == 'reduce':
            result = reduce_function(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        response = json.dumps(result).encode()
        connection.sendall(response)
    except Exception as e:
        print(f"Error in worker: {e}")
        traceback.print_exc()
    finally:
        connection.close()

def receive_full_data(connection):
    chunks = []
    while True:
        data = connection.recv(4096)
        if not data:
            break
        chunks.append(data.decode('utf-8'))
    return ''.join(chunks)

def worker_process():
    """Worker服务器进程"""
    server = create_unix_socket(WORKER_SOCKET_PATH)
    print("Worker server started.")
    try:
        with server:
            while True:
                conn, _ = server.accept()
                handle_coordinator_request(conn)
    except KeyboardInterrupt:
        print("Shutting down worker server.")
    finally:
        server.close()

if __name__ == "__main__":
    worker_process()
