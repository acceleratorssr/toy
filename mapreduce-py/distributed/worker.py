import socket
import json
import os
import traceback

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

WORKER_SOCKET_PATH = "/tmp/worker.sock"
TOP_N = 2  # 推荐结果的数量

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
    print("get map task")
    ids, vectors = zip(*data_chunk)
    similarities = cosine_similarities(query_vector, np.array(vectors))
    return list(zip(ids, similarities))

def reduce_function(map_results):
    similarity_scores = {}

    for result in map_results:
        data_id, similarity = result
        if data_id in similarity_scores:
            similarity_scores[data_id] += similarity
        else:
            similarity_scores[data_id] = similarity

    total_vectors = len(map_results)
    average_similarity = {data_id: score / total_vectors for data_id, score in similarity_scores.items()}

    sorted_results = sorted(average_similarity.items(), key=lambda x: x[1], reverse=True)

    formatted_results = [{"Text": data_id, "Similarity": similarity} for data_id, similarity in sorted_results[:TOP_N]]

    return formatted_results

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
