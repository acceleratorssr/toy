import socket
import json
import os
import traceback
from multiprocessing import Process, Pool, Manager
from typing import List, Tuple

from vector_test import EMBEDDINGS, KEYWORDS
from worker import WORKER_SOCKET_PATH
from embedding import EmbeddingService

SOCKET_PATH = "/tmp/coordinator.sock"
TOP_N = 2 # 推荐结果的数量

def create_unix_socket(path):
    if os.path.exists(path):
        os.remove(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(5)
    return server

def distribute_map_tasks(data, query_vector):
    results = communicate_with_worker('map', data, query_vector)

    return results

def distribute_reduce_task(map_results):
    """分配Reduce任务到Worker进程"""
    response = communicate_with_worker('reduce', map_results, None)
    return response

import numpy as np

def communicate_with_worker(task_type, data_chunk, query_vector=None):
    data_chunk = [(item_id, vector.tolist()) if isinstance(vector, np.ndarray) else (item_id, vector) for item_id, vector in data_chunk]

    if query_vector is not None:
        query_vector = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

    request = {
        'task': task_type,
        'data': data_chunk,
    }

    if query_vector is not None:
        request['query_vector'] = query_vector

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(WORKER_SOCKET_PATH)
    client.sendall(json.dumps(request).encode('utf-8'))
    client.shutdown(socket.SHUT_WR)

    response_data = receive_full_data(client)
    client.close()

    if not response_data:
        print("Error: No data received from worker.")
        return []

    try:
        return json.loads(response_data)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in worker response: {e}")
        print(f"Response data: {response_data}")
        return []


def receive_full_data(connection):
    chunks = []
    while True:
        data = connection.recv(4096)
        if not data:
            break
        chunks.append(data.decode('utf-8'))
    return ''.join(chunks)

def handle_client_connection(connection, data):
    try:
        received_data = receive_full_data(connection)
        query_vectors = np.array(json.loads(received_data))

        if query_vectors.ndim == 1:
            query_vectors = np.expand_dims(query_vectors, axis=0)

        if query_vectors.shape[1] != 4096:
            raise ValueError("All vectors must be of length 4096")

        map_results = []

        for query_vector in query_vectors:
            results = distribute_map_tasks(data, query_vector)
            map_results.extend(results)

        top_results = distribute_reduce_task(map_results)

        response = json.dumps(top_results).encode()
        connection.sendall(response)
    except Exception as e:
        print(f"Error in coordinator: {e}")
        traceback.print_exc()
    finally:
        connection.close()



def server_process(data):
    server = create_unix_socket(SOCKET_PATH)
    print("Coordinator server started.")
    try:
        with server:
            while True:
                conn, _ = server.accept()
                handle_client_connection(conn, data)
    except KeyboardInterrupt:
        print("Shutting down coordinator server.")
    finally:
        server.close()


def load_data():
    keywords = KEYWORDS
    data: List[Tuple[str, np.ndarray]] = []

    for i in range(len(keywords)):
        vector = np.array(EMBEDDINGS[i])
        data.append((keywords[i], vector))
    return data


if __name__ == "__main__":
    # # real data
    # api_key = os.environ.get("ARK_API_KEY")
    # model_name = os.getenv('RP')
    #
    # embedding_service = EmbeddingService(api_key)
    #
    # keywords = KEYWORDS
    #
    # data = [(keyword, embedding_service.get_embeddings(model_name, [keyword])) for keyword in keywords]

    data: List[Tuple[str, np.ndarray]] = load_data()  # Assuming load_data() loads or generates data
    with Manager() as manager:
        data = manager.list(data)
        server = Process(target=server_process, args=(data,))
        server.start()
        server.join()
