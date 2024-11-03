# -*- coding: utf-8 -*-

import socket
import os
import json
import time
import traceback
from typing import List, Tuple
import numpy as np
from multiprocessing import Process, Manager, Pool
from sklearn.metrics.pairwise import cosine_similarity

from vector_test import EMBEDDINGS, QUERY_VECTOR, KEYWORDS
from embedding import EmbeddingService

SOCKET_PATH = "/tmp/recommend.sock"
TOP_N = 3  # 推荐结果的数量

def create_unix_socket():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    return server

def cosine_similarities(vector, matrix):
    vector = vector.reshape(1, -1)
    return cosine_similarity(vector, matrix).flatten()

def map_function(data_chunk, query_vector):
    for item_id, vector in data_chunk:
        if vector.shape != query_vector.shape:
            raise ValueError(f"Shape mismatch: query_vector {query_vector.shape} vs vector {vector.shape}")

    """Map阶段：计算余弦相似度"""
    ids, vectors = zip(*data_chunk)
    similarities = cosine_similarities(query_vector, np.array(vectors))
    return list(zip(ids, similarities))

def reduce_function(results):
    """Reduce阶段：汇总并返回 Top N 相似度"""
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results[:TOP_N]

def handle_client_connection(connection, data):
    """处理客户端请求并返回推荐结果"""
    try:
        received_data = receive_full_data(connection)
        query_vector = np.array(json.loads(received_data))

        # 确保 query_vector 是一维数组
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be a one-dimensional array.")

        # 检查 data 的长度
        if len(data) == 0:
            print("No data available for processing.")
            response = json.dumps([]).encode()  # 发送空列表
            connection.sendall(response)
            return

        # 计算分片参数
        num_chunks = os.cpu_count() or 1
        chunk_size = max(1, len(data) // num_chunks)

        # 将 data 分片，避免创建不必要的 NumPy 数组
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 确保每个 chunk 是 (id, vector) 的列表
        chunks = [[(item[0], item[1]) for item in chunk] for chunk in chunks]

        # 并行映射计算
        with Pool(os.cpu_count()) as pool:
            map_results = pool.starmap(map_function, [(chunk, query_vector) for chunk in chunks])

        # Reduce 阶段汇总结果
        flat_results = [item for sublist in map_results for item in sublist]
        top_results = reduce_function(flat_results)

        response = json.dumps(top_results).encode()

        connection.sendall(response)
    except json.JSONDecodeError as json_err:
        print(f"JSON decoding error: {json_err}")
        response = json.dumps({"error": "Invalid JSON format."}).encode()
        connection.sendall(response)
    except Exception as e:
        print(f"Error handling client connection: {e}")
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
    full_data = ''.join(chunks)

    return full_data

def server_process(data):
    """服务器进程：接收连接，处理请求"""
    server = create_unix_socket()
    print("Unix socket server started.")
    try:
        with server:
            while True:
                conn, _ = server.accept()
                print("get Conn")
                handle_client_connection(conn, data)
                print("over Conn")
    except KeyboardInterrupt:
        print("Shutting down server.")
    finally:
        server.close()

if __name__ == "__main__":
    api_key = os.environ.get("ARK_API_KEY")
    model_name = os.getenv('RP')

    embedding_service = EmbeddingService(api_key)

    keywords = KEYWORDS

    # real data
    # data = [(keyword, embedding_service.get_embeddings(model_name, [keyword])) for keyword in keywords]

    # test data
    data: List[Tuple[str, np.ndarray]] = []

    for i in range(len(keywords)):
        vector = np.array(EMBEDDINGS[i])
        data.append((keywords[i], vector))

    with Manager() as manager:
        data = manager.list(data)
        server = Process(target=server_process, args=(data,))
        server.start()

        time.sleep(1)

        # 客户端请求示例
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(SOCKET_PATH)

            # real data
            # query_vector = embedding_service.get_embeddings(model_name, ["广州"])

            # test data
            query_vector = QUERY_VECTOR

            json_data = json.dumps(query_vector).encode('utf-8')

            client.sendall(json_data)
            client.shutdown(socket.SHUT_WR)
            print("over send")

            response_data = receive_full_data(client)
            client.close()
            if response_data:
                response = json.loads(response_data)
                print("推荐结果:")
                for item in response:
                    text = item[0]
                    similarity = item[1]
                    print(f"Text: {text}")
                    print(f"Similarity: {similarity}")
            else:
                print("No response received from server")
