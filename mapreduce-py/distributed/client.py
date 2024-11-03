import os
import socket
import json

from vector_test import QUERY_VECTOR,QUERY_VECTOR1,QUERY_VECTOR2
from embedding import EmbeddingService

SOCKET_PATH = "/tmp/coordinator.sock"

def send_query(query_vector):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(SOCKET_PATH)
        json_data = json.dumps(query_vector).encode('utf-8')
        client.sendall(json_data)
        client.shutdown(socket.SHUT_WR)

        response_data = receive_full_data(client)
        try:
            results = json.loads(response_data)

            for item in results:
                text = item["Text"]
                similarity = item["Similarity"]
                print(f"Text: {text}")
                print(f"Similarity: {similarity}")
        except KeyError as e:
            print(f"Key error: {e} - Ensure response structure matches expected format.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")

def receive_full_data(connection):
    chunks = []
    while True:
        data = connection.recv(4096)
        if not data:
            break
        chunks.append(data.decode('utf-8'))
    return ''.join(chunks)

if __name__ == "__main__":
    api_key = os.environ.get("ARK_API_KEY")
    model_name = os.getenv('EP')

    embedding_service = EmbeddingService(api_key)
    # real data
    # v = embedding_service.get_embeddings(model_name, ["广州"])
    # v1 = embedding_service.get_embeddings(model_name, ["房"])
    # v2 = embedding_service.get_embeddings(model_name, ["港澳台"])
    # query_vector = [v, v1, v2]

    # test data
    query_vector = [QUERY_VECTOR,QUERY_VECTOR1,QUERY_VECTOR2]

    # 支持不定向量查询
    send_query(query_vector)
