import socket
import json
from vector_test import QUERY_VECTOR

SOCKET_PATH = "/tmp/coordinator.sock"

def send_query(query_vector):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(SOCKET_PATH)
        json_data = json.dumps(query_vector).encode('utf-8')
        client.sendall(json_data)
        client.shutdown(socket.SHUT_WR)

        response_data = receive_full_data(client)
        if response_data:
            response = json.loads(response_data)
            print("推荐结果:")
            for item in response:
                text = item[0]
                similarity = item[1]
                print(f"Text: {text}, Similarity: {similarity}")
        else:
            print("No response received from server")

def receive_full_data(connection):
    chunks = []
    while True:
        data = connection.recv(4096)
        if not data:
            break
        chunks.append(data.decode('utf-8'))
    return ''.join(chunks)

if __name__ == "__main__":
    # real data
    # query_vector = embedding_service.get_embeddings(model_name, ["广州"])

    # test data
    query_vector = QUERY_VECTOR

    send_query(query_vector)
