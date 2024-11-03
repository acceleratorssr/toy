import os
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.create_embedding_response import CreateEmbeddingResponse


class EmbeddingService:
    def __init__(self, api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=120, max_retries=2):
        self.client = Ark(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def get_embeddings(self, model_name, input_texts):
        """
        获取文本的嵌入向量，并确保返回的向量长度为4096。

        :param model_name: 使用的模型名称
        :param input_texts: 需要被向量化的文本列表
        :return: 长度为4096的向量结果列表
        """
        try:
            resp = self.client.embeddings.create(model=model_name, input=input_texts)

            if isinstance(resp, CreateEmbeddingResponse):
                vectors = [embedding.embedding for embedding in resp.data]
                validated_vectors = [self._validate_vector(v) for v in vectors]

                return validated_vectors
            else:
                raise ValueError("Invalid response format or no data received.")
        except Exception as e:
            print(f"Error during embedding request: {e}")
            raise

    def _validate_vector(self, vector):
        """
        验证向量的长度，如果不符合则返回默认值。
        :param vector: 输入向量
        :return: 长度为4096的向量
        """
        if len(vector) == 4096:
            return vector
        else:
            print(f"Warning: Vector length {len(vector)} does not match expected length 4096.")
            return [0.0] * 4096  # 返回长度为4096的默认向量

# eg
if __name__ == "__main__":
    api_key = os.environ.get("ARK_API_KEY")
    model_name = os.getenv('RP')

    embedding_service = EmbeddingService(api_key)

    # 要向量化的文本
    input_texts = ["花椰菜又称菜花、花菜，是一种常见的蔬菜。"]

    # 获取嵌入向量
    try:
        vectors = embedding_service.get_embeddings(model_name, input_texts)
        print("Generated Embeddings:", vectors)
    except Exception as e:
        print(f"Error occurred: {e}")
