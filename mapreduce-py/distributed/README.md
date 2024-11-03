# 说明
实现了一个基于 Unix Socket 的协调者和工作者模型，用于处理向量相似度计算;
允许用户通过单个或多个查询向量获取相似文本的推荐结果;

## 目录结构
- coordinator.py: coordinate 进程，负责接收 client 请求，分配任务给 worker，并处理返回的结果;
- worker.py: worker 进程，负责接收任务，执行相似度计算并返回结果;
- embedding.py：封装了与向量化大模型的交互，负责获取文本的嵌入向量;
- vector_test.py：包含测试数据;

## 功能
1. 协调者（Coordinator）

创建 Unix Socket 服务器，等待客户端连接;

接收查询向量，分发 map 和 reduce 任务给 worker;

收集 worker 的计算结果，并返回给客户端;

2. 工人（Worker）

创建 Unix Socket 服务器，等待协调者的请求;

执行余弦相似度计算，将相似度结果返回给协调者;

3. 嵌入服务（Embedding Service）

与外部向量化大模型交互，通过 API 获取文本的嵌入向量;

默认为 4096 维度，目前使用的向量化大模型不支持自定义降维，需要自己裁剪；

## 使用说明

- 启动工作者进程：
```bash
python3 worker.py
```

- 启动协调者进程：
```bash
python3 coordinator.py
```

- 发送查询请求
在 client.py 中，可以定义要查询的向量并通过 Unix Socket 将请求发送给协调者，协调者将处理请求并返回相似文本;

test eg:
```bash
python3 client.py
```

- 向量化大模型
需要在环境变量中设置大模型的 `ARK_API_KEY` 和 模型的endpoint: `EP`；