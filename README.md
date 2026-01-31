# Memory

## 快速使用

### 配置环境

```
conda create -n yourenv python=3.13
pip install -r requirements.txt
```

确保你的命令行根目录在`/memory`下

### 数据转换

**简便过程**：
1. 将所有收集数据放入`./raw_data`目录下，请确保为以下格式：
```
├── raw_data
│   ├── 李强 # 用户名
│   │   ├── calendar.json
│   │   ├── call.json
│   │   ├── QA.json 
│   │   ├── fitness_health.json
│   │   ├── agent_chat.json
│   │   ├── photo.json
│   │   ├── note.json
│   │   ├── sms.json
│   │   ├── contact.json
│   │   ├── push.json
│   ├── 徐静 # 用户名
│   │   ├── calendar.json
│   │   ├── call.json
│   │   ├── QA.json
│   │   ├── fitness_health.json
│   │   ├── agent_chat.json
│   │   ├── photo.json
│   │   ├── note.json
│   │   ├── sms.json
│   │   ├── contact.json
│   │   ├── push.json
```
2. 直接`sh preprocess.sh`，即可自动生成`./dataset/out.json`文件，同时将该文件复制入所有记忆系统指定位置。（如果是windows系统，建议使用`git bash`终端，实在没办法可选择使用.ps1对应文件，第一次运行可能出现权限问题，需要使用`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`，如果仍存在问题，可以逐步解析`preprocess.sh`中的命令一步一步运行直至最终生成`our.json`）

### 系统使用

1. 上步会将将`out.json`复制入对应的记忆系统目录，保持命名为`our.json`

   1. MemOS: `EverMemOS-main/evaluation/data/our/our.json`
   2. Hindsight: `hindsight/benchmarks/our/datasets/our.json`
   3. MemU: `memU-experiment-main/data/our.json`

2. 修改记忆系统环境

   1. 简便操作
      1. 配置`./set_env.py`中所有待配置变量
      2. 执行`python ./set_env.py`
      3. **重要**：务必确定`NOW_EMBEDDING_MODEL`变量会被更新为上次已经被设置过得EMBEDDING模型，否则替换会失败。初始时为`your_embedding_model_here`，如果你已经执行过`set_env.py`，则需要把上次使用的`MEMU_EMBEDDING_MODEL`的值赋值给`NOW_EMBEDDING_MODEL`

   2. 详细操作（以下内容描述上步操作做了什么，如果你已经成功执行上一步操作，无需执行下一步操作）
      1. MemOS: 
         1. 修改`EverMemOS-main/.env`中的`LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`为自己的模型和Key
         2. 修改`EverMemOS-main/evaluation/config/systems/memos.yaml`中的`api_key`为自己的MemOS Key
         3. 修改`EverMemOS-main/evaluation/config/systems/memos.yaml`中的`LLM/api_key`为自己的模型API Key
      2. Hindsight:
         1. 修改`memory/hindsight/.env`中的`HINDSIGHT_API_LLM_API_KEY`、`HINDSIGHT_API_LLM_BASE_URL`、`HINDSIGHT_API_LLM_MODEL`、`HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY`、`HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL`和`HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL`为自己的模型和Key
      3. MemU:
         1. 在`memory/memU-experiment-main/.env`中配置自己的`OPENAI_API_KEY`和`OPENAI_BASE_URL`
         2. 在`memory/memU-experiment-main/memu/memory/embeddings.py`中第37行和第90行中`kwargs.get("model", "")`中后一个参数改为自己的模型

3. 了解重要文件和目录

以下为三个记忆系统重要记忆结果文件位置：

   1. MemOS: `EverMemOS-main/evaluation/results/our-memos`
   2. Hindsight: `hindsight/benchmarks/our/results`
   3. MemU: `memU-experiment-main/memory`和`memU-experiment-main/enhanced_memory_test_results.json`

当你已经在这些目录或文件中创建了相关文件，但是还需要进行新一次评测时，请考虑是否要删除相关文件，这取决于记忆过程是否完全进行，如果完全进行，则无需删除，否则务必删除上述目录中的文件或文件，否则下次运行会跳过某些步骤。

4. 执行评测

   1. MemOS：
      在根目录下执行：
      ```
      cd EverMemOS-main
      python -m evaluation.cli --dataset our --system memos
      ```

   2. Hindsight:
      (请确保可连接至huggingface)在根目录下执行：
      ```
      cd hindsight
      ./run-our.sh
      ```

   3. MemU:
      在根目录下执行：
      ```
      cd memU-experiment-main
      python locomo_test.py --data-file data/our.json --chat-deployment qwen3-max
      ```

      额外需要注意的是，你的chat-deployment模型和embedding模型需要匹配你的api key
5. 运行分析

确认三个记忆系统评测结果文件都生成后，回到根目录，运行以下命令：

```
python quick_analyze.py
```

分析统计结果会保存在`./result.json`中

如果你并没有完整三个记忆系统评测结果文件，请手动在`quick_analyze.py`中修改

```
final_results = {
        "method_a": extract_results_method_a(data_a),
        "method_b": extract_results_method_b(data_b),
        "method_c": extract_results_method_c(data_c),
    }
```

将某几行注释掉，method_a对应memOS，method_b对应hindsight,method_c对应memU