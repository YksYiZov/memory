# Memory

## 快速使用

### 配置环境

```
conda create -n yourenv python=3.13
pip install -r requirements.txt
```

### 数据转换

1. 需要将`./locomo_transfer.py`中第五行的`PROTAGONIST`修改成你的User名
2. 务必将原始数据中`phone_data`相关和`QA`相关的`json`文件置入`./raw_data`目录中
3. 执行命令`sh ./transfer.sh`
4. 一切正常的话你会看到`./our.json`出现到根目录
5. 如果此次仅处理单用户，那么到此结束。如果处理多个用户还需按下面步骤进行（如果想一个用户一个用户进行需要做好对用户评测结果的转存，否则上个用户的评测结果可能被覆盖）
6. 将`./our.json`保存或重命名为其他文件，比如处理User为徐静时，可以另存为在`./dataset/徐静.json`，此时第一个用户处理完成
7. 随后重复前四个步骤，处理第二个用户，一样将输出的`./our.json`保存在一个位置
8. 合并所有用户文件，将所有用户复制到同一个`our.json`中，此`our.json`即为最终使用的数据文件

### 系统使用

1. 将`out.json`复制入对应的记忆系统目录，保持命名为`our.json`

   1. MemOS: `EverMemOS-main/evaluation/data/our/our.json`
   2. Hindsight: `hindsight/benchmarks/our/datasets/our.json`
   3. MemU: `memU-experiment-main/data/our.json`

2. 修改记忆系统环境

   1. MemOS: 
      1. 修改`EverMemOS-main/.env`中的`LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL`为自己的模型和Key
      2. 修改`EverMemOS-main/evaluation/config/systems/memos.yaml`中的`api_key`为自己的MemOS Key
   2. Hindsight:
      1. 修改`memory/hindsight/.env`中的`HINDSIGHT_API_LLM_API_KEY`、`HINDSIGHT_API_LLM_BASE_URL`、`HINDSIGHT_API_LLM_MODEL`、`HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY`、`HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL`和`HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL`为自己的模型和Key
   3. MemU:
      1. 在`memory/memU-experiment-main/.env`中配置自己的`OPENAI_API_KEY`和`OPENAI_BASE_URL`
      2. 在`memory/memU-experiment-main/memu/memory/embeddings.py`中将所有的`text-embedding-v4`替换为自己的embedding模型

3. 考虑旧文件

以下为三个记忆系统重要记忆结果文件位置：

   1. MemOS: `EverMemOS-main/evaluation/results/our-memos`
   2. Hindsight: `hindsight/benchmarks/our/results`
   3. MemU: `memU-experiment-main/memory`

如果你是第一次运行：

直接删除目录下所有文件（不要删除目录）

如果你不是第一次运行，请确保你知道自己在做什么否则不要轻易删除。这些目录下保存的文件会影响后续运行：MemOS中的记录会影响下次运行从哪一步开始；HindSight中的结果记录可能会导致下次结果生成有问题；MemU中的文件会直接影响下次记忆生成。这些文件可能会为下次评测提供便利也可能会导致下次在错误的条件下评测，如有问题请联系我们。

4. 执行评测

   1. MemOS：

      ```
      cd EverMemOS-main
      python -m evaluation.cli --dataset our --system memos
      ```

   2. Hindsight:

      ```
      cd hindsight
      ./run-our.sh
      ```

   3. MemU:

      ```
      cd memU-experiment-main
      python locomo_test.py --data-file data/our.json --chat-deployment qwen3-max
      ```

5. 运行分析

确认三个记忆系统评测结果文件都生成后，回到根目录，运行以下命令：

```
python quick_analyze.py
```

分析结果会保存在`./result.json`中