# memOS调用API接口，LOCOMO数据集评测

## 1、docker安装

##/2、yaml配置
### 1
D:\agentstu\EverMemOS-main\evaluation\config\systems\memos.yaml配置如下：
api_key: "mpg-xxxxxxxxx"#memOS官网申请apikey
### 2
    docker-compose.yaml 配置文件如下：
services:
  app:
    build: .
    container_name: evermemos-app
    volumes:
      - ./:/app
      - ./evaluation:/app/evaluation
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
      - PYTHONUTF8=1
      - PYTHONPATH=/app
    # 核心关键：执行永久前台运行的命令，让容器一直存活，专为测评设计
    command: tail -f /dev/null
    restart: "no"
    stdin_open: true
    tty: true

## 3、.env配置

MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DB=evermemos
REDIS_HOST=localhost
REDIS_PORT=6379
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
MILVUS_HOST=localhost
MILVUS_PORT=19530
LLM_API_KEY=sk-xxxxxxxxxx
LLM_MODEL=gpt-5-mini
LLM_AUTH_TYPE=api_key
LLM_BASE_URL=https://api.bianxieai.com/v1
OPENROUTER_API_KEY=sk-xxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxx

## 4、启动容器
 powershell进入项目目录
 cd D:\agentstu\EverMemOS-main
 ##### 第二步：彻底清理所有残留（容器、镜像、数据卷、缓存，无残留）
docker-compose down -v --rmi all --remove-orphans
#####  启动容器
docker-compose up -d 

##### 检测容器状态
docker ps


## 5、补齐依赖
**进入容器**docker-compose exec app /bin/bash
**安装依赖**pip install rich python-dotenv pyyaml requests pandas numpy openai tiktoken aiohttp aiolimiter tenacity httpx asyncio python-dotenv>=1.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
**退出**exit


## 6、测评
**清空缓存**
docker-compose exec app rm -rf /app/evaluation/results/locomo-memos

**冒烟测试**
docker-compose exec app python -m evaluation.cli --dataset locomo --system memos


**跑第一个会话测试**
docker-compose exec app python -m evaluation.cli --dataset locomo --system memos --from-conv 0 --to-conv 1 --stages all
**跑全部会话测试**
docker-compose exec app python -m evaluation.cli --dataset locomo --system memos