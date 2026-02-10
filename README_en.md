# Memory

[简体中文](README.md) | [English](README_en.md)

## Quick Start

### Environment Configuration

```
conda create -n yourenv python=3.13
pip install -r requirements.txt
```
Ensure your command line root directory is under `/memory`
### Data Conversion
**Simplified Process**:
1. Place all collected data in the `./raw_data` directory, ensuring it is in the following format:

```
├── raw_data
│ ├── Li Qiang # Username
│ │ ├── calendar.json
│ │ ├── call.json
│ │ ├── QA.json
│ │ ├── fitness_health.json
│ │ ├── agent_chat.json
│ │ ├── photo.json
│ │ ├── note.json
│ │ ├── sms.json
│ │ ├── contact.json
│ │ ├── push.json
│ ├── Xu Jing # Username
│ │ ├── calendar.json
│ │ ├── call.json
│ │ ├── QA.json
│ │ ├── fitness_health.json
│ │ ├── agent_chat.json
│ │ ├── photo.json
│ │ ├── note.json
│ │ ├── sms.json
│ │ ├── contact.json
│ │ ├── push.json
```
2. Simply run `sh preprocess.sh` to automatically generate the `./dataset/out.json` file and copy it to the specified location on all memory systems. (For Windows systems, it is recommended to use the `git bash` terminal. If this is not possible, you can use the corresponding `.ps1` file. The first run may encounter permission issues; you will need to use `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. If problems persist, you can step-by-step parse and execute the commands in `preprocess.sh` until `our.json` is finally generated.)

### System Usage
1. The previous step will copy `out.json` into the corresponding memory system directory, keeping the name `our.json`.
   1. MemOS: `MemOS/evaluation/data/our/our.json`
   2. Hindsight: `Hindsight/benchmarks/our/datasets/our.json`
   3. MemU: `MemU/data/our.json`
2. Modify the memory system environment

   1. Simple operation

      1. Configure all variables to be configured in `./set_env.py`

      2. Execute `python ./set_env.py`

      3. **Important**: Ensure that the `NOW_EMBEDDING_MODEL` variable is updated to the previously set EMBEDDING model; otherwise, the replacement will fail. Initially, it's `your_embedding_model_here`. If you've already executed `set_env.py`, you need to assign the value of `MEMU_EMBEDDING_MODEL` used last time to `NOW_EMBEDDING_MODEL`.

   2. Detailed Operations (The following describes what the previous step did. If you've successfully executed the previous step, you don't need to execute the next step.)

      1. MemOS:

         1. Modify `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` in `MemOS/.env` to your own model and key.

         2.  Modify `api_key` in `MemOS/evaluation/config/systems/memos.yaml` to your own MemOS key.

         3.  Modify `LLM/api_key` in `MemOS/evaluation/config/systems/memos.yaml` to your own model API key.

      2.  Hindsight:

          1.  Modify `HINDSIGHT_API_LLM_API_KEY`, `HINDSIGHT_API_LLM_BASE_URL`, `HINDSIGHT_API_LLM_MODEL`, `HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY`, `HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL`, and `HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL` in `Hindsight/.env` to your own model and key.

      3.  MemU:

          1.  Configure your own `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `MemU/.env`.

          2.  In `MemU/memu/memory/embeddings.py`, on lines 37 and 90, change the second parameter of `kwargs.get("model", "")` to your own model.

3.  Understanding Important Files and Directories

The following are the locations of important memory result files for the three memory systems:

1. MemOS: `MemOS/evaluation/results/our-memos`

2. Hindsight: `Hindsight/benchmarks/our/results`

3. MemU: `MemU/memory` and `MemU/enhanced_memory_test_results.json`

If you have already created relevant files in these directories or files, but need to perform a new evaluation, consider whether to delete the relevant files. This depends on whether the memory process has been completed completely. If it has been completed completely, there is no need to delete them; otherwise, be sure to delete the files in the above directories or files, otherwise some steps will be skipped on the next run.

4. Execute Evaluations

   1. MemOS:

   Execute in the root directory:

   ```
   cd MemOS

   python -m evaluation.cli --dataset our --system memos

   ```

   2. Hindsight:

   (Ensure you can connect to huggingface) Execute in the root directory:

   ```
   cd Hindsight

   ./run-our.sh

   ```

   3. MemU:

   Execute in the root directory:

   ```
   cd MemU

   python locomo_test.py --data-file data/our.json --chat-deployment gpt-5-mini-2025-08-07

   ```

Additionally, note that your chat-deployment model and embedding model need to match your API key.

5. Run Analysis

After confirming that the evaluation result files for the three memory systems have been generated, return to the root directory and run the following command:

``` python quick_analyze.py ```


The analysis and statistical results will be saved in `./result.json`.

If you do not have complete evaluation result files for all three memory systems, please manually modify `quick_analyze.py` accordingly.

By default, method_a corresponds to MemOS, method_b to Hindsight, and method_c to MemU.

## Dataset Statistics
| **Metric**                         | **Value** |
| ---------------------------------- | --------- |
| Total users                        | 10        |
| Duration per user                  | 1 year    |
| Total events                       | 51,491    |
| Events per user (mean)             | 5,149     |
| Events per user (median)           | 5,147     |
| Persona tokens (mean)              | 17,730.8  |
| Relationships (mean)               | 24.4      |
| Locations (mean)                   | 15        |
| Contacts per user (mean)           | 21.9      |
| Calls per user (mean)              | 1,040.4   |
| SMS messages per user (mean)       | 1,812.9   |
| Calendar events per user (mean)    | 234.4     |
| Agent chat conversations per user  | 688.5     |
| Photos per user (mean)             | 1,233.4   |
| Notes per user (mean)              | 363.1     |
| Push notifications per user (mean) | 2,350.3   |
| Fitness health records per user    | 301.0     |
| Monthly summaries per user         | 12        |
| Summary length (words, per user)   | 1,737.7   |
| Context depth (tokens) per user    | 3.66M     |
| Total dataset size                 | 332 MB    |
| Generation time per user           | 8 hours   |
