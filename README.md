# PreConBench - 医疗对话预问诊基准测试系统

## 中文版

### 1. 项目简介
PreConBench是一个用于评估医疗对话预问诊能力的基准测试系统。该系统模拟医生和患者之间的对话，通过评估对话质量和生成的病历记录来测试不同大语言模型在医疗预问诊场景下的表现。

主要特点：
- 支持多个医生模型和患者模型的并行测试
- 提供对话生成和评估两个主要功能
- 支持慢思考模式（Slow Think Mode）以提升模型表现
- 多线程处理以提高效率
- 自动保存对话记录和评估结果

### 2. 运行脚本
系统提供两个主要的运行脚本：

1. 主实验脚本：
```bash
bash scripts/run_benchmark.sh \
    --doctor_model_names "model1" "model2" \
    --doctor_model_url "url1" "url2" \
    --doctor_model_key "key1" "key2" \
    --patient_model_name "model_name" \
    --patient_model_url "patient_url" \
    --patient_model_key "patient_key" \
    --judge_model_name "judge_model" \
    --judge_model_url "judge_url" \
    --judge_model_key "judge_key" \
    --num_patient 50 \
    --num_dialog_max_turn 8 \
    --num_thread 16 \
    --doctor_slow_think_mode "Off" \
    --patient_slow_think_mode "Off" \
    --run_dialog "True" \
    --run_eval "True"
```

2. 对照实验脚本：
```bash
bash scripts/run_controlled_experiment_turn.sh  \
    --doctor_model_names "model1" "model2" \
    --doctor_model_url "url1" "url2" \
    --doctor_model_key "key1" "key2" \
    --patient_model_name "model_name" \
    --patient_model_url "patient_url" \
    --patient_model_key "patient_key" \
    --judge_model_name "judge_model" \ 
    --judge_model_url "judge_url" \
    --judge_model_key "judge_key" \
    --num_patient 50 \
    --num_dialog_max_turn 3 \  # 可选：3, 8, 15
    --num_thread 16
```

**脚本作用总览**
- 按目录顺序：core → data → scripts。

- core
  - `core/main.py`: 主入口，解析参数、读取数据、初始化模型，生成医患对话与病历并调用评测，保存结果。
  - `core/models.py`: 模型封装，提供 GPTModel/AgentModel/HuatuoModel 与 Patient/Doctor 慢思考模型及对话接口。
  - `core/dialog_manager.py`: 对话管理与流程控制，格式化提示词、维护对话记忆，轮次推进与病历生成，统计 token。
  - `core/evaluate.py`: 评测模块，调用评测模型对病历规范性/正确性、对话规范性及病历真实性打分并返回 tokens。
  - `core/data_utils.py`: 数据读取与整理，从 Excel/JSON 载入基准数据，整理字段并按性格模板扩展样本。
  - `core/args_input.py`: 命令行参数与配置汇总，组织路径/模型/数值/模式参数，并装配对话/评测提示词（零/一样例）。
  - `core/logger.py`: 日志初始化与控制台/文件输出配置。
  - `core/calculate_score.py`: 结果后处理与统计，解析评测输出，计算模型均分、分组统计与 token 汇总，写入 JSON。

- data
  - `data/bench_data/extract_info.py`: 从真实病历抽取并重排主诉/现病史/既往史，支持失败样本重试，生成 `process_case.json`。
  - `data/bench_data/bench_data.py`: 统计信息点与字数，划分难度标签，绘制分布并导出 `cases_counted*.json`。
  - `data/prompts/patient.py`: 病人角色提示词与“性格”模板（含慢思考提示）。
  - `data/prompts/doctor.py`: 医生角色提示词、最终病历生成提示与医生慢思考提示。
  - `data/prompts/judge_record.py`: 病历规范性/正确性评测提示词（含零/一样例模板）。
  - `data/prompts/judge_dialog.py`: 对话用语规范性评测提示与示例。
  - `data/prompts/judge_basic.py`: 病历真实性（是否来源于对话）评测提示与示例。
  - `data/prompts/dissect.py`: 主诉/现病史/既往史要素抽取与转述规则提示。

- scripts
  - `scripts/run_benchmark.sh`: 运行主实验全流程（对话生成+评测），参数见脚本。
  - `scripts/run_controlled_experiment_turn.sh`: 运行“对话轮次”对比实验（3/8/15 轮）。
  - `scripts/run_controlled_experiment_judge.sh`: 运行“不同评测模型”对比实验（仅评测阶段）。


### 3. 重要参数说明

#### 必需参数：
- `--doctor_model_names`: 医生模型名称列表
- `--doctor_model_url`: 医生模型API URL列表
- `--doctor_model_key`: 医生模型API密钥列表
- `--patient_model_name`: 患者模型名称
- `--patient_model_url`: 患者模型API URL
- `--patient_model_key`: 患者模型API密钥
- `--judge_model_name`: 评估模型名称
- `--judge_model_url`: 评估模型API URL
- `--judge_model_key`: 评估模型API密钥

#### 可选参数：
- `--num_patient`: 测试患者数量（默认：10）
- `--num_dialog_max_turn`: 最大对话轮数（默认：8）
- `--num_thread`: 线程数（默认：16）
- `--doctor_slow_think_mode`: 医生慢思考模式（"On"/"Off"，默认："Off"）
- `--patient_slow_think_mode`: 患者慢思考模式（"On"/"Off"，默认："On"）
- `--run_dialog`: 是否运行对话生成（"True"/"False"，默认："True"）
- `--run_eval`: 是否运行评估（"True"/"False"，默认："True"）

### 4. 对照实验说明
对照实验脚本（dialog_turn_run.py）主要用于研究对话轮数对模型表现的影响：
- 固定患者性格为00A
- 支持测试3、8、15轮对话
- 使用cases_counted_difficulty.json中的50个数据点
- 支持测试多个模型：
  - qwen2.5-7b-instruct
  - gpt-4o-2024-11-20
  - Qwen2.5-72B-Instruct
  - deepseek-r1
  - deepseek-v3

## English Version

### 1. Project Introduction
PreConBench is a benchmark system for evaluating medical dialogue pre-consultation capabilities. The system simulates doctor-patient dialogues and tests different large language models' performance in medical pre-consultation scenarios by assessing dialogue quality and generated medical records.

Key Features:
- Support parallel testing of multiple doctor and patient models
- Provide dialogue generation and evaluation functionalities
- Support Slow Think Mode to enhance model performance
- Multi-threading for improved efficiency
- Automatic saving of dialogue records and evaluation results

### 2. Running Scripts
The system provides two main running scripts:

1. Main Experiment Script:
```bash
bash scripts/run_benchmark.sh \
    --doctor_model_names "model1" "model2" \
    --doctor_model_url "url1" "url2" \
    --doctor_model_key "key1" "key2" \
    --patient_model_name "model_name" \
    --patient_model_url "patient_url" \
    --patient_model_key "patient_key" \
    --judge_model_name "judge_model" \
    --judge_model_url "judge_url" \
    --judge_model_key "judge_key" \
    --num_patient 50 \
    --num_dialog_max_turn 8 \
    --num_thread 16 \
    --doctor_slow_think_mode "Off" \
    --patient_slow_think_mode "Off" \
    --run_dialog "True" \
    --run_eval "True"
```

2. Controlled Experiment Script:
```bash
bash scripts/run_controlled_experiment_turn.sh \
    --doctor_model_names "model1" "model2" \
    --doctor_model_url "url1" "url2" \
    --doctor_model_key "key1" "key2" \
    --patient_model_name "model_name" \
    --patient_model_url "patient_url" \
    --patient_model_key "patient_key" \
    --judge_model_name "judge_model" \
    --judge_model_url "judge_url" \
    --judge_model_key "judge_key" \
    --num_patient 50 \
    --num_dialog_max_turn 3 \  # Options: 3, 8, 15
    --num_thread 16
```

### 3. Important Parameters

#### Required Parameters:
- `--doctor_model_names`: List of doctor model names
- `--doctor_model_url`: List of doctor model API URLs
- `--doctor_model_key`: List of doctor model API keys
- `--patient_model_name`: Patient model name
- `--patient_model_url`: Patient model API URL
- `--patient_model_key`: Patient model API key
- `--judge_model_name`: Judge model name
- `--judge_model_url`: Judge model API URL
- `--judge_model_key`: Judge model API key

#### Optional Parameters:
- `--num_patient`: Number of test patients (default: 10)
- `--num_dialog_max_turn`: Maximum dialogue turns (default: 8)
- `--num_thread`: Number of threads (default: 16)
- `--doctor_slow_think_mode`: Doctor slow think mode ("On"/"Off", default: "Off")
- `--patient_slow_think_mode`: Patient slow think mode ("On"/"Off", default: "On")
- `--run_dialog`: Whether to run dialogue generation ("True"/"False", default: "True")
- `--run_eval`: Whether to run evaluation ("True"/"False", default: "True")

### 4. Controlled Experiment Description
The controlled experiment script (dialog_turn_run.py) is mainly used to study the impact of dialogue turns on model performance:
- Fixed patient character as 00A
- Supports testing with 3, 8, and 15 dialogue turns
- Uses 50 data points from cases_counted_difficulty.json
- Supports testing multiple models:
  - qwen2.5-7b-instruct
  - gpt-4o-2024-11-20
  - Qwen2.5-72B-Instruct
  - deepseek-r1
  - deepseek-v3
