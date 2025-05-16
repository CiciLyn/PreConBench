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