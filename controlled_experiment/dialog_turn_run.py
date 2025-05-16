'''
Same code from core/main.py
See the influence of dialog turn: 3, 8, 15.
On the 50 data points of cases_counted_difficulty.json ( the PreConBench data)

The patient character is set to be only 00A. 
The tested models are: 
    - qwen2.5-7b-instruct
    - gpt-4o-2024-11-20
    - Qwen2.5-72B-Instruct
    - deepseek-r1
    - deepseek-v3
'''

"""
Main script for the medical dialogue benchmark system.

This script performs the following functions:
1. Parse input parameters
2. Read and process data
3. Initialize models
4. Process dialogues and evaluations
5. Save results to disk
"""

# Standard library imports
import os
import sys
import time
import json
from typing import Union, List, Dict, Optional, Tuple
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import pandas as pd
import threading

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Local imports
from core.logger import logger
from core.args_input import parameters
from core.models import GPTModel, AgentModel, HuatuoModel, PatientSlowThinkModel, DoctorSlowThinkModel
from core.data_utils import read_patient_data
from core.dialog_manager import DialogManager, DialogResult
from core.evaluate import evaluate_record_dialog


@dataclass
class PatientDialogResult:
    """Container for patient dialog processing results.

    Attributes:
        dialog: The complete dialog text
        medical_record: The generated medical record
        dialog_tokens: Dictionary tracking token usage for the dialog
        doctor_messages: List of doctor's messages
        patient_messages: List of patient's messages
    """

    dialog: str
    medical_record: str
    dialog_tokens: Dict[str, int]
    doctor_messages: List[Dict[str, str]]
    patient_messages: List[Dict[str, str]]


@dataclass
class EvaluationResult:
    """Container for evaluation results.

    Attributes:
        scores: Dictionary of evaluation scores
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used
    """

    scores: Dict[str, float]
    input_tokens: int
    output_tokens: int


def process_one_patient_dialog(
    patient_model: GPTModel,
    doctor_model: GPTModel,
    prompts: Dict[str, str],
    num_params: Dict[str, int],
    row: pd.Series,
    idx: int,
    verbose: bool = False,
) -> Optional[PatientDialogResult]:
    """Process a single patient's dialogue.

    Args:
        patient_model: The language model instance for the patient role
        doctor_model: The language model instance for the doctor role
        prompts: Dictionary containing prompt templates:
            - patient_system_prompt_template: Template for patient's system prompt
            - doctor_system_prompt_template: Template for doctor's system prompt
            - final_doctor_record_prompt_template: Template for medical record generation
        num_params: Dictionary containing numerical parameters:
            - num_turn: Maximum number of dialogue turns
        row: Pandas Series containing patient information:
            - 姓名: Patient's name
            - 性别: Patient's gender
            - 年龄: Patient's age
            - 职业: Patient's occupation
            - 第一科室: Department
            - character: Patient's personality
            - 主诉_extracted: Chief complaint
            - 现病史_extracted: Present illness history
            - 既往史_extracted: Past medical history
        idx: Index of the patient in the dataset
        verbose: Whether to print detailed logs

    Returns:
        PatientDialogResult containing:
            - dialog: The complete dialogue text
            - medical_record: The generated medical record
            - dialog_tokens: Dictionary tracking token usage for the dialog
            - doctor_messages: List of doctor's messages
            - patient_messages: List of patient's messages
        Returns None if dialogue generation failed
    """
    dialog_manager = DialogManager(
        patient_llm=patient_model,
        doctor_llm=doctor_model,
        patient_system_prompt_template=prompts["patient_system_prompt_template"],
        patient_slow_think_prompt_template=prompts["patient_slow_think_prompt_template"],
        doctor_system_prompt_template=prompts["doctor_system_prompt_template"],
        doctor_slow_think_prompt_template=prompts["doctor_slow_think_prompt_template"],
        final_doctor_record_prompt_template=prompts[
            "final_doctor_record_prompt_template"
        ],
        max_turn_num=num_params["num_turn"],
        patient_info=row.to_dict(),
    )

    start_time = time.time()
    dialog_result: DialogResult = dialog_manager.generate_dialog(verbose=verbose)
    (dialog, medical_record, dialog_tokens) = (
        dialog_result.dialog,
        dialog_result.medical_record,
        dialog_result.tokens_dict,
    )
    api_time = time.time() - start_time
    if verbose:
        logger.info(f"API调用完成 - 病人ID: {idx}, 耗时: {api_time:.2f}秒")

    if not medical_record or not dialog:
        logger.error(f"未获得有效对话结果 (病人ID: {idx})")
        return None

    return PatientDialogResult(
        dialog=dialog,
        medical_record=medical_record,
        dialog_tokens=dialog_tokens,
        doctor_messages=dialog_manager.doctor_memory.messages,
        patient_messages=dialog_manager.patient_memory.messages,
    )


def process_one_patient_evaluate(
    judge_model: GPTModel,
    prompts_judge: Dict[str, str],
    dialog_record_result: Tuple[str, str],
    row: pd.Series,
    idx: int,
    verbose: bool = False,
) -> Optional[EvaluationResult]:
    """Evaluate a single patient's dialogue and record.

    Args:
        judge_model: The language model instance for evaluation
        prompts_judge: Dictionary containing judge prompt templates
        dialog_record_result: Tuple containing:
            - dialog: The complete dialogue text
            - medical_record: The generated medical record
        row: Pandas Series containing patient information:
            - 主诉: Chief complaint
            - 现病史: Present illness history
            - 既往史: Past medical history
            - record_label: Record label
            - character_label: Character label
        idx: Index of the patient in the dataset
        verbose: Whether to print detailed logs

    Returns:
        EvaluationResult containing:
            - scores: Dictionary of evaluation scores
            - input_tokens: Number of input tokens used
            - output_tokens: Number of output tokens used
        Returns None if evaluation failed
    """
    gt_main_easy = row["主诉"]
    gt_present_easy = row["现病史"]
    gt_past_easy = row["既往史"]
    model_dialog, model_record = dialog_record_result
    if model_dialog is None or model_record is None:
        return EvaluationResult(scores={}, input_tokens=0, output_tokens=0)
    scores, in_tokens, out_tokens = evaluate_record_dialog(
        judge_model=judge_model,
        prompts_judge=prompts_judge,
        record=model_record,
        dialog=model_dialog,
        gt_record_main=gt_main_easy,
        gt_record_present=gt_present_easy,
        gt_record_past=gt_past_easy,
        verbose=verbose,
    )

    if scores is None:
        return None

    scores["record_label"] = row["record_difficulty_level"]
    scores["word_count_level"] = row["word_count_level"]
    scores["character_label"] = row["character_label"]
    return EvaluationResult(
        scores=scores, input_tokens=in_tokens, output_tokens=out_tokens
    )


def save_results(results, file_path):
    """Save results to a JSON file.

    Args:
        results: Data to save
        file_path: Path to save the file
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def save_messages(messages: List[Dict[str, str]], file_path: str) -> None:
    """Save messages to a JSON file.

    Args:
        messages: List of message dictionaries
        file_path: Path to save the file
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def init_model(
    model_config: Union[dict, List[dict]], model_type: Union[str, List[str]]
):
    """Initialize models based on configuration.

    Args:
        model_config: Model configuration

    Returns:
        Initialized model
    """
    if isinstance(model_config, dict):
        model_configs = [model_config]
    if isinstance(model_type, str):
        model_types = [model_type]

    for config, model_type in zip(model_configs, model_types):
        if (
            config["model_name"] in {"deepseek-r1"}
            and config.get("temperature", 0.0) == 0.0
        ):
            config["temperature"] = 0.6
            logger.info(
                f"It is recommended to set the temperature to 0.6 for the {config['model_name']} model."
            )
        if (
            config["model_name"] in {"Qwen2.5-72B-Instruct"}
            and config.get("max_tokens", 8192) == 8192
        ):
            config["max_tokens"] = 4096
            logger.info(
                f"It is recommended to set the max_tokens to 4096 for the {config['model_name']} model."
            )
        if config["model_name"] in {"huatuo_agent"} and model_type == "gpt_model":
            model_type = "huatuo_model"
            logger.info(
                f"It is recommended to use Huatuo Model as the doctor model for the {config['model_name']} model."
            )

    if model_type == "gpt_model":
        assert len(model_configs) == 1
        return GPTModel(model_configs[0])
    elif model_type == "agent_model":
        return AgentModel(model_configs)
    elif model_type == "huatuo_model":
        return HuatuoModel(model_configs[0])
    elif model_type == "patient_slow_think":
        return PatientSlowThinkModel(model_configs[0])
    elif model_type == "doctor_slow_think":
        return DoctorSlowThinkModel(model_configs[0])
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def run_dialog_generation(patient_df, patient_model, doctor_model, prompts_dialog, num_params, save_path, verbose=False):
    """多线程生成对话并保存json"""
    logger.info(f"开始对话生成，模型: {doctor_model.model_name}")
    
    # 检查已存在的对话json
    existing_dialog = {}
    existing_patient_messages = []
    existing_doctor_messages = []
    
    # 检查对话文件是否存在
    dialog_path = f"{save_path}_dialog.json"
    if os.path.exists(dialog_path):
        with open(dialog_path, "r", encoding="utf-8") as f:
            existing_dialog = {item["index"]: item for item in json.load(f)}
        logger.info(f"已存在对话结果，共{len(existing_dialog)}条")
    
    # 检查消息文件是否存在
    patient_messages_path = f"{save_path}_patient_messages.json"
    if os.path.exists(patient_messages_path):
        with open(patient_messages_path, "r", encoding="utf-8") as f:
            existing_patient_messages = json.load(f)
    
    doctor_messages_path = f"{save_path}_doctor_messages.json"
    if os.path.exists(doctor_messages_path):
        with open(doctor_messages_path, "r", encoding="utf-8") as f:
            existing_doctor_messages = json.load(f)
    
    # 只处理未生成的对话
    tasks_dialog = []
    for idx, row in patient_df.iterrows():
        # 如果已经生成过对话，跳过
        if idx in existing_dialog and existing_dialog[idx].get("dialog") and existing_dialog[idx].get("record"):
            continue
        tasks_dialog.append((patient_model, doctor_model, prompts_dialog, num_params, row, idx))
    
    logger.info(f"总对话数: {len(patient_df)}，已生成数: {len(existing_dialog)}，待生成数: {len(tasks_dialog)}")
    
    # 如果没有需要生成的对话，直接返回
    if not tasks_dialog:
        logger.info("没有需要生成的对话，直接返回")
        return list(existing_dialog.values())
    
    # 多线程生成对话
    dialog_results = []
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), num_params["num_thread"])) as dialog_executor:
        def dialog_worker(args):
            try:
                return process_one_patient_dialog(*args, verbose=verbose)
            except Exception as e:
                logger.error(f"生成对话，病人ID: {args[-1]} 时发生错误: {str(e)}")
                return None
        dialog_results = list(dialog_executor.map(dialog_worker, tasks_dialog))
    
    # 合并新旧结果
    dialog_json = []
    patient_messages = []
    doctor_messages = []
    
    # 先添加已存在的结果
    for idx in sorted(existing_dialog.keys()):
        dialog_json.append(existing_dialog[idx])
        if idx < len(existing_patient_messages):
            patient_messages.append(existing_patient_messages[idx])
        if idx < len(existing_doctor_messages):
            doctor_messages.append(existing_doctor_messages[idx])
    
    # 添加新生成的结果
    for idx, row in patient_df.iterrows():
        if idx not in existing_dialog:
            dialog_result = dialog_results.pop(0)
            if dialog_result is None:
                dialog_json.append({"index": idx, "dialog": None, "record": None})
                patient_messages.append([])
                doctor_messages.append([])
            else:
                dialog_json.append({
                    "index": idx,
                    "dialog": dialog_result.dialog,
                    "record": dialog_result.medical_record,
                    "patient_info": row.to_dict(),
                    "dialog_tokens": dialog_result.dialog_tokens,
                    # "doctor_messages": dialog_result.doctor_messages,
                    # "patient_messages": dialog_result.patient_messages
                })
                patient_messages.append(dialog_result.patient_messages)
                doctor_messages.append(dialog_result.doctor_messages)
    
    # 保存结果
    save_results(dialog_json, dialog_path)
    save_messages(patient_messages, patient_messages_path)
    save_messages(doctor_messages, doctor_messages_path)
    logger.info(f"对话生成完成，已保存到 {save_path}")
    return dialog_json


def run_evaluation(patient_df, judge_model, prompts_judge, dialog_json_path, num_params, eval_save_path, verbose=False):
    """多线程评测，从json读取对话，遇到无效对话用空EvaluationResult占位，保证index对齐"""
    logger.info(f"开始评测，读取对话文件: {dialog_json_path}_dialog.json")
    with open(f"{dialog_json_path}_dialog.json", "r", encoding="utf-8") as f:
        dialog_results = json.load(f)
    
    # 检查已存在的评测json
    existing_eval = {}
    if os.path.exists(eval_save_path):
        with open(eval_save_path, "r", encoding="utf-8") as f:
            existing_eval = {item["index"]: item for item in json.load(f)}
        logger.info(f"已存在评测结果，共{len(existing_eval)}条")
    existing_dialog = {}
    if os.path.exists(f"{dialog_json_path}_dialog.json"):
        with open(f"{dialog_json_path}_dialog.json", "r", encoding="utf-8") as f:
            existing_dialog = {item["index"]: item for item in json.load(f)}
    
    # 只处理未评测的对话
    tasks_evaluate = []
    for idx, row in patient_df.iterrows():
        # 已经评测过，跳过
        if idx in existing_eval and existing_eval[idx].get("evaluate_scores"):
            continue
        result = existing_dialog.get(idx)
        # 添加到待评测列表
        tasks_evaluate.append((judge_model, prompts_judge, (result["dialog"], result["record"]), row, idx))
    
    logger.info(f"总对话数: {len(dialog_results)}，已评测数: {len(existing_eval)}，待评测数: {len(tasks_evaluate)}")
    
    # 如果没有需要评测的对话，直接返回
    if not tasks_evaluate:
        logger.info("没有需要评测的对话，直接返回")
        return list(existing_eval.values())
    
    # 多线程评测
    def eval_worker(args):
        try:
            return process_one_patient_evaluate(*args, verbose=verbose)
        except Exception as e:
            logger.error(f"评测时发生错误: {str(e)}")
            return EvaluationResult(scores={}, input_tokens=0, output_tokens=0)
    
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), num_params["num_thread"])) as eval_executor:
        eval_results = list(eval_executor.map(eval_worker, tasks_evaluate))
    
    # 合并新旧结果
    eval_json = []
    # eval_results_iter = iter(eval_results)


    # 先添加已存在的结果
    for idx in sorted(existing_eval.keys()):
        eval_json.append(existing_eval[idx])

    # 添加新评测的结果  
    for idx, row in patient_df.iterrows():
        if idx not in existing_eval:
            eval_result = eval_results.pop(0)
            if eval_result is None or eval_result.scores is None:
                eval_json.append(
                    {
                        "index": idx,
                        "dialog": existing_dialog[idx]["dialog"],
                        "record": existing_dialog[idx]["record"],
                        "patient_info": existing_dialog[idx]["patient_info"],
                        # "patient_character": patient_df.loc[idx]["character"],
                        # "patient_character_label": patient_df.loc[idx]["character_label"],
                        "dialog_tokens": existing_dialog[idx]["dialog_tokens"],
                        "evaluate_scores": {},
                        "evaluate_tokens": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    }
                )
            else:
                entry = {
                    "index": idx,
                    "dialog": existing_dialog[idx]["dialog"],
                    "record": existing_dialog[idx]["record"],
                    "patient_info": existing_dialog[idx]["patient_info"],
                    # "patient_character": patient_df.loc[idx]["character"],
                    # "patient_character_label": patient_df.loc[idx]["character_label"],
                    "dialog_tokens": existing_dialog[idx]["dialog_tokens"],
                    "evaluate_scores": eval_result.scores,
                    "evaluate_tokens": {
                        "input_tokens": eval_result.input_tokens,
                        "output_tokens": eval_result.output_tokens,
                    }
                }
                eval_json.append(entry)
    save_results(eval_json, eval_save_path)
    logger.info(f"评测完成，已保存到 {eval_save_path}")
    return eval_json


def main():
    """主执行函数，支持只跑对话、只跑评测、或全流程"""
    # Parse parameters
    paths, models, num_params, mode_params, prompts_dialog, prompts_judge = parameters
    logger.info(f"病人模型为: {models['patient_model']['model_name']}")
    logger.info(f"评估模型为: {models['judge_model']['model_name']}")
    # Read data and initialize models
    patient_df = read_patient_data(paths["bench_data_path"], num_params["num_patient"])
    if mode_params["patient_slow_think_mode"] == "On":
        patient_model = init_model(models["patient_model"], "patient_slow_think")
    else:
        patient_model = init_model(models["patient_model"], "gpt_model")
    if mode_params["doctor_slow_think_mode"] == "On":
        doctor_models = [init_model(model, "doctor_slow_think") for model in models["doctor_models"]]
    else:
        doctor_models = [init_model(model, "gpt_model") for model in models["doctor_models"]]
    judge_model = init_model(models["judge_model"], "gpt_model")
    for doctor_model in doctor_models:
        logger.info(f"处理医生模型: {doctor_model.model_name}")
        base_path = f"{paths['save_dir']}/{num_params['num_turn']}/{doctor_model.model_name}"
        dialog_json_path = f"{base_path}/{doctor_model.model_name}"
        eval_json_path = f"{base_path}/{doctor_model.model_name}_results.json"
        # 判断运行模式
        run_dialog = mode_params.get("run_dialog", True)
        run_eval = mode_params.get("run_eval", True)
        if run_dialog:
            run_dialog_generation(patient_df, patient_model, doctor_model, prompts_dialog, num_params, dialog_json_path, verbose=False)
        if run_eval:
            run_evaluation(patient_df, judge_model, prompts_judge, dialog_json_path, num_params, eval_json_path, verbose=False)


if __name__ == "__main__":
    main()
