'''
Same code from core/main.py
See the influence of different judge models.
Directly use the dialogue results/{model_name}/{model_name}_results.json.
Only dialogues with 00A character label are selected.
For each doctor model, only select the top 50 dialogue and its evaluation result. 

The judge models are:
    - gpt-4o-2024-11-20
    - deepseek-r1

'''

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





def run_evaluation(patient_df, judge_model, prompts_judge, dialog_json_path, num_params, eval_save_path, verbose=False):
    """多线程评测，从json读取对话，遇到无效对话用空EvaluationResult占位，保证index对齐"""
    logger.info(f"开始评测，读取对话文件: {dialog_json_path}_dialog.json")
    with open(f"{dialog_json_path}_dialog.json", "r", encoding="utf-8") as f:
        all_dialog_results = json.load(f)
        # 只保留character_label为00A的对话
        dialog_results = [item for item in all_dialog_results 
                        if item.get("patient_character_label") == "00A" or item.get("patient_info", {}).get("character_label") == "00A"]
    
    # 检查已存在的评测json
    existing_eval = {}
    if os.path.exists(eval_save_path):
        with open(eval_save_path, "r", encoding="utf-8") as f:
            all_existing_eval = json.load(f)
            # 只保留character_label为00A的评测结果
            existing_eval = {item["index"]: item for item in all_existing_eval 
                            if item.get("patient_character_label") == "00A"}
            
    existing_dialog = {}
    if os.path.exists(f"{dialog_json_path}_dialog.json"):
        with open(f"{dialog_json_path}_dialog.json", "r", encoding="utf-8") as f:
            existing_dialog = {item["index"]: item for item in json.load(f)
                               if item.get("patient_character_label") == "00A" or item.get("patient_info", {}).get("character_label") == "00A"}
    
    # 只处理未评测的对话
    tasks_evaluate = []
    for idx, row in patient_df.iterrows():
        # 已经评测过，跳过
        if row["character_label"] != "00A":
            continue
        if idx in existing_eval and existing_eval[idx].get("evaluate_scores"):
            continue
        result = existing_dialog.get(4 * idx)
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
                        "dialog": existing_dialog[4 * idx]["dialog"],
                        "record": existing_dialog[4 * idx]["record"],
                        "patient_info": existing_dialog[4 * idx]["patient_info"],
                        "patient_character": patient_df.loc[idx]["character"],
                        "patient_character_label": patient_df.loc[idx]["character_label"],
                        "dialog_tokens": existing_dialog[4 * idx]["dialog_tokens"],
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
                    "dialog": existing_dialog[4 * idx]["dialog"],
                    "record": existing_dialog[4 * idx]["record"],
                    "patient_info": existing_dialog[4 * idx]["patient_info"],
                    "patient_character": patient_df.loc[idx]["character"],
                    "patient_character_label": patient_df.loc[idx]["character_label"],
                    "dialog_tokens": existing_dialog[4 * idx]["dialog_tokens"],
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
        base_path = f"{paths['save_dir']}/{judge_model.model_name}/{doctor_model.model_name}"
        dialogue_path = f"./results/{doctor_model.model_name}"
        dialog_json_path = f"{dialogue_path}/{doctor_model.model_name}"
        eval_json_path = f"{base_path}/{doctor_model.model_name}_results.json"
        # 判断运行模式
        run_eval = mode_params.get("run_eval", True)
        if run_eval:
            run_evaluation(patient_df, judge_model, prompts_judge, dialog_json_path, num_params, eval_json_path, verbose=False)


if __name__ == "__main__":
    main()
