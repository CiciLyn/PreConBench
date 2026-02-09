"""
Evaluation module for medical dialogue and record analysis.

This module provides functionality for:
1. Record normalization and correctness
2. Dialogue quality assessment
3. Performance metrics calculation
"""

from typing import Dict, Tuple, List
import re
import time
from retry import retry

from core.models import GPTModel
from core.logger import logger


def character_match(text: str) -> str:
    """Remove non-Chinese, non-English, non-digit characters from the beginning of a string.

    Args:
        text: Input string to be processed

    Returns:
        Processed string with only Chinese, English, numbers and spaces at the beginning
    """
    return re.sub(r"^[^\u4e00-\u9fa5a-zA-Z0-9\s]+", "", text).strip()


@retry(tries=3, delay=1, backoff=2, logger=logger)
def _chat_with_retry(judge_model: GPTModel, messages: List[Dict]) -> Tuple[str, int, int]:
    """Helper function to call judge model's chat method with retry mechanism.
    
    Args:
        judge_model: The judge language model
        messages: List of message dictionaries
        
    Returns:
        Tuple containing response text, input tokens, and output tokens
    """
    return judge_model.chat(messages)


def evaluate_record_dialog(
    judge_model: GPTModel,
    prompts_judge: Dict[str, List[str]],
    record: str,
    dialog: str,
    gt_record_main: str,
    gt_record_present: str,
    gt_record_past: str,
    verbose: bool = False,
) -> Tuple[Dict[str, str], int, int]:
    """Evaluate a medical record and dialogue using the judge model.

    Args:
        judge_model: Judge language model
        prompts_judge: Dictionary of judge prompts, where each value is a list [system_prompt, user_prompt]
        record: Generated record to evaluate
        dialog: Generated dialogue to evaluate
        gt_record_main: Ground truth main complaint
        gt_record_present: Ground truth present illness
        gt_record_past: Ground truth past history
        verbose: Whether to print detailed information

    Returns:
        Tuple containing scores, input tokens, and output tokens
    """
    scores_record = {}
    total_in_tokens = 0
    total_out_tokens = 0

    # Extract and clean record sections
    try:
        if "[主诉]" in record:
            record_sections = {
                "main": character_match(
                    record.rsplit("[现病史]", 1)[0].strip().rsplit("[主诉]", 1)[1].strip()
                ),
                "present": character_match(
                    record.rsplit("[现病史]", 1)[1].rsplit("[既往史]", 1)[0].strip()
                ),
                "past": character_match(
                    record.rsplit("[既往史]", 1)[1]
                ),
            }
        elif "主诉" in record:
            record_sections = {
                "main": character_match(
                    record.rsplit("现病史", 1)[0].strip().rsplit("主诉", 1)[1].strip()
                ),
                "present": character_match(
                    record.rsplit("现病史", 1)[1].rsplit("既往史", 1)[0].strip()
                ),
                "past": character_match(
                    record.split("既往史", 1)[1]
                ),
            }
        else:
            # 如果找不到主诉标记，将整个记录作为主诉
            record_sections = {
                "main": character_match(record),
                "present": "",
                "past": "",
            }
    except Exception as e:
        logger.error(f"提取模型生成的记录部分时发生错误: {str(e)}")
        print(record)
        # 返回一个有效的记录部分结构
        record_sections = {
            "main": character_match(record),
            "present": "",
            "past": "",
        }

    gt_sections = {
        "main": gt_record_main,
        "present": gt_record_present,
        "past": gt_record_past,
    }

    for name, prompts in prompts_judge.items():
        if not any(
            keyword in name for keyword in ["record_norm", "record_corr", "dialog_norm", "record_truth"]
        ):
            continue

        sys_prompt, user_prompt = prompts

        if "record_norm" in name:
            user_prompt_format = user_prompt.format(model_record=record)
        elif "record_corr" in name:
            # Extract section type and metric type from the prompt name
            # Format: record_corr_{section_type}_{metric_type}
            parts = name.split("_")
            section_type = parts[2]  # main, present, or past
            metric_type = parts[3]  # TP, FP, or FN

            # Map section type to the correct variable names used in the prompt template
            section_map = {
                "main": {"model": "model_main", "true": "true_main"},
                "present": {"model": "model_present", "true": "true_present"},
                "past": {"model": "model_past", "true": "true_past"},
            }

            user_prompt_format = user_prompt.format(
                **{
                    section_map[section_type]["model"]: record_sections[section_type],
                    section_map[section_type]["true"]: gt_sections[section_type],
                    "metric_type": metric_type,
                }
            )
        elif "dialog_norm" in name:  # dialog_norm
            user_prompt_format = user_prompt.format(model_dialog=dialog)
        elif "record_truth" in name:  # record_truth
            user_prompt_format = user_prompt.format(record=record, dialog=dialog)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_format},
        ]

        ts = time.time()
        response, in_tokens, out_tokens = _chat_with_retry(judge_model, messages)
        te = time.time()
        logger.info(f"Judge API call time: {te - ts:.2f} s")
        if verbose:
            logger.info(f"evaluation response of {name}: {response}")
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens
        scores_record[name] = response

    return scores_record, total_in_tokens, total_out_tokens
