"""
Result Analysis Module

This module provides functionality to analyze and process evaluation results from medical dialogue benchmarks.
It processes JSON result files and generates various statistical metrics and scores.

Input: results/model_name/{model_name}_results.json
Output: Various statistical metrics in JSON format
"""

import json
import os
import sys
import re
from typing import Dict, List, Tuple, Union, Any

import numpy as np


# Constants
CHOICE_PERCENTAGE_MAP = {"A": 1, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0}
WEIGHT_RECORD_CORR = {"main": 30, "present": 50, "past": 20}

def read_json_file(file_path: str) -> list:
    """Read a JSON file with automatic encoding detection.

    Args:
        file_path: Path to the JSON file

    Returns:
        List containing the JSON data

    Raises:
        Exception: If file cannot be read with either encoding
    """
    try:
        with open(file_path, "r", encoding="gbk") as f:
            return json.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"无法读取文件 {file_path}: {str(e)}")


def extract_evaluate_scores(results_dir: str = "./results") -> Tuple[Dict, Dict]:
    """Extract and process evaluation scores from all model result files.

    Returns:
        Tuple containing:
        - Dictionary of model scores with averages
        - Dictionary of detailed model statistics
    """
    model_scores = {}
    model_stats = {}

    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        scores_file = os.path.join(model_dir, f"{model_name}_results.json")
        if not os.path.exists(scores_file):
            continue

        try:
            results = read_json_file(scores_file)
            # 过滤掉dialog或record为空的条目
            valid_results = [
                result for result in results
                if result.get("dialog") not in (None, "") and result.get("record") not in (None, "")
            ]
            scores = [result["evaluate_scores"] for result in valid_results if "evaluate_scores" in result]
            stats = analyze_scores(scores, model_name)

            model_stats[model_name] = stats
            model_scores[model_name] = calculate_model_averages(stats)

        except Exception as e:
            print(f"读取{model_name}的results.json时出错: {str(e)}")

    return model_scores, model_stats


def calculate_model_averages(stats: Dict) -> Dict:
    def safe_mean(lst):
        valid = [x for x in lst if x is not None]
        return sum(valid) / len(valid) if valid else 0

    record_correctness_recall_average = safe_mean(stats["record_correctness_recall_score"])
    record_norm_score_average = safe_mean(stats["record_norm_score"])
    dialog_norm_score_average = safe_mean(stats["dialog_norm_score"])
    record_truth_score_average = safe_mean(stats["record_truth_score"])

    return {
        "total_samples": len([x for x in stats["record_correctness_score"] if x is not None]),
        "record_correctness_recall_average": record_correctness_recall_average,
        "record_norm_average": record_norm_score_average,
        "record_truth_average": record_truth_score_average,
        "dialog_norm_average": dialog_norm_score_average,
    }


def analyze_scores(scores: List[Dict], model_name: str) -> Dict:
    stats = {
        "record_norm_score": [],
        "record_norm_choice": [],
        "dialog_norm_score": [],
        "dialog_norm_choice": [],
        "record_correctness_number": [],
        "record_correctness_score": [],
        "record_correctness_recall_score": [],
        "record_truth_score": [],
        "record_truth_choice": [],
        "record_label": [],
        "character_label": [],
    }

    for idx, score in enumerate(scores):
        # Process record norm scores
        record_norm_text = score.get("record_norm_prompt", "")
        record_norm_dict, record_norm_score = score_map_record_norm(
            record_norm_text, model_name, idx
        )
        stats["record_norm_choice"].append(record_norm_dict)
        stats["record_norm_score"].append(record_norm_score)

        # Process dialog norm scores
        dialog_norm_text = score.get("dialog_normative_prompt", "")
        dialog_norm_dict, dialog_norm_score = score_map_dialog_norm(dialog_norm_text, model_name, idx)
        stats["dialog_norm_choice"].append(dialog_norm_dict)
        stats["dialog_norm_score"].append(dialog_norm_score)

        # Process record truth scores
        record_truth_text = score.get("record_truth_prompt", "")
        record_truth_dict, record_truth_score = score_map_dialog_norm(record_truth_text, model_name, idx)
        stats["record_truth_choice"].append(record_truth_dict)
        stats["record_truth_score"].append(record_truth_score)

        # Process record correctness scores
        correctness_data = score_map_record_correctness(score, model_name, idx)
        stats["record_correctness_number"].append(correctness_data[0])
        stats["record_correctness_score"].append(correctness_data[2])  # 只存float
        stats["record_correctness_recall_score"].append(correctness_data[2])  # 只存float

        # Record labels
        stats["record_label"].append(score.get("word_count_level", ""))
        stats["character_label"].append(score.get("character_label", ""))

    return stats


def score_map_record_norm(text: str, model_name: str, idx: int) -> Tuple[Union[Dict, None], Union[float, None]]:
    try:
        pattern = r"##\s*选项\s*(?:\(|\s)*([A-F])(?:\)|\s)*"
        options = re.findall(pattern, text)
        if len(options) != 3:
            return None, None
        record_norm_dict = {"主诉": options[0], "现病史": options[1], "既往史": options[2]}
        record_norm_score = (
            WEIGHT_RECORD_CORR["main"] * CHOICE_PERCENTAGE_MAP[options[0]]
            + WEIGHT_RECORD_CORR["present"] * CHOICE_PERCENTAGE_MAP[options[1]]
            + WEIGHT_RECORD_CORR["past"] * CHOICE_PERCENTAGE_MAP[options[2]]
        )
        return record_norm_dict, record_norm_score
    except Exception:
        return None, None


def score_map_dialog_norm(text: str, model_name: str, idx: int) -> Tuple[Union[Dict, None], Union[float, None]]:
    try:
        pattern = r"##\s*选项\s*(?:\(|\s)*([A-F])(?:\)|\s)*"
        options = re.findall(pattern, text)
        if len(options) != 1:
            return None, None
        dialog_norm_dict = {"dialog_norm": options[0]}
        dialog_norm_score = 100 * CHOICE_PERCENTAGE_MAP[options[0]]
        return dialog_norm_dict, dialog_norm_score
    except Exception:
        return None, None


def score_map_record_correctness(
    score_dict: Dict, model_name: str, idx: int
) -> Tuple[Union[Dict, None], Union[Dict, None], Union[float, None]]:
    try:
        record_corr_number_dict = {}
        record_corr_dict = {}
        # Extract record correctness keys
        record_corr_keys = [key for key in score_dict.keys() if "record_corr" in key]
        pattern = r"##\s*信息项个数\s*(\d+(?:\.\d+)?)"

        # Process each record correctness key
        for key in record_corr_keys:
            match = re.search(pattern, score_dict[key])
            record_corr_number_dict[key] = float(match.group(1)) if match else 0.0

        # Calculate recall scores
        for section in ["main", "present", "past"]:
            tp = record_corr_number_dict.get(f"record_corr_{section}_TP_prompt", 0.0)
            fn = record_corr_number_dict.get(f"record_corr_{section}_FN_prompt", 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            record_corr_dict[section] = {
                "recall": recall
            }

        # Calculate total score with weights
        record_corr_score_recall = (
            WEIGHT_RECORD_CORR["main"] * record_corr_dict["main"]["recall"]
            + WEIGHT_RECORD_CORR["present"] * record_corr_dict["present"]["recall"]
            + WEIGHT_RECORD_CORR["past"] * record_corr_dict["past"]["recall"]
        )
        return record_corr_number_dict, record_corr_dict, record_corr_score_recall
    except Exception:
        return None, None, None


def score_by_record_label(model_stats: Dict) -> Dict:
    """Calculate scores grouped by record label (easy/hard) for each model separately.
    
    Returns:
        Dict: {
            "model_name": {
                "easy": {
                    "num_samples": int,
                    "averages": {
                        "record_norm": float,
                        "dialog_norm": float,
                        "record_truth": float,
                        "record_correctness_score": float
                    }
                },
                "hard": {
                    "num_samples": int,
                    "averages": {
                        "record_norm": float,
                        "dialog_norm": float,
                        "record_truth": float,
                        "record_correctness_score": float
                    }
                }
            },
            ...
        }
    """
    results = {}
    record_labels = ["easy", "hard"]
    
    for model_name, stats in model_stats.items():
        results[model_name] = {}
        
        # 获取每个难度类型的样本索引
        indices = {
            label: [i for i, l in enumerate(stats["record_label"]) if l == label]
            for label in record_labels
        }
        
        cared_keys = [
            "record_norm_score",
            "dialog_norm_score",
            "record_truth_score",
            "record_correctness_recall_score"
        ]
        
        # 计算每个难度类型的得分
        for record_label in record_labels:
            scores = {
                "record_norm_average": 0,
                "dialog_norm_average": 0,
                "record_truth_average": 0,
                "record_correctness_recall_average": 0
            }
            
            for key in cared_keys:
                valid_vals = [stats[key][i] for i in indices[record_label] if stats[key][i] is not None]
                if valid_vals:
                    scores[key.replace("_score", "_average")] = sum(valid_vals) / len(valid_vals)
            
            # 构建该难度类型的结果
            results[model_name][record_label] = {
                "num_samples": len(indices[record_label]),
                "averages": {
                    "record_norm": scores["record_norm_average"],
                    "dialog_norm": scores["dialog_norm_average"],
                    "record_truth": scores["record_truth_average"],
                    "record_correctness_score": scores["record_correctness_recall_average"]
                }
            }
    
    return results


def score_by_character_label(model_stats: Dict) -> Dict:
    """Calculate scores grouped by character label (00A/11B) for each model separately.
    
    Returns:
        Dict: {
            "model_name": {
                "00A": {
                    "num_samples": int,
                    "averages": {
                        "record_norm": float,
                        "dialog_norm": float,
                        "record_truth": float,
                        "record_correctness_score": float
                    }
                },
                "00B": {...},
                "11C": {...},
                "11D": {...}
            },
            ...
        }
    """
    characters = ["00A", "00B", "11C", "11D"]
    score_initial_dict = {
        "record_norm_average": 0,
        "dialog_norm_average": 0,
        "record_truth_average": 0,
        "record_correctness_recall_average": 0,
    }
    
    results = {}
    for model_name, stats in model_stats.items():
        results[model_name] = {}
        scores_dict = {char: score_initial_dict.copy() for char in characters}
        
        # 获取每个性格类型的样本索引
        indices = {
            label: [i for i, l in enumerate(stats["character_label"]) if l == label]
            for label in characters
        }
        
        keys = [
            "record_norm_score",
            "dialog_norm_score",
            "record_truth_score",
            "record_correctness_recall_score"
        ]
        
        # 计算每个性格类型的得分
        for character in characters:
            for key in keys:
                valid_vals = [stats[key][i] for i in indices[character] if stats[key][i] is not None]
                if valid_vals:
                    scores_dict[character][key.replace("_score", "_average")] = sum(valid_vals) / len(valid_vals)
            
            # 构建该性格类型的结果
            results[model_name][character] = {
                "num_samples": len(indices[character]),
                "averages": {
                    "record_norm": scores_dict[character]["record_norm_average"],
                    "dialog_norm": scores_dict[character]["dialog_norm_average"],
                    "record_truth": scores_dict[character]["record_truth_average"],
                    "record_correctness_score": scores_dict[character]["record_correctness_recall_average"],
                }
            }
    
    return results


def score_by_record_character_label(model_stats: Dict) -> Dict:
    """Calculate scores grouped by both record and character labels."""
    results = {}
    for model_name, stats in model_stats.items():
        results[model_name] = {}
        cared_keys = [
            "record_norm_score",
            "dialog_norm_score",
            "record_correctness_recall_score",
            "record_truth_score",
        ]
        cared_characters = ["00A", "00B", "11C", "11D"]
        for record_label in ["easy", "hard"]:
            for character_label in cared_characters:
                key = f"{record_label}_{character_label}"
                results[model_name][key] = {}
                indices = [
                    i
                    for i, label in enumerate(stats["record_label"])
                    if label == record_label
                    and stats["character_label"][i] == character_label
                ]
                for score_key in cared_keys:
                    if indices:
                        scores = [float(stats[score_key][i]) for i in indices if isinstance(stats[score_key][i], (int, float))]
                        score = sum(scores) / len(scores) if scores else 0
                    else:
                        score = 0
                    results[model_name][key][score_key] = score
    return results


def save_results(data: Dict, file_path: str) -> None:
    """Save results to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def calculate_detailed_scores(result: Dict, model_name: str, idx: int) -> Dict:
    """Calculate detailed scores for a single result."""
    scores = {}
    record_norm_text = result["evaluate_scores"]["record_norm_prompt"]
    record_norm_dict, record_norm_score = score_map_record_norm(record_norm_text, model_name, idx)
    if record_norm_dict is None:
        scores["record_norm"] = {
            "main_complaint": None,
            "present_illness": None,
            "past_history": None,
            "total_score": None,
        }
    else:
        scores["record_norm"] = {
            "main_complaint": record_norm_dict["主诉"],
            "present_illness": record_norm_dict["现病史"],
            "past_history": record_norm_dict["既往史"],
            "total_score": record_norm_score,
        }

    dialog_norm_text = result["evaluate_scores"]["dialog_normative_prompt"]
    dialog_norm_dict, dialog_norm_score = score_map_dialog_norm(dialog_norm_text, model_name, idx)
    if dialog_norm_dict is None:
        scores["dialog_norm"] = {
            "score": None,
            "total_score": None,
        }
    else:
        scores["dialog_norm"] = {
            "score": dialog_norm_dict["dialog_norm"],
            "total_score": dialog_norm_score,
        }

    correctness_data = score_map_record_correctness(result["evaluate_scores"], model_name, idx)
    if correctness_data[0] is None:
        scores["record_correctness"] = {
            "main": {"TP": None, "FP": None, "FN": None, "recall": None},
            "present": {"TP": None, "FP": None, "FN": None, "recall": None},
            "past": {"TP": None, "FP": None, "FN": None, "recall": None},
            "total_recall": None,
            "total_score": None,
        }
    else:
        main_tp = correctness_data[0].get("record_corr_main_TP_prompt", 0)
        main_fn = correctness_data[0].get("record_corr_main_FN_prompt", 0)
        main_recall = main_tp / (main_tp + main_fn) if (main_tp + main_fn) > 0 else 0

        present_tp = correctness_data[0].get("record_corr_present_TP_prompt", 0)
        present_fn = correctness_data[0].get("record_corr_present_FN_prompt", 0)
        present_recall = present_tp / (present_tp + present_fn) if (present_tp + present_fn) > 0 else 0

        past_tp = correctness_data[0].get("record_corr_past_TP_prompt", 0)
        past_fn = correctness_data[0].get("record_corr_past_FN_prompt", 0)
        past_recall = past_tp / (past_tp + past_fn) if (past_tp + past_fn) > 0 else 0

        scores["record_correctness"] = {
            "main": {
                "TP": main_tp,
                "FP": correctness_data[0].get("record_corr_main_FP_prompt", 0),
                "FN": main_fn,
                "recall": main_recall,
            },
            "present": {
                "TP": present_tp,
                "FP": correctness_data[0].get("record_corr_present_FP_prompt", 0),
                "FN": present_fn,
                "recall": present_recall,
            },
            "past": {
                "TP": past_tp,
                "FP": correctness_data[0].get("record_corr_past_FP_prompt", 0),
                "FN": past_fn,
                "recall": past_recall,
            },
            "total_recall": correctness_data[2],
            "total_score": f"Recall: {correctness_data[2]:.2f}" if correctness_data[2] is not None else None,
        }
    return scores


def main(results_dir: str = "./results"):
    """Main execution function."""
    # Extract and process scores
    model_scores, model_stats = extract_evaluate_scores(results_dir)

    # Add detailed scores to each result
    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        results_file = os.path.join(model_dir, f"{model_name}_results.json")
        if not os.path.exists(results_file):
            continue

        try:
            results = read_json_file(results_file)
            for result, idx in zip(results, range(len(results))):
                result["score"] = calculate_detailed_scores(result, model_name, idx)
            # Save with '_scored' suffix
            scored_results_file = os.path.join(
                model_dir, f"{model_name}_results_scored.json"
            )
            save_results(results, scored_results_file)
        except Exception as e:
            print(f"处理{model_name}的results.json时出错: {str(e)}")

    # Print model scores
    print("\n" + "=" * 50)
    print("模型评估结果".center(50))
    print("=" * 50)
    for model_name, score in model_scores.items():
        print(f"\n{model_name}:")
        print("-" * 30)
        print(f"病历准确性recall: {score['record_correctness_recall_average']:.2f}")
        print(f"病历规范性: {score['record_norm_average']:.2f}")
        print(f"对话规范性: {score['dialog_norm_average']:.2f}")
        print(f"病历真实性truth: {score['record_truth_average']:.2f}")

        print(
            f"平均得分: {np.mean([score['record_correctness_recall_average'], score['record_norm_average'], score['dialog_norm_average']]):.2f}"
        )

    # Save results to files
    save_results(model_scores, f"{results_dir}/models_scores.json")
    save_results(model_stats, f"{results_dir}/model_stats.json")

    # Calculate and save record-character label results
    record_character_label_results = score_by_record_character_label(model_stats)
    save_results(
        record_character_label_results, f"{results_dir}/record_character_label_results.json"
    )

    # Calculate and print record label results
    record_label_results = score_by_record_label(model_stats)
    print("\n" + "=" * 50)
    print("按病历难度分类结果".center(50))
    print("=" * 50)
    print("\n简单病历:")
    print("-" * 30)
    for model_name in record_label_results.keys():
        print(f"模型: {model_name}")
        print(f"病历规范性: {record_label_results[model_name]['easy']['averages']['record_norm']:.2f}")
        print(f"对话规范性: {record_label_results[model_name]['easy']['averages']['dialog_norm']:.2f}")
        print(f"病历真实性truth: {record_label_results[model_name]['easy']['averages']['record_truth']:.2f}")
        print(
            f"病历recall精度: {record_label_results[model_name]['easy']['averages']['record_correctness_score']:.2f}"
        )
    print("-" * 30)
    print("\n困难病历:")
    for model_name in record_label_results.keys():
        print("-" * 30)
        print(f"模型: {model_name}")
        print(f"病历规范性: {record_label_results[model_name]['hard']['averages']['record_norm']:.2f}")
        print(f"对话规范性: {record_label_results[model_name]['hard']['averages']['dialog_norm']:.2f}")
        print(f"病历真实性truth: {record_label_results[model_name]['hard']['averages']['record_truth']:.2f}")
        print(
            f"病历recall精度: {record_label_results[model_name]['hard']['averages']['record_correctness_score']:.2f}"
        )

    # Calculate and print character label results
    character_label_results = score_by_character_label(model_stats)
    print("\n" + "=" * 50)
    print("按病人性格分类结果".center(50))
    print("=" * 50)
    for model_name in character_label_results.keys():
        print("-" * 30)
        print(f"模型: {model_name}")
        for character in character_label_results[model_name].keys():
            print(f"病人性格: {character}")
            print(
                f"病历recall精度: {character_label_results[model_name][character]['averages']['record_correctness_score']:.2f}"
            )
            print(
                f"病历规范性: {character_label_results[model_name][character]['averages']['record_norm']:.2f}"
            )
            print(
                f"对话规范性: {character_label_results[model_name][character]['averages']['dialog_norm']:.2f}"
            )
            print(
                f"病历真实性truth: {character_label_results[model_name][character]['averages']['record_truth']:.2f}"
            )

    # Save additional results
    save_results(record_label_results, f"{results_dir}/record_label_results.json")
    save_results(character_label_results, f"{results_dir}/character_label_results.json")


def by_record_difficulty_character_label(path:str) -> Dict:
    """Calculate record label results.

    Args:
        path: Path to the file containing model statistics

    Returns:
        Dictionary containing record label results
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化结果字典
    results = {}
    metrics = [
        "record_correctness_recall_score",
        "record_norm_score",
        "dialog_norm_score",
        "record_truth_score"
    ]
    
    # 对每个组合计算平均值
    for record_label in ["easy", "hard"]:
        for character_label in ["00A", "00B", "11C", "11D"]:
            key = f"{record_label}_{character_label}"
            results[key] = {}
            
            # 对每个指标计算平均值
            for metric in metrics:
                values = []
                for model_name in data.keys():
                    if key in data[model_name]:
                        values.append(data[model_name][key][metric])
                if values:
                    results[key][metric] = sum(values) / len(values)
    
    return results

def model_tokens(results_dir: str = "./results") -> dict:
    """
    统计results目录下所有模型的对话token和评测token各key的总和。
    遍历每个模型文件夹下的model_name_results_scored.json，
    分别累加dialog_tokens（patient_in, patient_out, doctor_in, doctor_out）
    和evaluate_tokens（input_tokens, output_tokens）每个key的值。
    返回所有模型的详细token统计，并保存为model_tokens_summary.json。
    """
    all_model_tokens = {}
    dialog_token_keys = ["patient_in", "patient_out", "doctor_in", "doctor_out"]
    eval_token_keys = ["input_tokens", "output_tokens"]
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        scored_file = os.path.join(model_path, f"{model_dir}_results_scored.json")
        if not os.path.exists(scored_file):
            continue
        try:
            with open(scored_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            print(f"读取{scored_file}失败: {e}")
            continue
        dialog_token_sum = {k: 0 for k in dialog_token_keys}
        eval_token_sum = {k: 0 for k in eval_token_keys}
        for item in results:
            # 累加dialog_tokens
            dialog_tokens = item.get("dialog_tokens", {})
            for k in dialog_token_keys:
                v = dialog_tokens.get(k, 0)
                if isinstance(v, (int, float)):
                    dialog_token_sum[k] += v
            # 累加evaluate_tokens
            eval_tokens = item.get("evaluate_tokens", {})
            for k in eval_token_keys:
                v = eval_tokens.get(k, 0)
                if isinstance(v, (int, float)):
                    eval_token_sum[k] += v
        all_model_tokens[model_dir] = {
            "dialog_tokens": dialog_token_sum,
            "evaluate_tokens": eval_token_sum
        }
    # 保存为json
    with open(os.path.join(results_dir, "model_tokens_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_model_tokens, f, ensure_ascii=False, indent=2)
    return all_model_tokens

if __name__ == "__main__":
    # main()
    # main_result_dir = 'results'
    # main(results_dir=main_result_dir)
    # result_path = f'{main_result_dir}/record_character_label_results.json'
    # results = by_record_difficulty_character_label(result_path)
    # print("=" * 20, "按病历难度和病人性格分类结果", "=" * 20)
    # from pprint import pprint
    # pprint(results)

    # # 简单病例和简单病人
    # all_model_tokens = model_tokens(main_result_dir)


    abllation_results_dir = 'controlled_experiemnt/results/different_judge/Qwen2.5-72B-Instruct'
    main(results_dir=abllation_results_dir)

    result_path = f'{abllation_results_dir}/record_character_label_results.json'
    results = by_record_difficulty_character_label(result_path)
    print("=" * 20, "按病历难度和病人性格分类结果", "=" * 20)
    from pprint import pprint
    pprint(results)

    # 简单病历和简单病人 00A
    all_model_tokens = model_tokens(abllation_results_dir)

