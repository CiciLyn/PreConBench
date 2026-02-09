"""
Argument parser for the medical dialogue benchmark system.

This module handles all command-line arguments and organizes them into four main categories:
1. Path parameters: Data paths and save directories
2. Model parameters: Model names, API keys, and URLs
3. Numerical parameters: Counts for patients, turns, and threads
4. Mode parameters: First visit or follow-up visit mode
"""

# Standard library imports
import argparse

# Local imports
from data.prompts.doctor import (
    doctor_task_description_prompt,
    final_doctor_record_prompt,
    doctor_slow_think_prompt,
)

from data.prompts.patient import (
    patient_task_description_prompt,
    patient_slow_think_prompt,
)

from data.prompts.judge_record import (
    record_nor_prompt_sys,
    record_nor_prompt_user,
    record_corr_main_TP_prompt_sys,
    record_corr_main_TP_prompt_user,
    record_corr_main_FP_prompt_sys,
    record_corr_main_FP_prompt_user,
    record_corr_main_FN_prompt_sys,
    record_corr_main_FN_prompt_user,
    record_corr_present_TP_prompt_sys,
    record_corr_present_TP_prompt_user,
    record_corr_present_FP_prompt_sys,
    record_corr_present_FP_prompt_user,
    record_corr_present_FN_prompt_sys,
    record_corr_present_FN_prompt_user,
    record_corr_past_TP_prompt_sys,
    record_corr_past_TP_prompt_user,
    record_corr_past_FP_prompt_sys,
    record_corr_past_FP_prompt_user,
    record_corr_past_FN_prompt_sys,
    record_corr_past_FN_prompt_user,
    # One-shot example
    record_nor_example_1,
    record_corr_main_TP_example_1, 
    record_corr_main_FP_example_1, 
    record_corr_main_FN_example_1, 
    record_corr_present_TP_example_1, 
    record_corr_present_FP_example_1, 
    record_corr_present_FN_example_1,
    record_corr_past_TP_example_1, 
    record_corr_past_FP_example_1, 
    record_corr_past_FN_example_1

)
from data.prompts.judge_dialog import (
    dialog_normative_sys,
    dialog_normative_user,
    # One-shot example
    dialog_normative_example_1,
)
from data.prompts.judge_basic import (
    record_truth_sys,
    record_truth_user,
    # One-shot example
    record_truth_example_1,
)


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Medical Dialogue Benchmark System")

    # Data path arguments
    parser.add_argument(
        "--bench_data_path",
        type=str,
        default="./data/bench_data.xlsx",
        help="Path to benchmark data Excel file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./result/",
        help="Directory to save all results",
    )
    # Test models
    parser.add_argument(
        "-doc_names",
        "--doctor_model_names",
        nargs="+",
        required=True,
        default=["deepseek-v3", "Qwen2.5-32B-Instruct", "qwen2.5-7b-instruct"],
        help="List of doctor model names",
    )
    parser.add_argument(
        "-doc_url",
        "--doctor_model_url",
        nargs="+",
        type=str,
        required=True,
        help="List of API URLs for doctor models",
    )
    parser.add_argument(
        "-doc_key",
        "--doctor_model_key",
        nargs="+",
        type=str,
        required=True,
        help="List of API keys for doctor models",
    )
    parser.add_argument(
        "-doc_think_mode",
        "--doctor_slow_think_mode",
        choices=[
            'On',
            'Off',
        ],
        default="Off",
        help="Whether to use the doctor slow think mode",
    )

    # Patient model
    parser.add_argument(
        "-p_name",
        "--patient_model_name",
        default="deepseek-v3",
        help="Name of the patient model",
    )
    parser.add_argument(
        "-p_url",
        "--patient_model_url",
        type=str,
        required=True,
        help="API URL for patient model",
    )
    parser.add_argument(
        "-p_key",
        "--patient_model_key",
        type=str,
        required=True,
        help="API key for patient model",
    )

    parser.add_argument(
        "-p_think_mode",
        "--patient_slow_think_mode",
        choices=[
            'On',
            'Off',
        ],
        default="Off",
        help="Whether to use the patient slow think mode",
    )


    # Judge model
    parser.add_argument(
        "-j_name",
        "--judge_model_name",
        default="gpt-4o-2024-11-20",
        help="Name of the judge model",
    )
    parser.add_argument(
        "-j_url",
        "--judge_model_url",
        type=str,
        required=True,
        help="API URL for judge model",
    )
    parser.add_argument(
        "-j_key",
        "--judge_model_key",
        type=str,
        required=True,
        help="API key for judge model",
    )

    # Prompt file paths
    parser.add_argument(
        "-p_prompt",
        "--patient_prompt",
        type=str,
        default="./data/prompts/patient.py",
        help="Path to patient prompt file",
    )
    parser.add_argument(
        "-doc_prompt",
        "--doctor_prompt",
        type=str,
        default="./data/prompts/doctor.py",
        help="Path to doctor prompt file",
    )
    parser.add_argument(
        "-jr_prompt",
        "--judge_record_prompt",
        type=str,
        default="./data/prompts/judge_record.py",
        help="Path to judge record prompt file",
    )
    parser.add_argument(
        "-jr_prompt_mode",
        "--judge_record_prompt_mode",
        choices=[
            "zero-shot",
            "one-shot",
        ],
        default="zero-shot",
        help="Prompt mode for judge record. Whether zero-shot or one-shot.",
    )
    parser.add_argument(
        "-jd_prompt",
        "--judge_dialog_prompt",
        type=str,
        default="./data/prompts/judge_dialog.py",
        help="Path to judge dialog prompt file",
    )

    # Numerical parameters
    parser.add_argument(
        "-num_p",
        "--num_patient",
        type=int,
        default=10,
        help="Number of patients to test",
    )
    parser.add_argument(
        "-num_turn",
        "--num_dialog_max_turn",
        type=int,
        default=8,
        help="Maximum number of dialogue turns",
    )
    parser.add_argument(
        "-num_thread",
        "--num_thread",
        type=int,
        default=16,
        help="Number of threads to use",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        default="main",
        help="Scenario mode: main or debug",
    )
    parser.add_argument(
        "--run_dialog",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Whether to run dialog generation (True/False)",
    )
    parser.add_argument(
        "--run_eval",
        type=str,
        choices=["True", "False"],
        default="True",
        help="Whether to run evaluation (True/False)",
    )

    return parser


def organize_parameters(args):
    """Organize parsed arguments into structured dictionaries."""
    # Path parameters
    paths = {
        "bench_data_path": args.bench_data_path,
        "save_dir": args.save_dir,
    }

    # Model parameters
    models = {
        "doctor_models": [
            {"model_name": name, "api_url": url, "api_key": key}
            for name, url, key in zip(
                args.doctor_model_names, args.doctor_model_url, args.doctor_model_key
            )
        ],
        "patient_model": {
            "model_name": args.patient_model_name,
            "api_url": args.patient_model_url,
            "api_key": args.patient_model_key,
        },
        "judge_model": {
            "model_name": args.judge_model_name,
            "api_url": args.judge_model_url,
            "api_key": args.judge_model_key,
        },
    }

    # Numerical parameters
    num_params = {
        "num_patient": args.num_patient,
        "num_turn": args.num_dialog_max_turn,
        "num_thread": args.num_thread,
    }

    # Mode parameters
    mode_params = {
        "mode": args.mode,
        "patient_slow_think_mode": args.patient_slow_think_mode,
        "doctor_slow_think_mode": args.doctor_slow_think_mode,
        "run_dialog": args.run_dialog == "True",
        "run_eval": args.run_eval == "True",
    }

    # Prompt configurations
    prompts_dialog = {
        "patient_system_prompt_template": patient_task_description_prompt,
        "doctor_system_prompt_template": doctor_task_description_prompt,
        "final_doctor_record_prompt_template": final_doctor_record_prompt,
        "patient_slow_think_prompt_template": patient_slow_think_prompt,
        "doctor_slow_think_prompt_template": doctor_slow_think_prompt,
    }
    if args.judge_record_prompt_mode == "one-shot":
        dialog_normative_example_user = dialog_normative_example_1 + dialog_normative_user
        record_nor_example_user = record_nor_example_1 + record_nor_prompt_user
        record_corr_TP_main_example_user = record_corr_main_TP_example_1 + record_corr_main_TP_prompt_user
        record_corr_FP_main_example_user = record_corr_main_FP_example_1 + record_corr_main_FP_prompt_user
        record_corr_FN_main_example_user = record_corr_main_FN_example_1 + record_corr_main_FN_prompt_user
        record_corr_TP_present_example_user = record_corr_present_TP_example_1 + record_corr_present_TP_prompt_user
        record_corr_FP_present_example_user = record_corr_present_FP_example_1 + record_corr_present_FP_prompt_user
        record_corr_FN_present_example_user = record_corr_present_FN_example_1 + record_corr_present_FN_prompt_user
        record_corr_TP_past_example_user = record_corr_past_TP_example_1 + record_corr_past_TP_prompt_user
        record_corr_FP_past_example_user = record_corr_past_FP_example_1 + record_corr_past_FP_prompt_user
        record_corr_FN_past_example_user = record_corr_past_FN_example_1 + record_corr_past_FN_prompt_user
        record_truth_example_user = record_truth_example_1 + record_truth_user
    elif args.judge_record_prompt_mode == "zero-shot": 
        dialog_normative_example_user = dialog_normative_user
        record_nor_example_user = record_nor_prompt_user
        record_corr_TP_main_example_user = record_corr_main_TP_prompt_user
        record_corr_FP_main_example_user = record_corr_main_FP_prompt_user
        record_corr_FN_main_example_user = record_corr_main_FN_prompt_user
        record_corr_TP_present_example_user = record_corr_present_TP_prompt_user
        record_corr_FP_present_example_user = record_corr_present_FP_prompt_user
        record_corr_FN_present_example_user = record_corr_present_FN_prompt_user
        record_corr_TP_past_example_user = record_corr_past_TP_prompt_user
        record_corr_FP_past_example_user = record_corr_past_FP_prompt_user
        record_corr_FN_past_example_user = record_corr_past_FN_prompt_user
        record_truth_example_user = record_truth_user
    prompts_judge = {
        "record_norm_prompt": [
            record_nor_prompt_sys, 
            record_nor_example_user
        ],
        "record_corr_main_TP_prompt": [
            record_corr_main_TP_prompt_sys,
            record_corr_TP_main_example_user,
        ],
        # "record_corr_main_FP_prompt": [
        #     record_corr_main_FP_prompt_sys,
        #     record_corr_FP_main_example_user,
        # ],
        "record_corr_main_FN_prompt": [
            record_corr_main_FN_prompt_sys,
            record_corr_FN_main_example_user,
        ],
        "record_corr_present_TP_prompt": [
            record_corr_present_TP_prompt_sys,
            record_corr_TP_present_example_user,
        ],
        # "record_corr_present_FP_prompt": [
        #     record_corr_present_FP_prompt_sys,
        #     record_corr_FP_present_example_user,
        # ],
        "record_corr_present_FN_prompt": [
            record_corr_present_FN_prompt_sys,
            record_corr_FN_present_example_user,
        ],
        "record_corr_past_TP_prompt": [
            record_corr_past_TP_prompt_sys,
            record_corr_TP_past_example_user,
        ],
        # "record_corr_past_FP_prompt": [
        #     record_corr_past_FP_prompt_sys,
        #     record_corr_FP_past_example_user,
        # ],
        "record_corr_past_FN_prompt": [
            record_corr_past_FN_prompt_sys,
            record_corr_FN_past_example_user,
        ],
        "dialog_normative_prompt": [
            dialog_normative_sys, 
            dialog_normative_example_user
        ],
        "record_truth_prompt": [
            record_truth_sys,
            record_truth_example_user,
        ],
    }

    return [paths, models, num_params, mode_params, prompts_dialog, prompts_judge]


# Parse arguments and organize parameters
parser = create_parser()
args = parser.parse_args()
parameters = organize_parameters(args)
