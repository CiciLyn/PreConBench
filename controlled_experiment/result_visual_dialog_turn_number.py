'''
Calculate the actual number of dialog turns in the dialog.
When the max dialogue turn number is 3, 8, 15 respectively.
'''

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def count_doctor_turns(dialog: str) -> int:
    """统计'[医生]'出现的次数"""
    return dialog.count('[病人]')

def get_model_dialog_turns(path, only_00A=False, max_cases=50):
    """遍历模型文件夹，统计每个模型的平均对话轮数"""
    model_turns = {}
    for model_name in os.listdir(path):
        model_dir = os.path.join(path, model_name)
        if not os.path.isdir(model_dir):
            continue
        dialog_file = os.path.join(model_dir, f'{model_name}_dialog.json')
        if not os.path.exists(dialog_file):
            continue
        with open(dialog_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        turns = []
        count = 0
        for item in data:
            if only_00A:
                # 只统计前50个00A
                if item.get('patient_info', {}).get('character_label') != '00A':
                    continue
                count += 1
                if count > max_cases:
                    break
            dialog = item.get('dialog', '')
            turns.append(count_doctor_turns(dialog))
        if turns:
            model_turns[model_name] = sum(turns) / len(turns)
        else:
            model_turns[model_name] = 0
    return model_turns

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # turn=3
    path_3 = os.path.join(base_path, 'results', 'dialog_turn', '3')
    turns_3 = get_model_dialog_turns(path_3)
    print('Dialog turn=3:')
    for model, avg_turn in turns_3.items():
        print(f'  {model}: {avg_turn:.2f}')
    # turn=8
    path_8 = os.path.join(base_path, '..', 'results')
    turns_8 = get_model_dialog_turns(path_8, only_00A=True, max_cases=50)
    print('Dialog turn=8 (first 50 with 00A):')
    for model, avg_turn in turns_8.items():
        print(f'  {model}: {avg_turn:.2f}')
    # turn=15
    path_15 = os.path.join(base_path, 'results', 'dialog_turn', '15')
    turns_15 = get_model_dialog_turns(path_15)
    print('Dialog turn=15:')
    for model, avg_turn in turns_15.items():
        print(f'  {model}: {avg_turn:.2f}')

    plot_dialog_turns(turns_3, turns_8, turns_15)

def plot_dialog_turns(turns_3, turns_8, turns_15, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)
    # 过滤掉huatuo_agent
    models = sorted([m for m in set(turns_3.keys()) | set(turns_8.keys()) | set(turns_15.keys()) if m != 'huatuo_agent'])
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['font.size'] = 14  # 字号大于12即可

    vals_3 = [turns_3.get(m, 0) for m in models]
    vals_8 = [turns_8.get(m, 0) for m in models]
    vals_15 = [turns_15.get(m, 0) for m in models]

    bars1 = ax.bar(x - width, vals_3, width, label='Turn 3', color='#2E86C1', alpha=0.8)
    bars2 = ax.bar(x,        vals_8, width, label='Turn 8', color='#E67E22', alpha=0.8)
    bars3 = ax.bar(x + width,vals_15, width, label='Turn 15', color='#27AE60', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Average Dialog Turns', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=14)
    ax.legend(fontsize=14)
    ax.set_ylim(0, max(vals_3 + vals_8 + vals_15) + 1)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dialog_turn_number_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(save_dir, 'dialog_turn_number_comparison.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    main()
