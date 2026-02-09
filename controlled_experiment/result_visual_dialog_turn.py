'''
visualization of the result of dialog turn: 3, 8, 15.
data path: 
    dialog_turn_3: ablation study/results/dialog_turn/3/
    doalog_turn_8: results/
    dialog_turn_15: ablation study/results/dialog_turn/15/

data amount: 50 patients with 00A trait only. 
'''

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def extract_scores_from_json(json_data: List[Dict], metrics: List[str] = None) -> Dict[str, float]:
    """从JSON数据中提取指定指标的平均分数"""
    if metrics is None:
        metrics = ['record_norm', 'dialog_norm', 'record_correctness']
    
    scores = {metric: [] for metric in metrics}
    
    for item in json_data:
        # 检查dialogue和record是否都为有效值
        if item.get('dialog') is None or item.get('dialog') == 'None':
            continue
            
        if item.get('score'):
            for metric in metrics:
                if f'{metric}' in item['score']:
                    # 从prompt中提取分数
                    prompt = item['score'][f'{metric}']
                    if 'total_recall' in prompt.keys() and metric == 'record_correctness':
                        try:
                            score = float(prompt['total_recall'])  # 将分数转换为浮点数
                            if not np.isnan(score):  # 检查是否为有效数值
                                scores[metric].append(score)
                        except (ValueError, TypeError):
                            # 如果转换失败，跳过这个分数
                            continue
                    elif 'total_score' in prompt.keys() and metric != 'record_correctness':
                        try:
                            score = float(prompt['total_score'])
                            if not np.isnan(score):  # 检查是否为有效数值
                                scores[metric].append(score)
                        except (ValueError, TypeError):
                            # 如果转换失败，跳过这个分数
                            continue
                    else:
                        continue
    
    # 计算平均值，只考虑非None值
    return {metric: np.mean(scores[metric]) if scores[metric] else 0 for metric in metrics}

def read_model_scores_turn_3_15(base_path: str, turn: int) -> Dict[str, Dict[str, float]]:
    """读取turn=3或15时的模型分数"""
    scores_path = os.path.join(base_path, f'ablation study/results/dialog_turn/{turn}/models_scores.json')
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"找不到文件: {scores_path}")
    
    with open(scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 定义键名映射
    key_mapping = {
        'record_norm_average': 'record_norm',
        'dialog_norm_average': 'dialog_norm',
        'record_correctness_recall_average': 'record_correctness'
    }
    
    # 转换键名
    converted_data = {}
    for model_name, scores in data.items():
        converted_data[model_name] = {
            key_mapping.get(old_key, old_key): value 
            for old_key, value in scores.items()
        }
    
    return converted_data

def read_model_scores_turn_8(base_path: str, num_patient: int) -> Dict[str, Dict[str, float]]:
    """读取turn=8时的模型分数"""
    results_path = base_path
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"找不到目录: {results_path}")
    
    model_scores = {}
    for model_name in os.listdir(results_path):
        model_path = os.path.join(results_path, model_name)
        if not os.path.isdir(model_path):
            continue
            
        scores_file = os.path.join(model_path, f'{model_name}_results_scored.json')
        if not os.path.exists(scores_file):
            continue
            
        with open(scores_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 只取前50个00A特征的患者
        filtered_data = []
        for item in data:
            if item.get('evaluate_scores').get("character_label") == '00A':
                filtered_data.append(item)
                if len(filtered_data) >= num_patient:
                    break
                    
        if filtered_data:
            model_scores[model_name] = extract_scores_from_json(filtered_data)
    
    return model_scores

def plot_metrics_comparison(scores_3: Dict[str, Dict[str, float]], 
                          scores_8: Dict[str, Dict[str, float]], 
                          scores_15: Dict[str, Dict[str, float]]):
    """Plot comparison of metrics across different dialog turns (vertical layout)"""
    # Define metric name mapping and order
    metric_names = {
        'record_correctness': 'Record Info.',
        'record_norm': 'Record Norm.',
        'dialog_norm': 'Dialog Norm.'
    }
    metrics = list(metric_names.keys())
    turns = [3, 8, 15]
    
    # Define color palette
    colors = ['#e0aaff', '#9d4edd', '#5a189a'] 
    
    # Prepare data, filter out huatuo_agent
    all_models = set(scores_3.keys()) | set(scores_8.keys()) | set(scores_15.keys())
    models = sorted([model for model in all_models if model != 'huatuo_agent'])
    model_names = {
    "gpt-4o-2024-11-20": "GPT-4o",
    "deepseek-v3": "DS-V3",
    "deepseek-r1": "DS-R1",
    "Qwen2.5-72B-Instruct": "Qwen72B",
    "qwen2.5-7b-instruct": "Qwen7B"
    }
    x = np.arange(len(models))
    width = 0.3
    
    # 3行1列纵向排布，整体更矮
    fig, axes = plt.subplots(3, 1, figsize=(18, 8))
    
    # Set font to Times New Roman
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    
    # y轴范围设置
    y_ranges = {
        'record_correctness': (30, 90),
        'record_norm': (60, 110),
        'dialog_norm': (60, 110)
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for i, (turn, scores) in enumerate([(3, scores_3), (8, scores_8), (15, scores_15)]):
            values = [scores.get(model, {}).get(metric, 0) for model in models]
            bars = ax.bar(x + i*width - width, values, width, 
                         label=f'Turn {turn}',
                         color=colors[i],
                         alpha=0.8)
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}',
                       ha='center', va='bottom', 
                       fontsize=22,
                       color='black',
                       fontweight='bold')
        # 显示每个模型的Turn 3到Turn 15的得分变化百分比
        for model_idx, model in enumerate(models):
            v3 = scores_3.get(model, {}).get(metric, 0)
            v15 = scores_15.get(model, {}).get(metric, 0)
            v8 = scores_8.get(model, {}).get(metric, 0)
            max_score = max([v3, v8, v15])
            min_score = min([v3, v8, v15])
            percent = (max_score - min_score) / min_score * 100
            percent_str = f"{percent:.1f}%"
            y_max = max_score
            ax.text(x[model_idx], 0.8*y_max, f'Δ={percent_str}', ha='center', va='center', fontsize=18, color='#3c096c', fontweight='bold', bbox=dict(facecolor='#f7ede2', alpha=0.8, edgecolor='none', pad=2))
        ax.set_ylabel('Score', fontsize=20)
        ax.set_title(f'{metric_names[metric]}', fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([model_names[model] for model in models], fontsize=22)
        ax.set_ylim(y_ranges[metric])
        ax.grid(True, linestyle='--', alpha=0.3)
        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(fontsize=22, framealpha=0.9, loc='upper left', bbox_to_anchor=(1, 1))
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Make left and bottom spines thicker
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(axis='y', labelsize=22)
        # 只在最后一个子图显示x轴标签
        if idx != len(metrics) - 1:
            ax.set_xticklabels([])
    # 增加子图之间的间距
    plt.subplots_adjust(hspace=0.3)
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    # 保存PNG和PDF版本
    plt.savefig('ablation study/results/dialog_turn/dialog_turn_comparison.png', 
                dpi=500, bbox_inches='tight',
                facecolor='white',
                pad_inches=0.2)
    plt.savefig('figures/dialog_turn_comparison.pdf', 
                bbox_inches='tight',
                facecolor='white',
                pad_inches=0.2)
    plt.close()

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 读取不同轮次的数据
    ablation_base_path = os.path.join(base_path, 'ablation study/results/dialog_turn')
    scores_3 = read_model_scores_turn_8(f"{ablation_base_path}/3", 50)
    scores_15 = read_model_scores_turn_8(f"{ablation_base_path}/15", 50)
    scores_8 = read_model_scores_turn_8(f"{base_path}/results", 50) # 之后要改成80

    
    # 绘制对比图
    plot_metrics_comparison(scores_3, scores_8, scores_15)

if __name__ == "__main__":
    main()

