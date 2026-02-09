'''
Three different judges: deepseek-v3, gpt-4o, deepseek-r1.
See their scores distribution differences. 
On the first 50 cases with patient trait 00A.
'''

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import collections
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

model_names_map = {
    "gpt-4o-2024-11-20": "GPT-4o",
    "deepseek-v3": "DS-V3",
    "Qwen2.5-72B-Instruct": "Qwen72B",
    "qwen2.5-7b-instruct": "Qwen7B"
}

def load_judge_data():
    """加载所有judge的数据"""
    base_dir = Path('ablation study/results/different_judge')
    judge_data = {}
    
    # 遍历所有judge文件夹
    for judge_dir in base_dir.glob('*'):
        if not judge_dir.is_dir():
            continue
            
        judge_name = judge_dir.name
        scores_file = judge_dir / 'models_scores.json'
        stats_file = judge_dir / 'model_stats.json'
        
        if not (scores_file.exists() and stats_file.exists()):
            continue
            
        # 读取数据
        with open(scores_file, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
            
        judge_data[judge_name] = {
            'scores': scores_data,
            'stats': stats_data
        }
    
    return judge_data

def calculate_casewise_correlations(judge_data):
    """
    计算judge之间在每个模型、每个metric的case-level分数上的相关性
    judge_data[judge]['stats'][model][metric] 是一个长度为50的分数列表
    """
    metrics = ['record_norm_score', 'dialog_norm_score', 'record_correctness_recall_score']
    judges = list(judge_data.keys())
    models = list(next(iter(judge_data.values()))['stats'].keys())
    correlations = {
        'pearson': {metric: {} for metric in metrics},
        'spearman': {metric: {} for metric in metrics}
    }

    for metric in metrics:
        for i, judge1 in enumerate(judges):
            for judge2 in judges[i+1:]:
                pearson_list = []
                spearman_list = []
                for model in models:
                    scores1 = judge_data[judge1]['stats'][model][metric]
                    scores2 = judge_data[judge2]['stats'][model][metric]
                    # 只保留都不是None的分数
                    valid_scores = [(s1, s2) for s1, s2 in zip(scores1, scores2) if s1 is not None and s2 is not None]
                    if len(valid_scores) > 1:
                        s1_list, s2_list = zip(*valid_scores)
                        pearson_corr, _ = pearsonr(s1_list, s2_list)
                        spearman_corr, _ = spearmanr(s1_list, s2_list)
                        pearson_list.append(pearson_corr)
                        spearman_list.append(spearman_corr)
                # 取所有模型的相关性均值
                correlations['pearson'][metric][f"{judge1}--{judge2}"] = np.mean(pearson_list) if pearson_list else None
                correlations['spearman'][metric][f"{judge1}--{judge2}"] = np.mean(spearman_list) if spearman_list else None

    return correlations

def plot_correlation_heatmaps(correlations):
    """绘制相关性热力图"""
    metrics = ['record_norm_average', 'dialog_norm_average', 'record_correctness_recall_average']
    metric_names = ['Record Norm.', 'Dialog Norm.', 'Record Info.']

    for corr_type in ['pearson', 'spearman']:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        judges = None
        short_names = None
        vmin, vmax = -1, 1  # 统一色阶

        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            # 准备数据
            corr_data = correlations[corr_type][metric]
            judge_pairs = list(corr_data.keys())
            corr_values = list(corr_data.values())

            # 获取所有唯一的judge名称
            if judges is None:
                judges = sorted(list(set([j for pair in judge_pairs for j in pair.split('--')])))
                short_names = {name: f'J{i+1}' for i, name in enumerate(judges)}

            n_judges = len(judges)
            # 创建热力图数据
            heatmap_data = np.zeros((n_judges, n_judges))
            for pair, value in zip(judge_pairs, corr_values):
                j1, j2 = pair.split('--')
                i1, i2 = judges.index(j1), judges.index(j2)
                heatmap_data[i1, i2] = value
                heatmap_data[i2, i1] = value

            # 绘制热力图（不带colorbar）
            sns.heatmap(
                heatmap_data, annot=True, cmap='coolwarm', vmin=vmin, vmax=vmax,
                xticklabels=[short_names[j] for j in judges],
                yticklabels=[short_names[j] for j in judges],
                cbar=False, ax=ax
            )
            ax.set_title(f'{metric_name}')

        # 添加全局colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Correlation')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(f'ablation study/results/different_judge/{corr_type}_correlation_heatmap.png')
        plt.savefig(f'figures/{corr_type}_correlation_heatmap.pdf', dpi=500)
        plt.close()

def prepare_boxplot_data(judge_data):
    """准备箱线图数据"""
    metrics = {
        'record_norm_score': 'Record Norm.',
        'dialog_norm_score': 'Dialog Norm.',
        'record_correctness_recall_score': 'Record Info.'
    }
    
    # 创建DataFrame存储数据
    data = []
    judges = list(judge_data.keys())
    
    for metric, metric_name in metrics.items():
        for model in judge_data[judges[0]]['stats'].keys():
            # 收集所有judge对该模型的评分
            model_scores = []
            for judge in judges:
                scores = judge_data[judge]['stats'][model][metric]
                # 处理None值
                valid_scores = [s for s in scores if s is not None]
                if len(valid_scores) > 0:
                    avg_score = np.mean(valid_scores)
                    scores = [s if s is not None else avg_score for s in scores]
                model_scores.extend(scores)
            
            # 添加到数据框
            for judge, score in zip(judges * len(scores), model_scores):
                data.append({
                    'Judge': judge,
                    'Metric': metric_name,
                    'Score': score,
                    'Model': model
                })
    
    return pd.DataFrame(data)

def plot_boxplots(df):
    """绘制箱线图"""
    # 过滤掉分数为0的数据
    df = df[df['Score'] != 0]
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 20})  # 设置全局字号

    metrics = df['Metric'].unique()
    axes = []
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 1, i+1)
        axes.append(ax)
        sns.boxplot(
            data=df[df['Metric'] == metric],
            x='Model', y='Score', hue='Judge', ax=ax
        )
        ax.set_title(f'{metric} Scores Distribution', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.tick_params(axis='x', labelrotation=15, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # 只在第二个子图显示legend，并放在图外
        if i == 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title='Judge', bbox_to_anchor=(1, 0.6), loc='upper left', fontsize=14, title_fontsize=14, framealpha=0.5)
        else:
            ax.get_legend().remove()
        # 只在最后一个子图显示x轴
        if i != 2:
            ax.set_xlabel('')
            ax.set_xticklabels([])

    plt.tight_layout(rect=[0, 0, 0.93, 1])  # 给legend留空间
    plt.savefig('ablation study/results/different_judge/score_distribution_boxplots.png')
    plt.savefig('figures/score_distribution_boxplots.pdf', dpi=500)
    plt.close()

def read_00A_scores(results_dir='results', judge_model='gpt-4o-2024-11-20', num_samples=50):
    metrics = ['record_norm', 'dialog_norm', 'record_correctness']
    score_keys = [f'{m}_score' if m != 'record_correctness' else 'record_correctness_recall_score' for m in metrics]
    model_scores = {}

    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        scores_file = os.path.join(model_path, f'{model_name}_results_scored.json')
        if not os.path.exists(scores_file):
            continue

        with open(scores_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 初始化每个模型的分数列表
        model_scores[model_name] = {key: [] for key in score_keys}
        count_00A = 0

        for item in data:
            char_label = item.get('evaluate_scores', {}).get('character_label')
            if char_label != '00A':
                # 非00A，跳过
                continue

            # 只取前num_samples个00A
            if count_00A >= num_samples:
                continue

            score_dict = item.get('score', {})
            for metric in metrics:
                if metric not in score_dict:
                    model_scores[model_name][f'{metric}_score'].append(None)
                    continue
                if item.get('dialog') == 'None' or item.get('dialog') is None:
                    if metric == 'record_correctness':
                        model_scores[model_name]['record_correctness_recall_score'].append(None)
                    else:
                        model_scores[model_name][f'{metric}_score'].append(None)
                    continue
                prompt = score_dict[metric]
                if metric == 'record_correctness':
                    value = prompt.get('total_recall')
                else:
                    value = prompt.get('total_score')
                # 转为float，异常为None
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
                if metric == 'record_correctness':
                    model_scores[model_name]['record_correctness_recall_score'].append(value)
                else:
                    model_scores[model_name][f'{metric}_score'].append(value)
            count_00A += 1

        # # 保证每个metric只有num_samples个分数（多余的截断，不足补None）
        # for metric in metrics:
        #     scores = model_scores[model_name][f'{metric}_score']
        #     if len(scores) > num_samples:
        #         model_scores[model_name][f'{metric}_score'] = scores[:num_samples]
        #     elif len(scores) < num_samples:
        #         model_scores[model_name][f'{metric}_score'] += [None] * (num_samples - len(scores))
    with open(f'ablation study/results/different_judge/{judge_model}/model_stats.json', 'w', encoding='utf-8') as f:
        f.truncate(0)
        json.dump(model_scores, f, ensure_ascii=False, indent=4)
    # 计算平均分并存入
    model_stats = {}
    for model_name, metric_scores in model_scores.items():
        model_stats[model_name] = {}
        for metric, scores in metric_scores.items():
            valid_scores = [s for s in scores if s is not None]
            avg = float(np.mean(valid_scores)) if valid_scores else None
            metric_name = metric.replace('_score', '_average')
            model_stats[model_name][metric_name] = avg

    # 保存所有分数和平均分到 model_stats.json
    output_path = f'ablation study/results/different_judge/{judge_model}/models_scores.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.truncate(0)
        json.dump(model_stats, f, ensure_ascii=False, indent=4)

    # return model_scores, model_stats

def plot_judge_scatter_and_spearman(judge_data, save_prefix):
    """
    只画record_correctness的散点图，将三对judge的比较都画在同一个子图上
    """
    metric = 'record_correctness_recall_score'
    metric_name = 'Record Info.'
    
    # 定义三对judge比较
    judge_pairs = [
        ('deepseek-v3', 'gpt-4o-2024-11-20', '#1f77b4'),  # 蓝色
        ('deepseek-v3', 'Qwen2.5-72B-Instruct', '#2ca02c'),  # 绿色
        ('gpt-4o-2024-11-20', 'Qwen2.5-72B-Instruct', '#ff7f0e')  # 橙色
    ]
    
    models = list(judge_data['deepseek-v3']['stats'].keys())
    exclude_models = ['huatuo_agent']
    models = [model for model in models if model not in exclude_models]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    p_values = {}  # 存储p值的字典
    
    
    for judge1, judge2, color in judge_pairs:
        x_vals, y_vals = [], []
        for model in models:
            v1 = judge_data[judge1]['stats'][model][metric]
            v2 = judge_data[judge2]['stats'][model][metric]
            for i in range(len(v1)):
                if v1[i] is not None and v2[i] is not None:
                    x_vals.append(v1[i])
                    y_vals.append(v2[i])
        
        # 计算pearson相关系数
        if len(x_vals) > 1:
            pearson_corr, p_value = pearsonr(x_vals, y_vals)
            key = f"{judge1}__{judge2}__{metric}"
            p_values[key] = p_value
            label = f"J1: {model_names_map.get(judge1, judge1)}\nJ2: {model_names_map.get(judge2, judge2)}\nPearson r = {pearson_corr:.2f}"
        else:
            label = f"J1: {model_names_map.get(judge1, judge1)}\nJ2: {model_names_map.get(judge2, judge2)}"
        
        # 画散点
        ax.scatter(x_vals, y_vals, label=label, alpha=0.6, s=70, c=color, edgecolor='k', linewidth=0.5)
    
    # 画y=x参考线
    min_val = 10
    max_val = 110
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6)
    
    # 设置坐标轴范围为20到100
    ax.set_xlim([10, 110])
    ax.set_ylim([10, 110])
    
    # 设置图表属性
    ax.set_xlabel('J1 Score', fontsize=22, fontweight='bold')
    ax.set_ylabel('J2 Score', fontsize=22, fontweight='bold')
    ax.set_title(metric_name, fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.03, 0.9), loc='upper left', fontsize=16, prop = {'size': 18, 'weight': 'bold'})
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f'ablation study/results/different_judge/{save_prefix}_record_correctness_scatter.png', bbox_inches='tight')
    plt.savefig(f'figures/{save_prefix}_record_correctness_scatter.pdf', bbox_inches='tight', dpi=500)
    plt.close()
    
    # 保存p值
    with open(f'ablation study/results/different_judge/{save_prefix}_p_values.json', 'w', encoding='utf-8') as f:
        json.dump(p_values, f, ensure_ascii=False, indent=4)

def plot_judge_bubble(judge_data, save_prefix):
    """
    为每对judge创建单独的bubble plot，每个图表包含dialog norm和record norm两个子图
    """
    metrics = [
        ('record_norm_score', 'Record Norm.'),
        ('dialog_norm_score', 'Dialog Norm.')
    ]
    
    # 定义三对judge比较及其颜色
    judge_pairs = [
        ('deepseek-v3', 'gpt-4o-2024-11-20', '#1f77b4'),  # 蓝色
        ('deepseek-v3', 'Qwen2.5-72B-Instruct', '#2ca02c'),  # 绿色
        ('gpt-4o-2024-11-20', 'Qwen2.5-72B-Instruct', '#ff7f0e')  # 橙色
    ]
    
    models = list(judge_data['deepseek-v3']['stats'].keys())
    exclude_models = ['huatuo_agent']
    models = [model for model in models if model not in exclude_models]

    # 为每对judge创建单独的图表
    for judge1, judge2, color in judge_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        for idx, (metric, metric_name) in enumerate(metrics):
            ax = axes[idx]
            all_data = []  # 存储所有数据点
            
            x_vals, y_vals = [], []
            for model in models:
                v1 = judge_data[judge1]['stats'][model][metric]
                v2 = judge_data[judge2]['stats'][model][metric]
                for i in range(len(v1)):
                    if v1[i] is not None and v2[i] is not None:
                        x_vals.append(v1[i])
                        y_vals.append(v2[i])
                        all_data.append({
                            'j1': v1[i],
                            'j2': v2[i]
                        })
            
            # 转换为DataFrame并计算每个点的出现次数
            df = pd.DataFrame(all_data)
            counts = df.groupby(['j1', 'j2']).size().reset_index(name='count')
            
            # 计算pearson相关系数
            pearson_corr, p_value = pearsonr(x_vals, y_vals)
            
            # 画气泡图
            scatter = ax.scatter(
                counts['j1'], 
                counts['j2'], 
                s=counts['count'] * 20,  # 气泡大小与count成正比
                alpha=0.6,
                c=color,  # 使用对应的颜色
                edgecolor='k',
                linewidth=0.5
            )
            
            # 添加相关系数文本
            ax.text(0.05, 0.95, 
                    f"Pearson r = {pearson_corr:.2f}\np = {p_value:.2e}", 
                    transform=ax.transAxes,
                    fontsize=20, 
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # 画y=x参考线
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # 设置坐标轴范围为20到100
            ax.set_xlim([20, 100])
            ax.set_ylim([20, 100])
            
            # 设置图表属性
            ax.set_xlabel(f'{model_names_map.get(judge1, judge1)} Score', fontsize=22, fontweight='bold')
            if idx == 0:
                ax.set_ylabel(f'{model_names_map.get(judge2, judge2)} Score', fontsize=22, fontweight='bold')
            else:
                ax.set_ylabel('')
            ax.tick_params(labelsize=18)
            
            # 添加子图标题
            ax.set_title(metric_name, fontsize=24, fontweight='bold', pad=20)
            
            sizes = [1, 2, 3, 4, 5]
            size_labels = [f"{size}" for size in sizes]
            size_elements = [plt.scatter([], [], s=size*20, c=color, alpha=0.6, edgecolor='k', linewidth=0.5, label=label) 
                           for size, label in zip(sizes, size_labels)]
            
            # 合并两个图例
            legend_elements = size_elements
            legend_labels = size_labels
            if idx == 1:
                ax.legend(handles=legend_elements, 
                         labels=legend_labels,
                         bbox_to_anchor=(1.05, 1), 
                         loc='upper left', 
                         prop = {'size': 18, 'weight': 'bold'})
        
        plt.tight_layout()
        # 使用judge对的名字作为文件名
        judge_pair_name = f"{judge1}_vs_{judge2}".replace('-', '_')
        plt.savefig(f'ablation study/results/different_judge/{judge_pair_name}_normativeness_bubble.png', bbox_inches='tight')
        plt.savefig(f'figures/{judge_pair_name}_normativeness_bubble.pdf', bbox_inches='tight', dpi=500)
        plt.close()

def calculate_judge_average_scores(judge_data):
    """
    计算每个judge在每个模型上的三个指标的平均分
    返回的数据结构：
    {
        "judge_name": {
            "model_name": {
                "record_norm_average": float,
                "dialog_norm_average": float,
                "record_correctness_recall_average": float
            }
        }
    }
    """
    metrics = [
        'record_norm_average',
        'dialog_norm_average',
        'record_correctness_recall_average'
    ]
    
    judge_average_scores = {}
    
    # 遍历每个judge
    for judge in judge_data:
        judge_average_scores[judge] = {}
        models = list(judge_data[judge]['stats'].keys())
        exclude_models = ['huatuo_agent']
        models = [model for model in models if model not in exclude_models]
        
        # 遍历每个模型
        for model in models:
            judge_average_scores[judge][model] = {}
            
            # 遍历每个指标
            for metric in metrics:
                avg_score = judge_data[judge]['scores'][model][metric]
                judge_average_scores[judge][model][metric] = avg_score
    
    # 保存为JSON文件
    output_path = 'ablation study/results/different_judge/judge_average_scores.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(judge_average_scores, f, ensure_ascii=False, indent=4)
    
    print(f"平均分数据已保存到: {output_path}")
    return judge_average_scores

def main():
    judge_data = load_judge_data()
    
    # 画record_correctness的散点图（1×3布局）
    plot_judge_scatter_and_spearman(judge_data, 'all_judges')
    
    # 画normativeness的bubble plot（暂时注释掉）
    # plot_judge_bubble(judge_data, 'all_judges')
    
    # 计算并保存平均分（暂时注释掉）
    # judge_average_scores = calculate_judge_average_scores(judge_data)
    
    print("分析完成，图表已保存到 ablation study/results/different_judge/ 目录下")

if __name__ == "__main__":
    main()
