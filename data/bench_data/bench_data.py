import os
import json
import re
import matplotlib.pyplot as plt

def count_info_num(cases):
    """统计每个病例的主诉、现病史和既往史的点数和字数

    Args:
        cases (list): 病例数据列表

    Returns:
        list: 添加了统计信息的病例列表
    """
    counted_cases = []
    for idx, case in enumerate(cases):
        try:
            processed_chief = case['处理后主诉']['process_info']
            processed_history = case['处理后现病史']['process_info']
            processed_past = case['处理后既往史']['process_info']
            
            chief_num = count_list_points(processed_chief)
            history_num = count_list_points(processed_history)
            past_num = count_list_points(processed_past)
            
            chief_word_num = word_count(processed_chief)
            history_word_num = word_count(processed_history)
            past_word_num = word_count(processed_past)
            
            case['主诉点数'] = chief_num
            case['现病史点数'] = history_num
            case['既往史点数'] = past_num
            case['主诉字数'] = chief_word_num
            case['现病史字数'] = history_word_num
            case['既往史字数'] = past_word_num
            counted_cases.append(case)
        except Exception as e:
            print(f"Error processing case {idx}: {e}")
            continue
    return counted_cases

def count_list_points(text):
    """统计文本中的列表点数
    
    Args:
        text (str): 包含列表的文本，每点以数字和点开头，用换行符分隔
        
    Returns:
        int: 列表中的总点数
    """
    if not text:
        return 0
        
    # 按换行符分割文本
    lines = text.split('\n')
    
    # 统计以数字和点开头的行数
    count = 0
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            count += 1
            
    return count

def word_count(text):
    """统计文本中除标点符号以外的字符数"""
    # 使用正则表达式移除所有标点符号
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    # 统计字符数
    return len(text_without_punctuation)

def plot_info_num(counted_cases):
    """绘制主诉、现病史和既往史的点数分布图"""
    # 提取数据
    chief_counts = [int(case['主诉点数']) for case in counted_cases]
    history_counts = [int(case['现病史点数']) for case in counted_cases]
    past_counts = [int(case['既往史点数']) for case in counted_cases]
    
    # 创建图形
    plt.figure(figsize=(15, 5))
    
    # 主诉点数分布
    plt.subplot(1, 3, 1)
    plt.hist(chief_counts, bins=range(0, max(chief_counts)+2), alpha=0.7)
    plt.title('Chief Complaint Points Distribution')
    plt.xlabel('Points')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 现病史点数分布
    plt.subplot(1, 3, 2)
    plt.hist(history_counts, bins=range(0, max(history_counts)+2), alpha=0.7)
    plt.title('History of Present Illness Points Distribution')
    plt.xlabel('Points')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 既往史点数分布
    plt.subplot(1, 3, 3)
    plt.hist(past_counts, bins=range(0, max(past_counts)+2), alpha=0.7)
    plt.title('Past Medical History Points Distribution')
    plt.xlabel('Points')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为PDF
    plt.savefig('cases_info_points_distribution.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def assign_record_difficulty_level(counted_cases):
    """根据info_num 将病例分为easy和hard, 添加record_difficulty_level字段, 并统计数量"""
    easy_count_points = 0
    easy_count_words = 0
    hard_count_points = 0
    hard_count_words = 0
    
    for idx, case in enumerate(counted_cases):
        total_points = case['主诉点数'] + case['现病史点数'] + case['既往史点数']
        total_words = case['主诉字数'] + case['现病史字数'] + case['既往史字数']
        
        if total_points <= 6:
            case['record_difficulty_level'] = 'easy'
            easy_count_points += 1
        else:
            case['record_difficulty_level'] = 'hard'
            hard_count_points += 1

        if total_words <= 120:
            easy_count_words += 1
            case['word_count_level'] = 'easy'
        else:
            hard_count_words += 1
            case['word_count_level'] = 'hard'
        case['idx'] = idx
        
    # 打印统计信息
    print("\n病例难度分布统计：")
    print(f"Easy cases by points: {easy_count_points} ({easy_count_points/len(counted_cases)*100:.2f}%)")
    print(f"Hard cases by points: {hard_count_points} ({hard_count_points/len(counted_cases)*100:.2f}%)")
    print(f"Easy cases by words: {easy_count_words} ({easy_count_words/len(counted_cases)*100:.2f}%)")
    print(f"Hard cases by words: {hard_count_words} ({hard_count_words/len(counted_cases)*100:.2f}%)")
    print(f"Total cases: {len(counted_cases)}")
    
    return counted_cases

if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(project_root, "data/bench_data/process_case.json")
    
    # 读取数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        print(f"成功加载 {file_path}, 包含 {len(cases)} 条病例")
    except Exception as e:
        print(f"读取文件时出错: {file_path}: {str(e)}")
        exit(1)
    
    # 统计信息
    counted_cases = count_info_num(cases)
    
    # 输出计数后的数据到新文件
    output_path = os.path.join(project_root, "data/bench_data/cases_counted.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(counted_cases, f, ensure_ascii=False, indent=2)
    print(f"Successfully counted {len(counted_cases)} cases to {output_path}")

    # 绘制点数分布图
    plot_info_num(counted_cases)

    # 分配难度等级
    counted_cases = assign_record_difficulty_level(counted_cases)
    
    # 输出带难度等级的数据到新文件
    output_path = os.path.join(project_root, "data/bench_data/cases_counted_difficulty.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(counted_cases, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(counted_cases)} cases with difficulty levels to {output_path}")
