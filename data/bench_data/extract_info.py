'''
Extract and reformat information items from the Real medical records.
Input: 
    - medical_record path: ./data_sum_分科室.xlsx
        - data source: iyy, 龙岗中心医院. 
    - extract prompts & difficulty level division.
Output: 
    - The extracted and re-format medical record: json.
        - extracted chief complaint
        - extracted present illness
        - extracted past illness
        - case difficulty level
        - Items to be checked: 主要做了哪些改动. 
'''

chief_complaint_classfy_prompt = '''
你是一个医疗领域的专家，现在需要你判断处理并重新排版主诉信息. 
# 判断处理:
如果主诉包含以下信息, 则主诉需要处理, 你将:
1. 删除掉具体的检查数字、药物剂量等细节信息;
2. 删除掉病人的基本信息(性别、年龄、科室等);
3. 仅保留主要症状(或疾病名称)以及时间信息. 
4. 如果主诉中没有标点符号分隔, 就不需要分成2点, 直接返回1点即可. 

# 处理要求:
对处理后的主诉, 你将:
1. 将主诉信息根据症状进行语义上的分点, 一般症状+时间为1个点;
2. 以分点列表的形式返回处理后的主诉. 
3. 提取的主诉点个数应该在2个以内, 可以比2个少, 但是不可以超过2个. 如果主诉点个数超过2个, 则需要根据重要性, 选择最重要的2个主诉点.

特别注意, 处理后的主诉中的信息应完全来自待处理的主诉, 禁止出现待处理的主诉中没有的信息. 

# 返回格式如下:
[判断处理]: 
处理了原来主诉中哪些信息, 以及处理原因. 

[处理后]:
1. 
...

'''

present_illness_classfy_prompt = '''
你是一个医疗领域的专家, 现在需要你判断并处理病历单中的现病史信息.

# 判断处理:
如果现病史中包含以下信息, 则现病史需要处理, 你将:
1. 总体按照句号对现病史进行分点;
2. 删除掉具体的检查数字、具体的检查项目指标的结果、药物剂量等细节信息, 仅保留检查大致名称和药物名称, 表示病人做了某项检查(CT、X光、B超...)或服用了某类药物;
3. 判断检查的时间: 如果是病人入院后做的检查, 需要删除掉; 如果是病人入院前、以前做的检查, 则需要保留;
4. 处理阴性信息: 阴性信息包括病人没有的症状、无特殊的一般情况(如睡眠、饮食、大小便、精神状态等)等等. 将所有的阴性症状合并到一个点中, 并挑选2-3个最重要、最有利于鉴别诊断的阴性症状;
5. 删除掉就诊信息, 例如"病人于...科室...医院就诊"等等类似信息.  
6. 阳性信息: 尽可能做保留. 

# 处理要求:
对处理后的现病史, 你将:
1. **每一点的信息**应有一定**信息量**, 原则上不少于15字, 不超过30字;
2. 分点后的内容**不应重复**;
3. 分点总数**不超过5个**, 可少于5个;
4. 禁止引入现病史中不存在的信息, 仅做语义上的整理与筛选. 

# 返回格式：
[判断处理]:  
对原来的现病史做了哪些处理, 以及处理原因. 

[处理后]:  
1.  
2.  
3.  
...

'''

past_illness_classfy_prompt = '''
你是一个医疗领域的专家，现在需要你判断并处理既往史信息。

# 判断处理:
如果既往史中包含以下信息, 则既往史需要处理, 你将:
1. 检查&药物: 删除掉具体的检查数字(常见的血压、血糖数字除外)、药物剂量等细节信息, 仅保留检查名称或药物名称, 表示病人做了检查;
2. 阴性信息: 若没有的项目中有具体的例子, 需要保留. 例如"否定糖尿病、高血压等慢性病", 则需要保留. 例如"否认特殊手术史", 则不需要保留; 
3. 阳性信息: 若既往史中没有阳性信息, 全是没有例子的阴性信息, 则需要改写成'无特殊既往史'; 若明确提及病人过去身体健康, 则可以概括为"既往体健"; 
4. 保留阳性信息(项目+时间信息+治疗方案), 也需要保留有例子的阴性信息. 

# 处理要求:
对处理后的既往史, 你将:
1. 将处理后的既往史中信息按照语义信息进行分点, 并以分点列表的形式返回处理后的既往史;
2. 分点后, 一点中的内容应在10字以内, 且点与点之间内容不重复;
3. 既往史的信息点个数应该在3个以内. 可以比3个少, 但是不可以超过3个. 如果既往史的信息点个数超过3个, 则需要根据重要性, 选择最重要的既往史信息点进行提取;
4. 若既往史中没有阳性信息, 全是没有例子的阴性信息, 则需要改写成'无特殊既往史'; 若明确提及病人过去身体健康, 则可以概括为"既往体健". 

特别注意:
1. 处理后的既往史中的信息应完全来自待处理的既往史, 禁止出现待处理的既往史中没有的信息; 
2. 只有在既往史是全阴性信息时, 才需要改写成'无特殊既往史'或"既往体健"; 若既往史中存在阳性信息, 则不能改写;
3. 提取后的既往史中的信息点个数应该在3个以内, 可以比3个少, 但是不可以超过3个.

# 返回格式如下:
[判断处理]: 
对原来的既往史做了哪些处理, 以及处理原因. 

[处理后]:
1. 
2. 
3.
... 

''' 

import pandas as pd
import json
from openai import OpenAI
from dataclasses import dataclass
from typing import Union, List, Dict
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from tqdm import tqdm
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 读取Excel文件
def load_excel(path: str) -> pd.DataFrame:
    """Load data from Excel file.

    Args:
        path: Path to Excel file

    Returns:
        DataFrame containing all data from the Excel file
    """
    try:
        return pd.read_excel(path)
    except Exception as e:
        print(f"Error loading Excel file {path}: {e}")
        raise 

def llm_chat(model_name: str, messages: list) -> str:
    '''
    '''
    client = OpenAI(
        api_key = 'api_key',
        base_url='http://47.238.136.89:3010/v1'
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1
    )
    reply = response.choices[0].message.content
    if '</think>' in reply:
        reply = reply.split('</think>')[1].strip()
    return reply


def llm_response_string_matching(reply: str) -> Dict[str, str]:
    str_1 = '[判断处理]:'
    str_2 = '[处理后]:'
    try:
        process_reason = reply.split(str_1)[1].split(str_2)[0].strip()
        process_info = reply.split(str_2)[1].strip()
        return {
            'process_info': process_info,
            'process_reason': process_reason
        }
    except Exception as e:
        print(f'Error processing reply: {reply}')
        return {
            'process_info': '',
            'process_reason': ''
        }



def extract_chief_copmplaint(real_chief: str, model_name: str):
    '''
    Extract the chief complaint from the real medical record.
    '''
    user_str = f'''
# 待处理的主诉如下:
反复上腹痛3个月, 加重伴黑便1周. 

[判断处理]:
主诉中没有多余的信息, 也没有需要删除、过滤掉的信息. 因此仅做分点处理.

[处理后]:
1. 反复上腹痛3个月. 
2. 加重伴黑便1周.

# 待处理的主诉如下:
{real_chief}

'''
    messages = [
        {'role': 'system', 'content': chief_complaint_classfy_prompt},
        {'role': 'user', 'content': user_str}
    ]
    try:
        reply = llm_chat(model_name=model_name, messages=messages)
        return llm_response_string_matching(reply)
    except Exception as e:
        print(f'Error extracting chief complaint: {e}')
        return {
            'process_info': '',
            'process_reason': ''
        }


def extract_present_illness(real_present_illness: str, model_name: str):
    '''
    Extract the present illness from the real medical record.
    '''
    user_str = f'''
# 待处理的现病史如下:
患者3个月前无诱因出现上腹部隐痛, 以餐后加重为主, 未予诊治。 1周前疼痛加剧, 转为持续性, 并出现黑便2-3次/日, 每次量约200g, 无呕血、晕厥。自服奥美拉唑(20mg qd)1周无效。病程中体重下降4kg, 伴乏力, 无发热、黄疸。外院查Hb 78g/L, 粪便潜血(+++), 幽门螺杆菌检测阳性。今为进一步诊治就诊。


[判断处理]:
1. 现病史中出现了具体的药物剂量(20mg qd), 因此需要删除掉. 
2. "外院查..."表示是入院前的检查, 因此不需要删除病人做了检查这一事实. 
3. 出现了"外院查Hb 78g/L"等具体检查数值结果, 因此需要删除掉具体的检查数值结果. 
4. 出现了"病人于...科室...医院就诊"等就诊信息, 因此需要删除掉. 

[处理后]:
1. 患者3个月前无诱因出现上腹痛, 以餐后加重为主. 
2. 1周前疼痛加剧, 转为持续性, 并出现黑便2-3次/日, 自服"奥美拉唑"1周无效.
3. 无呕血、晕厥, 病程中体重下降4kg, 伴乏力.
4. 外院检查发现粪便潜血阳性, 幽门螺杆菌检测阳性. 


# 待处理的现病史如下:
{real_present_illness}
'''
    
    messages = [
        {'role': 'system', 'content': present_illness_classfy_prompt},
        {'role': 'user', 'content': user_str}
    ]
    reply = llm_chat(model_name=model_name, messages=messages)
    return llm_response_string_matching(reply)


def extract_past_illness(real_past_illness: str, model_name: str):
    '''
    Extract the past illness from the real medical record.
    '''
    user_str = f'''
# 待处理的既往史如下:

既往体健, 无高血压、糖尿病等慢性病史, 无肝炎、结核等传染病史, 无手术史、外伤史、输血史, 无过敏史.

[判断处理]: 
1. 由于既往史中出现"既往体健", 因此需要保留. 
2. 有含有例子的阴性信息"无高血压、糖尿病等慢性病史, 无肝炎、结核等传染病史", 需要保留. 

[处理后]:
1. 既往体健
2. 无高血压、糖尿病等慢性病史
3. 无肝炎、结核等传染病史

# 待处理的既往史如下:
{real_past_illness}
'''
    
    messages = [
        {'role': 'system', 'content': past_illness_classfy_prompt},
        {'role': 'user', 'content': user_str}
    ]
    reply = llm_chat(model_name=model_name, messages=messages)
    return llm_response_string_matching(reply)
    

def process_single_row(row: pd.Series, model_name: str) -> Dict[str, str]:
    '''
    Process a single row of the medical record.
    '''
    try:
        return {
            '来源': row['来源'],
            '科室': row['第二科室'],
            '病人性别': row['性别'],
            '病人年龄': row['年龄'],
            '处理后主诉': extract_chief_copmplaint(row['主诉'], model_name=model_name),
            '处理后现病史': extract_present_illness(row['现病史'], model_name=model_name),
            '处理后既往史': extract_past_illness(row['既往史'], model_name=model_name),
            '原始主诉': row['主诉'],
            '原始现病史': row['现病史'],
            '原始既往史': row['既往史']
        }
    except Exception as e:
        print(f'Error processing row {row.name}: {e}')
        return {
            '来源': row['来源'],
            '科室': row['第二科室'],
            '病人性别': row['性别'],
            '病人年龄': row['年龄'],
            '处理后主诉': row['主诉'],
            '处理后现病史': row['现病史'],
            '处理后既往史': row['既往史'],
            '原始主诉': row['主诉'],
            '原始现病史': row['现病史'],
            '原始既往史': row['既往史']
        }
    


def save_results_json(results: List[Dict[str, str]], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def load_processed_results(path: str) -> Dict[str, Dict]:
    """加载已处理的结果文件

    Args:
        path: 结果文件路径

    Returns:
        已处理结果的字典，key为记录的唯一标识
    """
    if not os.path.exists(path):
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # 使用科室+性别+年龄作为唯一标识
        return {f"{item['原始主诉']}_{item['原始现病史']}_{item['原始既往史']}": item for item in results}
    except Exception as e:
        print(f"Error loading processed results: {e}")
        return {}

def get_record_key(row: pd.Series) -> str:
    """获取记录的唯一标识

    Args:
        row: DataFrame的一行数据

    Returns:
        记录的唯一标识
    """
    return f"{row['主诉']}_{row['现病史']}_{row['既往史']}"

def is_failed_case(case: dict) -> bool:
    """判断一个case是否处理失败（所有主要字段都为空）"""
    keys = ["来源", "科室", "病人性别", "病人年龄", "处理后主诉", "处理后现病史", "处理后既往史", "原始主诉", "原始现病史", "原始既往史"]
    return all((not case.get(k)) or (isinstance(case.get(k), dict) and not any(case.get(k).values())) for k in keys)


def retry_failed_cases(json_path: str, model_name: str, df: pd.DataFrame):
    """检测并重试处理失败的cases，替换原位置，其他内容不变"""
    # 读取原json
    with open(json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    print(f"原始case总数: {len(cases)}")
    failed_indices = [i for i, case in enumerate(cases) if is_failed_case(case)]
    print(f"检测到失败case数: {len(failed_indices)}，索引: {failed_indices}")
    if not failed_indices:
        print("没有需要重试的失败case！")
        return
    # 重新处理失败的case
    for idx in failed_indices:
        # 通过原始主诉、现病史、既往史在df中查找对应行
        # 这里假设df和cases顺序一致，直接用idx
        row = df.iloc[idx]
        try:
            new_case = process_single_row(row, model_name=model_name)
            cases[idx] = new_case
            print(f"重试成功: case idx={idx}")
        except Exception as e:
            print(f"重试失败: case idx={idx}, error: {e}")
    # 保存回原json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=4)
    print(f"重试完成，已更新{len(failed_indices)}个case。")

# 用法示例：
# retry_failed_cases('./data/bench_data/process_case.json', 'gpt-4o-2024-11-20', df)

if __name__ == '__main__':
    # test_len = 500
    # result_path = './data/bench_data/process_case.json'
    
    # # 加载已处理的结果
    # processed_results = load_processed_results(result_path)
    # print(f"已处理记录数: {len(processed_results)}")
    
    # # 加载数据
    # df = load_excel(path='./data/bench_data/bench_500.xlsx')
    # tasks_evaluate = []
    
    # # 只处理未处理的记录
    # for idx, row in df.iloc[:test_len].iterrows():
    #     record_key = get_record_key(row)
    #     if record_key not in processed_results:
    #         tasks_evaluate.append(row)
    
    # print(f"待处理记录数: {len(tasks_evaluate)}")
    
    # if tasks_evaluate:
    #     with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 16)) as executor:
    #         def worker(row):
    #             try:
    #                 return process_single_row(row, model_name='gpt-4o-2024-11-20')
    #             except Exception as e:
    #                 print(f'Error processing row {idx}: {e}')
    #                 return {
    #                     '来源': '',
    #                     '科室': '',
    #                     '病人性别': '',
    #                     '病人年龄': '',
    #                     '处理后主诉': '',
    #                     '处理后现病史': '',
    #                     '处理后既往史': '',
    #                     '原始主诉': '',
    #                     '原始现病史': '',
    #                     '原始既往史': ''
    #                 }
    #         process_result = list(tqdm(executor.map(worker, tasks_evaluate), total=len(tasks_evaluate)))
            
    #         # 合并新旧结果
    #         all_results = list(processed_results.values()) + process_result
    #         save_results_json(all_results, path=result_path)
    #         print(f"处理完成，总记录数: {len(all_results)}")
    # else:
    #     print("没有需要处理的记录")
    # 例如在主程序或单独脚本中这样调用
    df = load_excel(path='./data/bench_data/bench_500.xlsx')
    retry_failed_cases('./data/bench_data/process_case.json', 'gpt-4o-2024-11-20', df)
        






