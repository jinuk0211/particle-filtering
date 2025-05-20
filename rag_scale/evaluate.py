
import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict




def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        extracted_text = output
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
        
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)

        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        # final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer

def run_evaluation(df, input_list, output_list,start_index=0, dataset_name='gpqa', output_dir='/content/output', split=1, apply_backoff=False):


    # Existing evaluation for other datasets
    avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
    num_valid_answer = 0

    # If the dataset is GPQA, track metrics per domain
    domain_metrics = {}

    for idx, row in df.iterrows():
        
        
        
        df.at[idx, 'Output'] = row['Output']    

        if dataset_name in ['gpqa', 'medmcqa']:
            labeled_answer = df.at[idx,"Correct Answer"]
            # labeled_choice_answer = item["Correct Answer"]
            mode = 'choose'
        elif dataset_name in ['math500', 'aime', 'amc']:
            labeled_answer = df.at["answer"]
            mode = 'gen'
        elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            labeled_answer = df.at["answer"]
            mode = 'qa'
        elif dataset_name in ['pubhealth']:
            labeled_answer = item["answer"]
            mode = 'choose'
        elif dataset_name in ['popqa']:
            labeled_answer = df.at[idx,"possible_answers"]
            mode = 'qa'
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        metric, pred_answer = evaluate_predictions(output=df.at[idx,'Output'], labeled_answer=labeled_answer, mode=mode)
        # metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)
        # item['Pred_Answer'] = pred_answer
        # item['Metrics'] = metric
        # item['Question'] = input_prompt

        my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))
  
        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])
        # print(metric)
        if my_method_valid:
            num_valid_answer += 1

        # If the dataset is GPQA, attempt to track metrics per domain
        if dataset_name == 'gpqa':
            # domain = item.get("High-level domain", "Unknown")
            domain = 'Unknown'
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
            domain_metrics[domain]['total_num'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if my_method_valid:
                domain_metrics[domain]['num_valid_answer'] += 1

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

    # Compute overall metrics
    overall_results = {
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        # 'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
    }
    print(overall_results)
    # If the dataset is GPQA, output average metrics per domain
    domain_avg_metrics = {}
    if dataset_name == 'gpqa':
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
            }

    final_metrics = {'overall': overall_results}
    if dataset_name == 'gpqa':
        final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')
    df = df.to_dict(orient='list')
    # json_str = json.dumps(df)
    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(df, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=
        False)        