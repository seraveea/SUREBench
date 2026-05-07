from tqdm import tqdm
import re
from collections import Counter
from datasets import load_dataset
from eval_pipelines.utils import get_files, area_under_accuracy_coverage_curve, auroc, build_qa_report
import pandas as pd
from path_config import dataset_cache_dir


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text.replace(',', '').replace('$', ''))
    except Exception:
        pass

    frac_match = re.search(r'(-?\d+)\s*/\s*(-?\d+)', text)
    if frac_match:
        num = float(frac_match.group(1))
        den = float(frac_match.group(2))
        if den != 0:
            return num / den

    num_matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if not num_matches:
        return None
    try:
        return float(num_matches[-1].replace(',', ''))
    except Exception:
        return None


def _to_float_first(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    num_matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if not num_matches:
        return None
    try:
        return float(num_matches[0].replace(',', ''))
    except Exception:
        return None


def _looks_truncated(text):
    s = str(text).rstrip()
    if not s:
        return True

    # Strong markers usually indicate the answer section is present.
    if re.search(r'(?i)(####+|\\boxed\{|final\s+answer|final\s+numerical\s+answer)', s):
        return False

    non_empty_lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    last_line = non_empty_lines[-1] if non_empty_lines else s

    if re.fullmatch(r'[$\s]*[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*', last_line):
        return False

    # Any numeric signal in the tail is usually useful, unless the tail is
    # obviously cut off in the middle of an expression/sentence.
    has_number = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', last_line) is not None
    dangling_tail = re.search(
        r'(?i)(?:[=+\-*/×x:]|\\\[|\\\(|\\text\{?|\{|\(|\[|\b(?:of|from|by|for|to|the|a|an|in|on|at)\b)\s*$',
        last_line,
    ) is not None

    # Tail like "... = 9 x 2" usually means the generation stopped before
    # writing the final computed value.
    dangling_expression = re.search(
        r'(?i)(?:=\s*)?-?\d+(?:,\d{3})*(?:\.\d+)?\s*[×x*/+\-]\s*-?\d+(?:,\d{3})*(?:\.\d+)?\s*$',
        last_line,
    ) is not None

    if has_number and not dangling_tail and not dangling_expression:
        return False

    # Ends with punctuation is less likely to be cut off mid-thought.
    if s[-1] in '.!?)]}':
        return False

    # Unbalanced delimiters near the tail are a strong sign of truncation.
    if last_line.count('{') > last_line.count('}'):
        return True
    if last_line.count('(') > last_line.count(')'):
        return True
    if last_line.count('[') > last_line.count(']'):
        return True

    return True


def extract_predicted_answer(text, conservative=False):
    """Extract final numeric answer from free-form GSM8K output."""
    s = str(text)

    # Prefer explicit final-answer style markers when present.
    marker_match = re.search(r'####+\s*([^\n\r]+)', s)
    if marker_match:
        marker_value = _to_float(marker_match.group(1))
        if marker_value is not None:
            return marker_value

    angle_match = re.findall(r'<<\s*([^<>]+?)\s*>>', s)
    if angle_match:
        angle_value = _to_float(angle_match[-1])
        if angle_value is not None:
            return angle_value

    boxed_match = re.findall(r'\\boxed\{([^{}]+)\}', s)
    if boxed_match:
        boxed_value = _to_float(boxed_match[-1])
        if boxed_value is not None:
            return boxed_value

    final_line_patterns = [
        r'(?im)^\s*(?:final\s+answer|answer)\s*[:：-]?\s*(.+)$',
        r'(?i)(?:the\s+)?final\s+(?:numerical\s+)?answer\s*(?:is|:)\s*(.+)',
        r'(?i)(?:the\s+)?answer\s*(?:is|:)\s*(.+)',
        r'(?im)^\s*(?:thus|therefore|so),?\s+.*?\b(?:is|are)\s+(.+)$',
        r'(?im)^\s*(?:thus|therefore|so),?\s+.*?\b(?:makes?|earns?|gets?|costs?)\s+(.+)$',
    ]
    for pattern in final_line_patterns:
        matches = re.findall(pattern, s)
        if matches:
            # For explicit final-answer spans, use first numeric token to avoid
            # being distracted by trailing confidence/auxiliary numbers.
            final_value = _to_float_first(matches[-1])
            if final_value is not None:
                return final_value

    if conservative and _looks_truncated(s):
        return None

    return _to_float(s)


def extract_ground_truth(text):
    """Aligned with tianlwang/eval_gsm8k utils.py::extract_ground_truth."""
    return _to_float(str(text).split('####')[-1].strip())


def _pick_prediction_number(candidates, conservative=False):
    values = [extract_predicted_answer(c, conservative=conservative) for c in candidates]
    values = [v for v in values if v is not None]
    if not values:
        return None

    # Use majority vote under numeric buckets for repeated decoding runs.
    buckets = [round(v, 6) for v in values]
    counter = Counter(buckets)
    return float(counter.most_common(1)[0][0])


def _is_correct_numeric(pred, gold, tol=1e-6):
    if pred is None or gold is None:
        return 0
    return 1 if abs(float(pred) - float(gold)) <= tol else 0


def _load_gsm8k_test(args):
    """Load GSM8K test set with optional local cache path."""
    question_file = getattr(args, 'question_file', '')
    if question_file:
        return get_files(question_file)

    cache_dir = getattr(args, 'gsm8k_cache_dir', '') or dataset_cache_dir('gsm8k', args=args)
    dataset = load_dataset('gsm8k', 'main', cache_dir=cache_dir)
    return dataset['test']


def gsm8k_evaluation(args):
    question_set = _load_gsm8k_test(args)
    answer_set = get_files(args.generation_path)['answers']
    uq_set = get_files(args.uq_path) if args.uq_eval else None

    total, exact_match = 0, 0
    f1, correct = 0.0, 0.0
    rougel = 0.0
    accuracy = {} # key is qid, value is acc/uq
    model_name = str(getattr(args, 'model_name', '')).lower()
    conservative_parse = ('gpt4o' in model_name) or ('gpt-4o' in model_name)

    for idx, question_instance in enumerate(tqdm(question_set)):
        question_id = str(question_instance.get('idx', idx))
        standard_answer = extract_ground_truth(question_instance['answer'])
        
        if len([d for d in answer_set if d.get("qid") == question_id]) == 0:
            continue
        else:
            possible_answer = [d for d in answer_set if d.get("qid") == question_id]
            if len(possible_answer) == 1:
                current_answer = possible_answer[0]
            else:
                current_answer = possible_answer[-1]
            candidate_answers = current_answer.get('answers', [])
            pred_value = _pick_prediction_number(candidate_answers, conservative=conservative_parse)
            current_acc = _is_correct_numeric(pred_value, standard_answer)

            total += 1
            accuracy[question_id] = current_acc
            correct += current_acc
            exact_match += current_acc
            # Keep report fields compatible with existing pipeline output shape.
            f1 += current_acc
            rougel += current_acc
            
    if not args.uq_eval:
        return build_qa_report(total, exact_match, f1, correct, rougel, accuracy)
        
    # Create a list to store data for DataFrame
    data = []
    # Align uq_set with accuracy dictionary
    if 'kle' in args.uq_path:
        for question_id in accuracy:
            # Find matching item in uq_set where qid equals question_id
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                uq_item = matching_items[0]['entropies']  # Take the first match
                
                uq_item['accuracy'] = accuracy[question_id]
                uq_item['qid'] = question_id
                # Extract entropies if they exist
                data.append(uq_item)

        df = pd.DataFrame(data)
        # For the 'kle' path, calculate AUROC and AUARC for all columns except 'accuracy' and 'qid'
        results = {}
        # Get all columns except 'accuracy' and 'qid'
        uq_columns = [col for col in df.columns if col not in ['accuracy', 'qid']]
        
        # Calculate AUROC and AUARC for each column
        for column in uq_columns:
            auroc_score = auroc(df[column], df['accuracy'])
            auarc_score = area_under_accuracy_coverage_curve(df[column], df['accuracy'])
            
            # Store results with column name as key prefix
            results[f"{column}_auroc"] = auroc_score
            results[f"{column}_auarc"] = auarc_score
        
        # Calculate basic metrics
        exact_match = 100.0*exact_match/total
        f1 = 100.0*f1/total
        correct = 100.0*correct/total
        rougel = 100.0*rougel/total
        
        # Add basic metrics to results
        results['exact_match'] = exact_match
        results['f1'] = f1
        results['correct'] = correct
        results['rougel'] = rougel
        results['total'] = total
        
        print('--------------------------------')
        print(f"Total: {total}")
        print(f"Exact match: {exact_match:.2f}")
        print(f"F1 score: {f1:.2f}")
        print(f"Correct: {correct:.2f}")
        print(f"Rouge-L: {rougel:.2f}")
        print('--------------------------------')
        
        # Print AUROC and AUARC for each column
        for column in uq_columns:
            print(f"{column}_auroc: {results[f'{column}_auroc']:.4f}")
            print(f"{column}_auarc: {results[f'{column}_auarc']:.4f}")
        
        return results
    
    else:
        # For non-'kle' paths, align uq_set with accuracy
        for question_id in accuracy:
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                uq_item = matching_items[0]
                uq_value = uq_item.get('entropy', uq_item.get('uncertainty', 0))
                data.append({
                    'qid': question_id,
                    'uq': uq_value,
                    'accuracy': accuracy[question_id]
                })
        
        df = pd.DataFrame(data)
        
        # Calculate AUROC and AUARC
        auroc_score = auroc(df['uq'], df['accuracy'])
        auarc_score = area_under_accuracy_coverage_curve(df['uq'], df['accuracy'])
        
        # Calculate basic metrics
        exact_match = 100.0*exact_match/total
        f1 = 100.0*f1/total
        correct = 100.0*correct/total
        rougel = 100.0*rougel/total
        
        results = {
            'auroc': auroc_score,
            'auarc': auarc_score,
            'exact_match': exact_match,
            'f1': f1,
            'correct': correct,
            'rougel': rougel,
            'total': total
        }
        
        print('--------------------------------')
        print(f"Total: {total}")
        print(f"Exact match: {exact_match:.2f}")
        print(f"F1 score: {f1:.2f}")
        print(f"Correct: {correct:.2f}")
        print(f"Rouge-L: {rougel:.2f}")
        print(f"AUROC: {auroc_score:.4f}")
        print(f"AUARC: {auarc_score:.4f}")
        print('--------------------------------')
        
        return results
