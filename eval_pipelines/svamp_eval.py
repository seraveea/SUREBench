from tqdm import tqdm
import re
from eval_pipelines.utils import get_files, area_under_accuracy_coverage_curve, auroc, build_qa_report
import pandas as pd
from path_config import data_path



def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    # First try direct numeric conversion (handles plain numbers and scientific notation).
    try:
        return float(text.replace(',', ''))
    except Exception:
        pass

    # Support simple fraction forms like "3/4".
    frac_match = re.search(r'(-?\d+)\s*/\s*(-?\d+)', text)
    if frac_match:
        num = float(frac_match.group(1))
        den = float(frac_match.group(2))
        if den != 0:
            return num / den

    # Fallback to the last numeric span in free-form text.
    num_matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
    if not num_matches:
        return None
    try:
        return float(num_matches[-1].replace(',', ''))
    except Exception:
        return None


def _pick_prediction_number(candidates):
    values = [_to_float(c) for c in candidates]
    values = [v for v in values if v is not None]
    if not values:
        return None

    # Pick the most frequent numeric prediction (official setup uses one decode,
    # while our pipeline may keep multiple samples).
    buckets = [round(v, 6) for v in values]
    counter = {}
    for b in buckets:
        counter[b] = counter.get(b, 0) + 1
    best_bucket = max(counter.items(), key=lambda x: x[1])[0]
    return float(best_bucket)


def _is_correct_numeric(pred, gold, tol=0.1):
    if pred is None or gold is None:
        return 0
    return 1 if abs(float(pred) - float(gold)) <= tol else 0


def svamp_evaluation(args):
    question_set = get_files(data_path('SVAMP', 'test.json', args=args))
    answer_set = get_files(args.generation_path)['answers']
    uq_set = get_files(args.uq_path) if args.uq_eval else None

    total, exact_match = 0, 0
    f1, correct = 0.0, 0.0
    rougel = 0.0
    accuracy = {} # key is qid, value is acc/uq

    for question_instance in tqdm(question_set):
        question_id = question_instance['ID']
        standard_answer = _to_float(question_instance.get('Answer'))

        if len([d for d in answer_set if d.get("qid") == question_id]) == 0:
            continue
        else:
            possible_answer = [d for d in answer_set if d.get("qid") == question_id]
            if len(possible_answer) == 1:
                current_answer = possible_answer[0]
            else:
                current_answer = possible_answer[-1]
            candidate_answers = current_answer.get('answers', [])
            pred_value = _pick_prediction_number(candidate_answers)
            current_acc = _is_correct_numeric(pred_value, standard_answer, tol=0.1)

            total += 1
            accuracy[question_id] = current_acc
            correct += current_acc
            exact_match += current_acc
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
        results.update({
            "average_exact_match": exact_match, 
            "average_f1": f1, 
            "average_accuracy": correct, 
            "average_rougel": rougel,
            "Number of questions": total
        })
        
        return results

    else:
        for question_id in accuracy:
            
            # Find matching item in uq_set where qid equals question_id
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                uq_item = matching_items[0]  # Take the first match
                data.append({
                    'qid': question_id,
                    'accuracy': accuracy[question_id],
                    'uq': uq_item.get(args.uq_string)
                })

        # Create DataFrame
        df = pd.DataFrame(data)

        auroc_score = auroc(df['uq'], df['accuracy'])
        aurac = area_under_accuracy_coverage_curve(df['uq'], df['accuracy'])

        exact_match = 100.0*exact_match/total
        f1 = 100.0*f1/total
        correct = 100.0*correct/total
        rougel = 100.0*rougel/total


        return {"average_exact_match": exact_match, 
                "average_f1": f1, 
                "average_accuracy": correct, 
                "average_rougel": rougel,
                "AUROC":auroc_score,
                "AUARC": aurac,
                "Number of questions": total
                }

