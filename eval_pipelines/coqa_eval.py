from tqdm import tqdm
import json
from eval_pipelines.utils import normalize_answer, get_files, find_most_common, update_eval, area_under_accuracy_coverage_curve, auroc, build_qa_report
import pandas as pd
from path_config import data_path

def coqa_evaluation(args):
    question_set = get_files(data_path('coqa', 'coqa-dev-v1.0.json', args=args))['data']
    answer_set = get_files(args.generation_path)['answers']
    uq_set = get_files(args.uq_path) if args.uq_eval else None

    total, exact_match = 0, 0
    f1, correct = 0.0, 0.0
    rougel = 0.0
    accuracy = {} # key is qid, value is acc/uq

    for story_instance in tqdm(question_set):
        story_id  = story_instance['id']
        for question in story_instance['questions']:
            question_id = story_id + "/" + str(question['turn_id'])
            standard_answer = [d for d in story_instance['answers'] if d.get("turn_id") == question['turn_id']][0]
            standard_answer = [normalize_answer(standard_answer['input_text']),
                               normalize_answer(standard_answer['span_text'])]
            standard_answer = list(set(standard_answer))

            if len([d for d in answer_set if d.get("qid") == question_id]) == 0:
                continue
            else:
                possible_answer = [d for d in answer_set if d.get("qid") == question_id]
                if len(possible_answer) == 1:
                    current_answer = possible_answer[0]
                else:
                    current_answer = possible_answer[-1]
                candidate_answer = current_answer['answers']
                candidate_answer = [normalize_answer(item) for item in candidate_answer]
                # Pick the most frequent answer; if there is no clear mode, use the first one.
                most_common_answer = find_most_common(candidate_answer)
                if most_common_answer:
                    candidate_answer = [most_common_answer]
                else:
                    print([d for d in answer_set if d.get("qid") == question_id][0])
                    candidate_answer = [candidate_answer[0]]

                total, f1, accuracy, exact_match, correct, current_em, current_rougel = update_eval(candidate_answer[0], standard_answer,
                                                                        total, f1, accuracy, exact_match, correct,
                                                                        question_id)
                rougel += current_rougel


    if not args.uq_eval:
        return build_qa_report(total, exact_match, f1, correct, rougel, accuracy)
    
    if 'kle' in args.uq_path:
        data = []
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
    elif 'gwc' in args.uq_path:
        data = []
        for question_id in accuracy:
            # Find matching item in uq_set where qid equals question_id
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                # uq_item = matching_items[0]  # Take the first match
                # Extract all fields starting with 'U'
                uq_item = {k: v for k, v in matching_items[0].items() if k.startswith('U')}
                
                uq_item['accuracy'] = accuracy[question_id]
                uq_item['qid'] = question_id
                # Extract entropies if they exist
                data.append(uq_item)

        df = pd.DataFrame(data)
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
        
        exact_match = 100.0*exact_match/total
        f1 = 100.0*f1/total
        correct = 100.0*correct/total
        rougel = 100.0*rougel/total
        results.update({
            "average_exact_match": exact_match, 
            "average_f1": f1, 
            "average_accuracy": correct, 
            "average_rougel": rougel,
            "Number of questions": total
        })
        
        return results
    else:
        # Create a list to store data for DataFrame
        data = []
        # Align uq_set with accuracy dictionary
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

