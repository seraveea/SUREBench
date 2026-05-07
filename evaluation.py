import json
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from eval_pipelines.trivia_eval import trivia_evaluation
from eval_pipelines.coqa_eval import coqa_evaluation
from eval_pipelines.nq_eval import nq_evaluation
from eval_pipelines.squad_eval import squad_evaluation
from eval_pipelines.bioasq_eval import bioasq_evaluation
from eval_pipelines.svamp_eval import svamp_evaluation
from eval_pipelines.gsm8k_eval import gsm8k_evaluation
from eval_pipelines.gen_eval import (
    xsum_evaluation,
    aeslc_evaluation,
    cnn_dailymail_evaluation,
    multi_news_evaluation,
    wmt19_evaluation,
)
from eval_pipelines.utils import get_files, evaluate_uq_with_accuracy


def _resolve_generation_path(args):
    if args.generation_path:
        return args.generation_path

    dataset_variants = [args.dataset]
    if args.dataset == 'trivia':
        dataset_variants.append('triviaqa')
    if args.dataset == 'triviaqa':
        dataset_variants.append('trivia')

    candidates = []
    for ds_name in dataset_variants:
        candidates.extend([
            f"output/{ds_name}/{args.model_name}/{ds_name}_{args.model_name}.json",
            f"output/{ds_name}/{ds_name}_{args.model_name}.json",
            f"output/{ds_name}_{args.model_name}.json",
        ])
    for path in candidates:
        if os.path.exists(path):
            return path

    # Fall back to the newest layout candidate for clearer error visibility.
    return candidates[0]


def _resolve_uq_path(args):
    if args.uq_path:
        return args.uq_path
    return f"uq_output/{args.dataset}/{args.model_name}/{args.dataset}_{args.model_name}_{args.uq_type}.json"


def _resolve_qa_output_path(args):
    if args.qa_eval_path:
        return args.qa_eval_path
    return f"eval_results/{args.dataset}_{args.model_name}_accuracy.json"


def _resolve_uq_output_path(args):
    if args.uq_eval_path:
        return args.uq_eval_path
    return f"eval_results/{args.dataset}_{args.model_name}_{args.uq_type}_eval.json"


def _get_evaluator(dataset):
    evaluator_map = {
        'triviaqa': trivia_evaluation,
        'coqa': coqa_evaluation,
        'nq': nq_evaluation,
        'squad': squad_evaluation,
        'bioasq': bioasq_evaluation,
        'svamp': svamp_evaluation,
        'gsm8k': gsm8k_evaluation,
        'xsum': xsum_evaluation,
        'aeslc': aeslc_evaluation,
        'cnn_dailymail': cnn_dailymail_evaluation,
        'multi_news': multi_news_evaluation,
        'wmt19': wmt19_evaluation,
    }
    if dataset not in evaluator_map:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return evaluator_map[dataset]


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {path}")


def _run_qa_eval(args):
    evaluator = _get_evaluator(args.dataset)
    args.uq_eval = False
    qa_result = evaluator(args)
    qa_output_file = _resolve_qa_output_path(args)
    _save_json(qa_result, qa_output_file)
    return qa_result, qa_output_file


def _extract_accuracy_map(qa_result):
    if isinstance(qa_result, dict) and 'per_question_accuracy' in qa_result:
        per_q = qa_result['per_question_accuracy']
        if isinstance(per_q, dict) and per_q:
            sample_value = next(iter(per_q.values()))
            # Summarization tasks store per-question ROUGE/BLEU metrics.
            # Use ROUGE-L >= 50 as the positive label.
            if isinstance(sample_value, dict):
                return {
                    str(qid): 1 if float(metrics.get('rouge_l', 0.0)) >= 50.0 else 0
                    for qid, metrics in per_q.items()
                }
        return per_q

    # Backward compatibility with old accuracy-only JSON format.
    if isinstance(qa_result, dict) and qa_result and all(isinstance(v, (int, bool)) for v in qa_result.values()):
        return qa_result

    raise ValueError("Cannot find per-question accuracy in QA evaluation output.")


def _extract_rouge_l_map(qa_result):
    """
    Extract continuous ROUGE-L scores for summarization tasks.
    
    Returns ROUGE-L values in range [0, 100] instead of binary accuracy.
    """
    if isinstance(qa_result, dict) and 'per_question_accuracy' in qa_result:
        per_q = qa_result['per_question_accuracy']
        if isinstance(per_q, dict) and per_q:
            sample_value = next(iter(per_q.values()))
            # Check if it contains ROUGE metrics (dict format)
            if isinstance(sample_value, dict):
                return {
                    str(qid): float(metrics.get('rouge_l', 0.0))
                    for qid, metrics in per_q.items()
                }
            # If already continuous values, return as-is
            elif isinstance(sample_value, (int, float)):
                return {str(qid): float(v) for qid, v in per_q.items()}
    
    raise ValueError("Cannot find ROUGE-L scores in QA evaluation output.")


def _run_uq_eval(args):
    uq_path = _resolve_uq_path(args)
    qa_output_file = _resolve_qa_output_path(args)

    qa_result = get_files(qa_output_file)
    
    # For summarization tasks, extract continuous ROUGE-L values instead of binary accuracy
    summarization_tasks = ['xsum', 'aeslc', 'cnn_dailymail', 'multi_news', 'wmt19']
    is_summarization = any(task in args.dataset for task in summarization_tasks)
    
    if is_summarization:
        # Extract continuous ROUGE-L scores for summarization tasks
        accuracy_map = _extract_rouge_l_map(qa_result)
    else:
        # Extract binary accuracy for QA tasks
        accuracy_map = _extract_accuracy_map(qa_result)
    
    uq_set = get_files(uq_path)

    uq_result = evaluate_uq_with_accuracy(
        uq_set=uq_set,
        accuracy=accuracy_map,
        uq_path=uq_path,
        uq_string=args.uq_string,
        is_continuous=is_summarization,
    )
    uq_output_file = _resolve_uq_output_path(args)
    _save_json(uq_result, uq_output_file)
    return uq_result, uq_output_file


def main(args):
    args.generation_path = _resolve_generation_path(args)
    args.uq_path = _resolve_uq_path(args)

    if args.task == 'qa':
        _run_qa_eval(args)
        return

    if args.task == 'uq':
        _run_uq_eval(args)
        return

    # both
    _run_qa_eval(args)
    _run_uq_eval(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='', help='Dataset root directory. If empty, use SUREBENCH_DATA_ROOT or ./data')
    parser.add_argument('--task', choices=['qa', 'uq', 'both'], default='qa', help='评估任务：qa仅问答效果，uq仅不确定性效果，both按顺序都跑')
    parser.add_argument('--generation_path', default='', help="""答案生成的json文件路径（可选，不传则按约定路径自动推断）""")
    parser.add_argument('--uq_path', default='', help="""uncertainty的json文件路径（可选，不传则按约定路径自动推断）""")
    parser.add_argument('--qa_eval_path', default='', help="""QA评估结果json路径（可选）""")
    parser.add_argument('--uq_eval_path', default='', help="""UQ评估结果json路径（可选）""")
    parser.add_argument('--uq_eval', action='store_true', help="""兼容旧代码，现由--task控制，可忽略""")
    parser.add_argument('--uq_string', default='discrete_semantic_uncertainty', help="""uncertainty的字段名，有的uq methods里面会有多个uq""")
    parser.add_argument('--dataset', default='squad',help="""数据集选择 triviaqa, nq, coqa, squad, bioasq, svamp, gsm8k, xsum, aeslc, cnn_dailymail, multi_news, wmt19""")
    parser.add_argument('--model_name', default='llama8', help="""模型名称""")
    parser.add_argument('--uq_type', default='de_dse', help="""dse, gwc or kle""")
    parser.add_argument('--wmt19_subset', default='cs-en', help='wmt19子集，例如 cs-en, de-en, fi-en, gu-en, kk-en, lt-en, ru-en, zh-en')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    main(args)
