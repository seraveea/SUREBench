from collections import Counter

from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm import tqdm

from eval_pipelines.utils import (
    compute_rouge_l,
    find_most_common,
    get_files,
    normalize_answer,
)
from path_config import dataset_cache_dir


def _collect_answer_map(generation_path):
    answers = get_files(generation_path)["answers"]
    answer_map = {}
    for item in answers:
        qid = str(item.get("qid"))
        answer_map[qid] = item
    return answer_map


def _select_prediction(item):
    candidate_answer = [normalize_answer(x) for x in item.get("answers", [])]
    if not candidate_answer:
        return ""
    most_common_answer = find_most_common(candidate_answer)
    return most_common_answer if most_common_answer else candidate_answer[0]


def _ngram_counter(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(pred_tokens, ref_tokens, n):
    pred_counts = _ngram_counter(pred_tokens, n)
    ref_counts = _ngram_counter(ref_tokens, n)

    if not pred_counts or not ref_counts:
        return 0.0

    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


def _best_rouge_n_f1(pred_tokens, refs_tokens, n):
    if not refs_tokens:
        return 0.0
    return max(_rouge_n_f1(pred_tokens, ref_tokens, n) for ref_tokens in refs_tokens)


def _best_rouge_l_f1(pred_text, refs_text):
    return compute_rouge_l(pred_text, refs_text) if refs_text else 0.0


def _sentence_bleu_score(pred_tokens, refs_tokens):
    if not pred_tokens or not refs_tokens:
        return 0.0
    return sentence_bleu(refs_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1)


def _evaluate_summary_from_reference_map(answer_map, reference_map):
    total = 0
    rouge_1_sum = 0.0
    rouge_2_sum = 0.0
    rouge_l_sum = 0.0
    bleu_sum = 0.0
    per_question_metrics = {}

    for qid, refs in tqdm(reference_map.items()):
        qid = str(qid)
        if qid not in answer_map:
            continue

        pred = _select_prediction(answer_map[qid])
        refs = [normalize_answer(r) for r in refs if isinstance(r, str) and r]
        if not refs:
            continue

        pred_tokens = pred.split()
        refs_tokens = [ref.split() for ref in refs]

        rouge_1 = _best_rouge_n_f1(pred_tokens, refs_tokens, 1)
        rouge_2 = _best_rouge_n_f1(pred_tokens, refs_tokens, 2)
        rouge_l = _best_rouge_l_f1(pred, refs)
        bleu = _sentence_bleu_score(pred_tokens, refs_tokens)

        total += 1
        rouge_1_sum += rouge_1
        rouge_2_sum += rouge_2
        rouge_l_sum += rouge_l
        bleu_sum += bleu

        # Store per-question ROUGE/BLEU metrics instead of binary accuracy.
        per_question_metrics[qid] = {
            "rouge_1": rouge_1 * 100.0,
            "rouge_2": rouge_2 * 100.0,
            "rouge_l": rouge_l * 100.0,
            "bleu": bleu * 100.0,
        }

    if total == 0:
        return {
            "average_rouge_1": 0.0,
            "average_rouge_2": 0.0,
            "average_rouge_l": 0.0,
            "average_bleu": 0.0,
            "Number of questions": 0,
            "per_question_accuracy": per_question_metrics,
        }

    return {
        "average_rouge_1": 100.0 * rouge_1_sum / total,
        "average_rouge_2": 100.0 * rouge_2_sum / total,
        "average_rouge_l": 100.0 * rouge_l_sum / total,
        "average_bleu": 100.0 * bleu_sum / total,
        "Number of questions": total,
        "per_question_accuracy": per_question_metrics,
    }


def xsum_evaluation(args):
    print("Loading XSum dataset for evaluation...")
    ds = load_dataset("xsum", split="test", cache_dir=dataset_cache_dir("xsum", args=args))
    reference_map = {}
    for idx, sample in enumerate(ds):
        qid = str(sample.get("id", str(idx)))
        reference_map[qid] = [sample.get("summary", "")]
    answer_map = _collect_answer_map(args.generation_path)
    return _evaluate_summary_from_reference_map(answer_map, reference_map)


def aeslc_evaluation(args):
    ds = load_dataset("aeslc", split="test", cache_dir=dataset_cache_dir("aeslc", args=args))
    reference_map = {}
    for idx, sample in enumerate(ds):
        qid = str(sample.get("id", str(idx)))
        reference_map[qid] = [sample.get("subject_line", "")]
    answer_map = _collect_answer_map(args.generation_path)
    return _evaluate_summary_from_reference_map(answer_map, reference_map)


def cnn_dailymail_evaluation(args):
    ds = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split="test",
        cache_dir=dataset_cache_dir("cnn_dailymail", args=args),
    )
    reference_map = {}
    for idx, sample in enumerate(ds):
        qid = str(sample.get("id", str(idx)))
        reference_map[qid] = [sample.get("highlights", "")]
    answer_map = _collect_answer_map(args.generation_path)
    return _evaluate_summary_from_reference_map(answer_map, reference_map)


def multi_news_evaluation(args):
    ds = load_dataset(
        "Awesome075/multi_news_parquet",
        split="test",
        cache_dir=dataset_cache_dir("multi_news", args=args),
    )
    reference_map = {}
    for idx, sample in enumerate(ds):
        qid = str(idx)
        reference_map[qid] = [sample.get("summary", "")]
    answer_map = _collect_answer_map(args.generation_path)
    return _evaluate_summary_from_reference_map(answer_map, reference_map)


def wmt19_evaluation(args):
    if "-" not in args.wmt19_subset:
        raise ValueError("wmt19_subset format should be src-tgt, e.g. zh-en")
    source_lang, target_lang = args.wmt19_subset.split("-", 1)
    ds = load_dataset(
        "wmt19",
        args.wmt19_subset,
        split="validation",
        cache_dir=dataset_cache_dir("wmt19", args=args),
    )

    answer_map = _collect_answer_map(args.generation_path)
    reference_map = {}
    for idx, sample in enumerate(ds):
        qid = str(idx)
        translation = sample.get("translation", {})
        reference_map[qid] = [translation.get(target_lang, "")]

    qa_result = _evaluate_summary_from_reference_map(answer_map, reference_map)

    # WMT official recommendation uses BLEU; report corpus BLEU additionally.
    refs_tokens = []
    preds_tokens = []
    for qid, refs in reference_map.items():
        if qid not in answer_map:
            continue
        pred = _select_prediction(answer_map[qid])
        refs_tokens.append([normalize_answer(refs[0]).split()])
        preds_tokens.append(normalize_answer(pred).split())

    if preds_tokens:
        bleu = corpus_bleu(refs_tokens, preds_tokens, smoothing_function=SmoothingFunction().method1)
    else:
        bleu = 0.0

    qa_result["corpus_bleu"] = bleu * 100.0
    qa_result["translation_direction"] = f"{source_lang}-{target_lang}"
    return qa_result
