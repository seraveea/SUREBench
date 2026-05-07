import json
import re
import string
from collections import Counter
from sklearn import metrics
import pandas as pd
import numpy as np

def normalize_answer(s: object) -> object:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def get_files(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def find_most_common(answers):
    # Count answer frequencies.
    counter = Counter(answers)
    most_common = counter.most_common()
    if not most_common:
        return ''
    if len(most_common) == 1:
        return most_common[0][0]
    # If there is a unique most frequent answer, return it.
    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    # If there is no clear mode, default to the first answer.
    return answers[0]


def area_under_accuracy_coverage_curve(uq, acc):
    df = pd.DataFrame({"u": uq, 'a': acc}).sort_values('u', ascending=True)
    df['amean'] = df['a'].expanding().mean()
    return metrics.auc(np.linspace(0,1,len(df)), df['amean'])

def auroc(uq,acc):
    fpr, tpr, thresholds = metrics.roc_curve(acc.astype(int), -uq, pos_label=1)
    return metrics.auc(fpr, tpr)


def _normalize_for_prr(target):
    target = np.asarray(target, dtype=float)
    min_t, max_t = np.min(target), np.max(target)
    if np.isclose(min_t, max_t):
        min_t -= 1.
        max_t += 1.0
    return (target - min_t) / (max_t - min_t)



def prr(uq, acc):
    """
    SNNE-compatible PRR (implemented as aucpr in SNNE).

    This follows SNNE's prediction-rejection computation exactly:
    1) min-max normalize targets
    2) sort by uncertainty ascending (keep least-uncertain first)
    3) compute reversed prefix-average scores and average them
    
    Args:
        uq: uncertainty scores (higher means more uncertain)
        acc: target quality scores (binary or continuous)
    
    Returns:
        PRR score, where higher is better
    """
    y_true = _normalize_for_prr(acc)
    ue = np.asarray(uq, dtype=float)

    ue_argsort = np.argsort(ue)
    sorted_metrics = y_true[ue_argsort]

    num_obs = len(sorted_metrics)
    if num_obs == 0:
        return None
    cumsum = np.cumsum(sorted_metrics)
    scores = (cumsum / np.arange(1, num_obs + 1))[::-1]
    prr_score = float(np.sum(scores) / num_obs)
    return prr_score


def prr_continuous(uq, quality_scores):
    """
    SNNE-compatible PRR for continuous quality scores.

    SNNE's PRR already supports continuous targets via min-max normalization,
    so this is an alias for PRR with continuous inputs.
    
    Args:
        uq: uncertainty scores
        quality_scores: continuous quality scores (e.g., ROUGE-L)
    
    Returns:
        PRR score, where higher is better
    """
    return prr(uq, quality_scores)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def lcs_length(x, y):
    """Compute the length of the longest common subsequence between x and y."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(candidate, reference):
    """Compute ROUGE-L score."""
    lcs = lcs_length(candidate, reference)
    precision = lcs / len(candidate) if candidate else 0
    recall = lcs / len(reference) if reference else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_rouge_l(candidate, list_of_reference):
    return max([rouge_l(candidate.split(), i.split()) for i in list_of_reference])


def update_eval(answer, standard_answer, total, f1, accuracy, exact_match, correct, question_id):
    total += 1
    em_for_this_question = metric_max_over_ground_truths(exact_match_score, answer, standard_answer)
    
    # Update EM, F1, and ROUGE-L.
    exact_match += em_for_this_question
    f1 += metric_max_over_ground_truths(f1_score, answer, standard_answer)
    current_rougel = compute_rouge_l(answer, standard_answer)

    if em_for_this_question:
        correct += 1
        accuracy[question_id] = 1

    else:
        accuracy[question_id] = 0
    return total, f1, accuracy, exact_match, correct, em_for_this_question, current_rougel


def build_qa_report(total, exact_match, f1, correct, rougel, accuracy):
    if total == 0:
        return {
            "average_exact_match": 0.0,
            "average_f1": 0.0,
            "average_accuracy": 0.0,
            "average_rougel": 0.0,
            "Number of questions": 0,
            "per_question_accuracy": accuracy,
        }

    return {
        "average_exact_match": 100.0 * exact_match / total,
        "average_f1": 100.0 * f1 / total,
        "average_accuracy": 100.0 * correct / total,
        "average_rougel": 100.0 * rougel / total,
        "Number of questions": total,
        "per_question_accuracy": accuracy,
    }


def _safe_auroc(uq, acc):
    # roc_curve requires both positive and negative labels.
    if len(set(acc.astype(int).tolist())) < 2:
        return None
    return auroc(uq, acc)


def _safe_prr(uq, acc, is_continuous=False):
    """
    Safe wrapper for PRR calculation.
    
    Args:
        uq: uncertainty scores
        acc: accuracy labels (binary: 0/1) or quality scores (continuous: 0-100)
        is_continuous: if True, treats acc as continuous quality scores
    
    Returns:
        PRR score or None if validation fails
    """
    if len(uq) == 0 or len(acc) == 0:
        return None

    if is_continuous:
        return prr_continuous(uq, acc)

    return prr(uq, acc)


def _to_numeric_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        try:
            return float(np.mean(value))
        except Exception:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _resolve_uq_value(uq_item, uq_string, uq_path):
    """Resolve uncertainty value with method-aware fallbacks."""
    v = uq_item.get(uq_string)
    if _to_numeric_value(v) == _to_numeric_value(v):  # not NaN
        return v

    candidates = []
    if 'luq' in uq_path:
        candidates.extend(['luq_pair', 'luq'])
    if 'snne' in uq_path:
        candidates.extend(['snne'])
    if 'semantic_density' in uq_path:
        candidates.extend(['semantic_density_uncertainty', 'semantic_density'])

    candidates.extend([
        'discrete_semantic_uncertainty',
        'cluster_assignment_entropy',
        'entropy',
    ])

    for key in candidates:
        if key in uq_item:
            cand = uq_item.get(key)
            if _to_numeric_value(cand) == _to_numeric_value(cand):
                return cand

    # Last resort: first numeric scalar-like field (skip obvious metadata fields).
    skip_keys = {
        'qid', 'question_id', 'semantic_ids', 'answers', 'predictions',
        'luq_pair_per_response', 'snne_similarity', 'snne_variant',
        'snne_temperature', 'snne_self_similarity',
        'luq_bidirectional', 'luq_include_neutral',
        'semantic_density_per_response', 'semantic_density_unique_responses',
        'semantic_density_unique_frequencies', 'semantic_density_mean_per_response',
    }
    for key, value in uq_item.items():
        if key in skip_keys:
            continue
        if _to_numeric_value(value) == _to_numeric_value(value):
            return value

    return None


def _compute_uq_metrics(uq_series, acc_series, use_prr=False, is_continuous=False):
    """
    Compute uncertainty quality metrics.
    
    Args:
        uq_series: Uncertainty scores
        acc_series: Accuracy labels (binary: 0/1) or continuous quality scores (0-100)
        use_prr: If True, use PRR instead of AUROC; if False, use AUROC
        is_continuous: If True, treats acc_series as continuous quality scores (for summarization)
    
    Returns:
        (main_metric_score, secondary_metric_score) where:
        - main_metric: PRR if use_prr=True, otherwise AUROC
        - secondary_metric: AUARC in both cases
    """
    tmp_df = pd.DataFrame({
        'uq': uq_series.map(_to_numeric_value),
        'accuracy': acc_series,
    }).dropna(subset=['uq'])

    if tmp_df.empty:
        return None, None

    if use_prr:
        main_score = _safe_prr(tmp_df['uq'], tmp_df['accuracy'], is_continuous=is_continuous)
    else:
        main_score = _safe_auroc(tmp_df['uq'], tmp_df['accuracy'])
    
    # For continuous scores, AUARC calculation should also be adjusted
    if not is_continuous:
        auarc_score = area_under_accuracy_coverage_curve(tmp_df['uq'], tmp_df['accuracy'])
    else:
        # For continuous quality scores, calculate coverage-quality curve instead
        # This measures average quality at different coverage levels
        auarc_score = area_under_accuracy_coverage_curve(tmp_df['uq'], tmp_df['accuracy'])
    
    return main_score, auarc_score


def evaluate_uq_with_accuracy(uq_set, accuracy, uq_path, uq_string, is_continuous=False):
    """
    Evaluate uncertainty quality against accuracy labels.
    
    Args:
        uq_set: List of UQ result dictionaries
        accuracy: Dict mapping question IDs to accuracy/quality scores
        uq_path: Path to UQ file (used for task detection)
        uq_string: Name of the UQ field to evaluate
        is_continuous: If True, accuracy contains continuous quality scores (0-100) for summarization tasks
    
    For summarization tasks with continuous quality scores,
    uses PRR as the primary metric. For other tasks, uses AUROC.
    """
    if not accuracy:
        return {
            "Number of matched questions": 0,
            "warning": "Empty accuracy mapping; cannot evaluate UQ.",
        }

    # Detect if this is a summarization task based on uq_path
    summarization_tasks = ['xsum', 'aeslc', 'cnn_dailymail', 'multi_news', 'wmt19']
    is_summarization = any(task in uq_path for task in summarization_tasks)
    
    # Use continuous PRR if explicitly requested or if summarization + is_continuous
    use_continuous_prr = is_continuous and is_summarization
    data = []

    # consistency_triplet stores confidence-style consistency scores.
    # Higher score => more reliable => lower uncertainty, so convert by (1 - score).
    if 'consistency_triplet' in uq_path:
        consistency_cols = ['exact_match', 'bert_score', 'cosine_sim']
        for question_id, acc in accuracy.items():
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                uq_item = matching_items[0]
                row = {
                    'qid': question_id,
                    'accuracy': acc,
                }
                for col in consistency_cols:
                    row[col] = uq_item.get(col)
                data.append(row)

        df = pd.DataFrame(data)
        if df.empty:
            return {
                "Number of matched questions": 0,
                "warning": "No overlap between UQ file and accuracy file.",
            }

        results = {"Number of matched questions": len(df)}
        for col in consistency_cols:
            if col not in df.columns:
                continue

            col_df = df[[col, 'accuracy']].copy()
            col_df[col] = col_df[col].map(_to_numeric_value)
            col_df = col_df.dropna(subset=[col])
            if col_df.empty:
                if is_summarization:
                    results[f"{col}_prr"] = None
                else:
                    results[f"{col}_auroc"] = None
                results[f"{col}_auarc"] = None
                continue

            # Convert confidence to uncertainty for ranking-based UQ metrics.
            uq_as_uncertainty = 1.0 - col_df[col]
            main_score, auarc_score = _compute_uq_metrics(
                uq_as_uncertainty,
                col_df['accuracy'],
                use_prr=is_summarization,
                is_continuous=use_continuous_prr,
            )

            if is_summarization:
                results[f"{col}_prr"] = main_score
            else:
                results[f"{col}_auroc"] = main_score
            results[f"{col}_auarc"] = auarc_score

        return results

    if 'kle' in uq_path:
        for question_id, acc in accuracy.items():
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items and isinstance(matching_items[0].get('entropies'), dict):
                uq_item = dict(matching_items[0]['entropies'])
                uq_item['accuracy'] = acc
                uq_item['qid'] = question_id
                data.append(uq_item)

        df = pd.DataFrame(data)
        if df.empty:
            return {
                "Number of matched questions": 0,
                "warning": "No overlap between UQ file and accuracy file.",
            }

        results = {"Number of matched questions": len(df)}
        uq_columns = [col for col in df.columns if col not in ['accuracy', 'qid']]
        for column in uq_columns:
            main_score, auarc_score = _compute_uq_metrics(
                df[column], df['accuracy'], 
                use_prr=is_summarization,
                is_continuous=use_continuous_prr
            )
            if is_summarization:
                results[f"{column}_prr"] = main_score
            else:
                results[f"{column}_auroc"] = main_score
            results[f"{column}_auarc"] = auarc_score
        return results

    if 'gwc' in uq_path:
        for question_id, acc in accuracy.items():
            matching_items = [item for item in uq_set if item.get('qid') == question_id]
            if matching_items:
                # For GWC, evaluate both U* and C* uncertainty fields.
                uq_item = {
                    k: v for k, v in matching_items[0].items()
                    if k.startswith('U') or k.startswith('C')
                }
                uq_item['accuracy'] = acc
                uq_item['qid'] = question_id
                data.append(uq_item)

        df = pd.DataFrame(data)
        if df.empty:
            return {
                "Number of matched questions": 0,
                "warning": "No overlap between UQ file and accuracy file.",
            }

        results = {"Number of matched questions": len(df)}
        uq_columns = [col for col in df.columns if col not in ['accuracy', 'qid']]
        for column in uq_columns:
            main_score, auarc_score = _compute_uq_metrics(
                df[column], df['accuracy'],
                use_prr=is_summarization,
                is_continuous=use_continuous_prr
            )
            if is_summarization:
                results[f"{column}_prr"] = main_score
            else:
                results[f"{column}_auroc"] = main_score
            results[f"{column}_auarc"] = auarc_score
        return results

    for question_id, acc in accuracy.items():
        matching_items = [item for item in uq_set if item.get('qid') == question_id]
        if matching_items:
            uq_item = matching_items[0]
            data.append({
                'qid': question_id,
                'accuracy': acc,
                'uq': _resolve_uq_value(uq_item, uq_string, uq_path),
            })

    df = pd.DataFrame(data)
    if df.empty:
        return {
            "Number of matched questions": 0,
            "warning": "No overlap between UQ file and accuracy file.",
        }

    main_score, auarc_score = _compute_uq_metrics(
        df['uq'], df['accuracy'], 
        use_prr=is_summarization,
        is_continuous=use_continuous_prr
    )
    
    if is_summarization:
        return {
            "Number of matched questions": len(df),
            "PRR": main_score,
            "AUARC": auarc_score,
        }
    else:
        return {
            "Number of matched questions": len(df),
            "AUROC": main_score,
            "AUARC": auarc_score,
        }