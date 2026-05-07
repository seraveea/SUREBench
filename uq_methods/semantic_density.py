from collections import Counter

import numpy as np



def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _response_distance_from_label(label):
    # Entailment / neutral / contradiction -> 0 / 1/2 / 1 in the kernel domain.
    if label == 2:
        return 0.0
    if label == 1:
        return 0.5
    return 1.0


def _pairwise_semantic_distance(model, text_a, text_b, question, strict_entailment=False, bidirectional=True):
    if text_a == text_b:
        return 0.0

    def one_direction(source, target):
        implication, _ = model.check_implication(source, target, question)
        if strict_entailment:
            return 0.0 if implication == 2 else 1.0

        return _response_distance_from_label(int(implication))

    distance = one_direction(text_a, text_b)
    if bidirectional:
        distance = max(distance, one_direction(text_b, text_a))
    return float(distance)


def _dimension_invariant_kernel(distance):
    # Compact kernel in semantic space: 1 for equivalent, 1/2 for neutral, 0 for contradiction.
    return float(max(0.0, 1.0 - distance))


def _safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_discrete_semantic_density(responses, model, question, args):
    """
    Black-box semantic density fallback that does not require logits.

    This follows the paper's Eq. 5 style estimator:
    1) deduplicate the sampled responses;
    2) use their relative frequencies as empirical probabilities;
    3) estimate semantic similarity with pairwise NLI;
    4) aggregate the kernel-weighted density as a response-wise confidence.
    """
    if not responses:
        return {
            "semantic_density": 0.0,
            "semantic_density_uncertainty": 1.0,
            "semantic_density_per_response": [],
            "semantic_density_unique_responses": [],
            "semantic_density_unique_frequencies": [],
        }

    strict_entailment = _as_bool(args.strict_entailment)
    counts = Counter("" if response is None else str(response) for response in responses)
    unique_responses = list(counts.keys())
    total = float(sum(counts.values()))
    unique_frequencies = [counts[response] / total for response in unique_responses]

    pairwise_kernel = []
    for response_i in unique_responses:
        row = []
        for response_j in unique_responses:
            distance = _pairwise_semantic_distance(
                model,
                response_i,
                response_j,
                question,
                strict_entailment=strict_entailment,
                bidirectional=True,
            )
            row.append(_dimension_invariant_kernel(distance))
        pairwise_kernel.append(row)

    pairwise_kernel = np.asarray(pairwise_kernel, dtype=np.float64)
    frequency_vector = np.asarray(unique_frequencies, dtype=np.float64)

    # Density for each unique response: sum_i f_i K(v* - v_i).
    unique_density = pairwise_kernel @ frequency_vector
    semantic_density = float(np.dot(frequency_vector, unique_density))
    semantic_uncertainty = float(1.0 - semantic_density)

    unique_density_map = {
        response: float(unique_density[idx]) for idx, response in enumerate(unique_responses)
    }
    per_response_density = [unique_density_map["" if response is None else str(response)] for response in responses]

    return {
        "semantic_density": semantic_density,
        "semantic_density_uncertainty": semantic_uncertainty,
        "semantic_density_per_response": per_response_density,
        "semantic_density_unique_responses": unique_responses,
        "semantic_density_unique_frequencies": unique_frequencies,
        "semantic_density_mean_per_response": _safe_mean(per_response_density),
    }