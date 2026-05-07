from collections import Counter

import numpy as np
import torch



def _as_text(value):
    if value is None:
        return ""
    return str(value)


def _extract_prediction(result):
    if isinstance(result, tuple) and len(result) >= 2:
        label, confidence = result[0], result[1]
        try:
            return int(label), float(confidence)
        except Exception:
            return int(label), 0.0
    if isinstance(result, dict):
        label = result.get("label", result.get("prediction", 0))
        confidence = result.get("confidence", 0.0)
        return int(label), float(confidence)
    return int(result), 0.0


def _resolve_cache_confidence_dict(nli_source=None, cache_confidence_dict=None):
    """Resolve a current-cache style dict from either a model or a raw cache object."""
    if cache_confidence_dict is not None:
        return cache_confidence_dict
    if isinstance(nli_source, dict):
        return nli_source
    if hasattr(nli_source, "cache") and isinstance(getattr(nli_source, "cache"), dict):
        return getattr(nli_source, "cache")
    return None


def _lookup_prediction(model, text1, text2, question, cache_confidence_dict=None):
    key = (_as_text(question), _as_text(text1), _as_text(text2))
    if cache_confidence_dict is not None and key in cache_confidence_dict:
        return _extract_prediction(cache_confidence_dict[key])

    if hasattr(model, "check_implication"):
        result = model.check_implication(text1, text2, question)
        return _extract_prediction(result)

    return 0, 0.0


def _semantic_similarity_from_prediction(label, confidence):
    """Approximate 1 - semantic_distance from the official semantic-density code."""
    if label == 2:
        return float(confidence)
    if label == 1:
        return float(0.5 * confidence)
    return 0.0


def _semantic_density_similarity_from_prediction(label, confidence):
    """Approximate the official semantic-density score with only the top-class confidence.

    The official implementation needs the full 3-way logits to compute
    semantic_distance = p(contra) + 0.5 * p(neutral).
    Here we do not store the other two logits, so we approximate them by
    splitting the remaining mass equally:

        p(other1) = p(other2) = (1 - p(top)) * 0.5

    This preserves the top prediction and keeps the score continuous.
    """
    top = float(confidence)
    other = max(0.0, 1.0 - top) * 0.5

    if label == 2:
        semantic_distance = other + 0.5 * other
    elif label == 1:
        semantic_distance = other + 0.5 * top
    else:
        semantic_distance = top + 0.5 * other

    return float(1.0 - semantic_distance)


def _pairwise_continuous_similarity(model, text_a, text_b, question, cache_confidence_dict=None, bidirectional=True):
    if _as_text(text_a) == _as_text(text_b):
        return 1.0

    label_ab, conf_ab = _lookup_prediction(model, text_a, text_b, question, cache_confidence_dict)
    score_ab = _semantic_similarity_from_prediction(label_ab, conf_ab)

    if not bidirectional:
        return score_ab

    label_ba, conf_ba = _lookup_prediction(model, text_b, text_a, question, cache_confidence_dict)
    score_ba = _semantic_similarity_from_prediction(label_ba, conf_ba)
    return float(0.5 * (score_ab + score_ba))


def continuous_entailment_similarity_matrix(model, strings_list, question, cache_confidence_dict=None, bidirectional=False):
    """
    cache_confidence_dict: dict[(q, t1, t2)] -> (label, confidence)
    """
    cache_confidence_dict = _resolve_cache_confidence_dict(model, cache_confidence_dict)
    n = len(strings_list)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            score = _pairwise_continuous_similarity(
                model,
                strings_list[i],
                strings_list[j],
                question,
                cache_confidence_dict=cache_confidence_dict,
                bidirectional=bidirectional,
            )
            sim[i, j] = score
            sim[j, i] = score
    return sim


def continuous_get_luq_pair(similarity_matrix):
    sim = np.array(similarity_matrix, dtype=np.float32, copy=True)
    if sim.shape[0] == 0:
        return 0.0, []

    np.fill_diagonal(sim, 0.0)
    per_response = np.asarray(1.0 - np.max(sim, axis=1), dtype=np.float32)
    return float(per_response.mean()), per_response.tolist()


def compute_continuous_luq(responses, question, cache_confidence_dict, bidirectional=False, model=None):
    cache_confidence_dict = _resolve_cache_confidence_dict(model, cache_confidence_dict)
    sim = continuous_entailment_similarity_matrix(
        model,
        responses,
        question,
        cache_confidence_dict=cache_confidence_dict,
        bidirectional=bidirectional,
    )
    luq, per_response = continuous_get_luq_pair(sim)
    return {
        "semantic_similarity_matrix": sim.tolist(),
        "semantic_density": 1.0 - luq,
        "semantic_density_uncertainty": luq,
        "luq_pair": luq,
        "luq_pair_per_response": per_response,
        "luq_bidirectional": bool(bidirectional),
    }


def continuous_snne(
    similarity_matrix,
    variant="full",
    temperature=1.0,
    epsilon=1e-8,
    exclude_diagonal=True,
    weight=None,
):
    # Continuous version: no hard semantic labels, use the confidence-weighted kernel directly.
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)

    if similarity_matrix.numel() == 0:
        return torch.tensor(0.0)

    similarity_matrix = similarity_matrix / float(temperature)
    if exclude_diagonal:
        diag_inf = torch.full((similarity_matrix.size(0),), float("-inf"), device=similarity_matrix.device)
        similarity_matrix = similarity_matrix + torch.diag(diag_inf)

    positive_weight = similarity_matrix.clamp(min=0.0, max=1.0)
    negative_weight = (1.0 - positive_weight).clamp(min=0.0, max=1.0)

    numer_logits = similarity_matrix + torch.log(positive_weight + epsilon)
    denom_logits = similarity_matrix + torch.log(positive_weight + negative_weight + epsilon)

    if variant == "only_denom":
        loss = torch.logsumexp(denom_logits, dim=1, keepdim=True)
    elif variant == "only_num":
        loss = torch.logsumexp(numer_logits, dim=1, keepdim=True)
    elif variant == "full":
        numer = torch.logsumexp(numer_logits, dim=1, keepdim=True)
        denom = torch.logsumexp(denom_logits, dim=1, keepdim=True)
        loss = numer - denom
    elif variant == "num_minus_denom":
        numer = torch.logsumexp(numer_logits, dim=1, keepdim=True)
        denom = torch.logsumexp(denom_logits, dim=1, keepdim=True)
        loss = torch.log(torch.clamp(torch.exp(numer) - torch.exp(denom), min=epsilon))
    else:
        raise ValueError(f"Unknown continuous SNNE variant: {variant}")

    if weight is None:
        weight = torch.ones_like(loss)
    elif weight.size() != loss.size():
        weight = weight.view(-1, 1)

    loss = -(loss * weight).mean()
    return loss


def compute_continuous_snne(
    responses,
    question,
    cache_confidence_dict,
    temperature=1.0,
    exclude_diagonal=True,
    variant="full",
    bidirectional=True,
    model=None,
):
    cache_confidence_dict = _resolve_cache_confidence_dict(model, cache_confidence_dict)
    sim = continuous_entailment_similarity_matrix(
        model,
        responses,
        question,
        cache_confidence_dict=cache_confidence_dict,
        bidirectional=bidirectional,
    )
    loss = continuous_snne(
        similarity_matrix=sim,
        variant=variant,
        temperature=temperature,
        exclude_diagonal=exclude_diagonal,
    )
    return {
        "semantic_similarity_matrix": sim.tolist(),
        "continuous_snne": float(loss.detach().cpu().item()),
        "snne": float(loss.detach().cpu().item()),
        "continuous_snne_temperature": float(temperature),
        "continuous_snne_exclude_diagonal": bool(exclude_diagonal),
        "continuous_snne_variant": variant,
        "continuous_snne_bidirectional": bool(bidirectional),
    }


def _safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compute_continuous_semantic_density(
    responses,
    model,
    question,
    cache_confidence_dict=None,
    bidirectional=True,
):
    """Continuous semantic density following the official paper's weighted kernel view.

    The official implementation weights pairwise semantic density by generation likelihood.
    In this repository we only have sampled answers and the NLI cache, so we estimate the
    density from the empirical response frequencies and the cached confidence scores.
    """
    cache_confidence_dict = _resolve_cache_confidence_dict(model, cache_confidence_dict)
    if not responses:
        return {
            "semantic_density": 0.0,
            "semantic_density_uncertainty": 1.0,
            "semantic_density_per_response": [],
            "semantic_density_unique_responses": [],
            "semantic_density_unique_frequencies": [],
            "semantic_density_mean_per_response": 0.0,
        }

    counts = Counter(_as_text(response) for response in responses)
    unique_responses = list(counts.keys())
    total = float(sum(counts.values()))
    unique_frequencies = [counts[response] / total for response in unique_responses]

    pairwise_kernel = []
    for response_i in unique_responses:
        row = []
        for response_j in unique_responses:
            if _as_text(response_i) == _as_text(response_j):
                similarity = 1.0
            else:
                label_ab, conf_ab = _lookup_prediction(
                    model,
                    response_i,
                    response_j,
                    question,
                    cache_confidence_dict,
                )
                sim_ab = _semantic_density_similarity_from_prediction(label_ab, conf_ab)
                if bidirectional:
                    label_ba, conf_ba = _lookup_prediction(
                        model,
                        response_j,
                        response_i,
                        question,
                        cache_confidence_dict,
                    )
                    sim_ba = _semantic_density_similarity_from_prediction(label_ba, conf_ba)
                    similarity = float(0.5 * (sim_ab + sim_ba))
                else:
                    similarity = sim_ab
            row.append(similarity)
        pairwise_kernel.append(row)

    pairwise_kernel = np.asarray(pairwise_kernel, dtype=np.float64)
    frequency_vector = np.asarray(unique_frequencies, dtype=np.float64)

    unique_density = pairwise_kernel @ frequency_vector
    semantic_density = float(np.dot(frequency_vector, unique_density))
    semantic_uncertainty = float(1.0 - semantic_density)

    unique_density_map = {
        response: float(unique_density[idx]) for idx, response in enumerate(unique_responses)
    }
    per_response_density = [_as_text(response) for response in responses]
    per_response_density = [unique_density_map[response] for response in per_response_density]

    return {
        "semantic_density": semantic_density,
        "semantic_density_uncertainty": semantic_uncertainty,
        "semantic_density_per_response": per_response_density,
        "semantic_density_unique_responses": unique_responses,
        "semantic_density_unique_frequencies": unique_frequencies,
        "semantic_density_mean_per_response": _safe_mean(per_response_density),
    }


def _continuous_compute_laplacian_matrix(W):
    W = np.asarray(W, dtype=np.float64)
    num_strings = len(W)
    degree_diag = np.sum(W, axis=1)
    degree_sqrt_inv = np.diag(1.0 / np.sqrt(degree_diag + 1e-10))
    identity = np.eye(num_strings)
    return identity - degree_sqrt_inv @ W @ degree_sqrt_inv


def _continuous_compute_u_eigv(L):
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = np.sort(eigenvalues.real)
    uncertainty = 0.0
    for eigenvalue in eigenvalues:
        if 1.0 - eigenvalue > 0:
            uncertainty += 1.0 - eigenvalue
    return float(uncertainty)


def _continuous_compute_uc_deg(W):
    W = np.asarray(W, dtype=np.float64)
    num_strings = len(W)
    degree_diag = np.sum(W, axis=1)
    identity = np.eye(num_strings)
    degree = np.diag(degree_diag)
    result = (num_strings * identity - degree) / (num_strings ** 2)
    uncertainty = float(np.trace(result))
    confidence = degree_diag / num_strings
    return uncertainty, confidence


def _continuous_compute_uc_ecc(L, mask_eigenvalue=0.5):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    keep_mask = eigenvalues > mask_eigenvalue
    _, selected_eigenvectors = eigenvalues[keep_mask], eigenvectors[:, keep_mask]
    selected_eigenvectors = selected_eigenvectors.T
    confidence = (-1) * np.asarray([
        np.linalg.norm(vector - vector.mean(0), 2) for vector in selected_eigenvectors
    ])
    uncertainty = np.linalg.norm(confidence, 2)
    return float(uncertainty), confidence


def compute_continuous_gwc(
    responses,
    model,
    question,
    cache_confidence_dict=None,
    bidirectional=True,
    mask_eigenvalue=0.5,
):
    """Continuous GWC using the cached NLI confidence as a soft affinity matrix."""
    if not responses:
        return {
            "qid": None,
            "U_eigv": 0.0,
            "U_deg": 0.0,
            "C_deg": [],
            "U_ecc": 0.0,
            "C_ecc": [],
            "continuous_gwc_similarity_matrix": [],
        }

    cache_confidence_dict = _resolve_cache_confidence_dict(model, cache_confidence_dict)
    W = continuous_entailment_similarity_matrix(
        model,
        responses,
        question,
        cache_confidence_dict=cache_confidence_dict,
        bidirectional=bidirectional,
    )
    L = _continuous_compute_laplacian_matrix(W)
    U_eigv = _continuous_compute_u_eigv(L)
    U_deg, C_deg = _continuous_compute_uc_deg(W)
    U_ecc, C_ecc = _continuous_compute_uc_ecc(L, mask_eigenvalue=mask_eigenvalue)

    return {
        "U_eigv": U_eigv,
        "U_deg": U_deg,
        "C_deg": C_deg.tolist() if hasattr(C_deg, "tolist") else C_deg,
        "U_ecc": U_ecc,
        "C_ecc": C_ecc.tolist() if hasattr(C_ecc, "tolist") else C_ecc,
        "continuous_gwc_similarity_matrix": W.tolist(),
        "continuous_gwc_laplacian": L.tolist(),
    }
