import numpy as np


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _entailment_similarity_score(
    model,
    text1,
    text2,
    question,
    strict_entailment=False,
    exclude_neutral=True,
    bidirectional=False,
):
    # The NLI interface in this repo returns a discrete label (0/1/2) + confidence.
    def one_direction(a, b):
        implication, _ = model.check_implication(a, b, question)
        if strict_entailment:
            return 1.0 if implication == 2 else 0.0
        if exclude_neutral:
            return 1.0 if implication == 2 else 0.0
        if implication == 2:
            return 1.0
        if implication == 1:
            return 0.5
        return 0.0

    score = one_direction(text1, text2)
    if bidirectional:
        score = (score + one_direction(text2, text1)) / 2.0
    return score


def entailment_similarity_matrix(
    model,
    strings_list,
    question,
    strict_entailment=False,
    exclude_neutral=True,
    bidirectional=False,
):
    n = len(strings_list)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim[i, j] = _entailment_similarity_score(
                model,
                strings_list[i],
                strings_list[j],
                question,
                strict_entailment=strict_entailment,
                exclude_neutral=exclude_neutral,
                bidirectional=bidirectional,
            )
    return sim


def get_luq_pair(similarity_matrix):
    sim = np.array(similarity_matrix, dtype=np.float32, copy=True)
    if sim.shape[0] == 0:
        return 0.0, []

    # LUQ-pair: 1 - max_j S[i, j], ignoring self-similarity.
    np.fill_diagonal(sim, 0.0)
    per_response = np.asarray(1.0 - np.max(sim, axis=1), dtype=np.float32)
    return float(per_response.mean()), per_response.tolist()


def compute_discrete_luq_pair(responses, model, question, args):
    strict_entailment = _as_bool(args.strict_entailment)
    sim = entailment_similarity_matrix(
        model,
        responses,
        question,
        strict_entailment=strict_entailment,
        exclude_neutral=not args.luq_include_neutral,
        bidirectional=args.luq_bidirectional,
    )
    luq, per_response = get_luq_pair(sim)
    return {
        "luq_pair": luq,
        "luq_pair_per_response": per_response,
        "luq_bidirectional": bool(args.luq_bidirectional),
        "luq_include_neutral": bool(args.luq_include_neutral),
    }
