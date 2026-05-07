import torch
from uq_methods.dse import get_semantic_ids



def _normalize_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip().lower()
    return " ".join(text.split())


def _lcs_len(tokens_a, tokens_b):
    m, n = len(tokens_a), len(tokens_b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l_f1(text_a, text_b):
    a = _normalize_text(text_a).split()
    b = _normalize_text(text_b).split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    lcs = _lcs_len(a, b)
    precision = lcs / len(a)
    recall = lcs / len(b)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def lexical_similarity_matrix(strings_list):
    n = len(strings_list)
    sim = torch.eye(n, dtype=torch.float32)
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = _rouge_l_f1(strings_list[i], strings_list[j])
            sim[i, j] = s
            sim[j, i] = s
    return sim


def snne(
    similarity_matrix,
    labels,
    variant="only_denom",
    temperature=1.0,
    epsilon=1e-8,
    exclude_diagonal=True,
    weight=None,
):
    # Keep the formula consistent with the official implementation.
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int64)

    labels = labels.view(-1, 1)
    label_mask = labels != labels.T
    label_inf = torch.zeros_like(similarity_matrix)
    label_inf[label_mask] = float("-inf")

    similarity_matrix = similarity_matrix / float(temperature)
    if exclude_diagonal:
        diag_inf = torch.diag(torch.tensor(float("-inf")).expand(labels.size(0)))
        similarity_matrix = similarity_matrix + diag_inf

    logsumexp_numerators = torch.logsumexp(similarity_matrix + label_inf, dim=1, keepdim=True)
    logsumexp_denominators = torch.logsumexp(similarity_matrix, dim=1, keepdim=True)

    inf_mask = torch.isinf(logsumexp_numerators)
    logsumexp_numerators[inf_mask] = torch.log(torch.tensor(epsilon))

    if variant == "full":
        loss = logsumexp_numerators - logsumexp_denominators
    elif variant == "only_num":
        loss = logsumexp_numerators
    elif variant == "only_denom":
        # when using this, different NLI models may have almost identified output
        loss = logsumexp_denominators
    elif variant == "num_minus_denom":
        loss = (
            2 * torch.exp(logsumexp_numerators)
            - torch.exp(logsumexp_denominators)
            + torch.exp(torch.tensor(1.0 / float(temperature))) * logsumexp_numerators.size(0)
        )
        loss = torch.log(loss)
    else:
        raise ValueError(f"Unknown SNNE variant: {variant}")

    if weight is None:
        weight = torch.ones_like(loss)
    elif weight.size() != loss.size():
        weight = weight.view(-1, 1)

    loss = -(loss * weight).mean()
    return loss


def compute_discrete_snne(responses, model, question, args):
    strict_entailment = args.strict_entailment
    if isinstance(strict_entailment, str):
        strict_entailment = strict_entailment.lower() == "true"

    if not responses:
        return {
            "semantic_ids": [],
            "snne": 0.0,
            "snne_similarity": "lexical_rouge_l",
            "snne_variant": args.snne_variant,
            "snne_temperature": args.snne_temperature,
            "snne_self_similarity": args.snne_self_similarity,
        }

    # Follow the official workflow: cluster semantic IDs first, then compute SNNE on the similarity matrix.
    semantic_ids = get_semantic_ids(
        responses,
        model,
        question,
        strict_entailment=strict_entailment,
    )
    sim = lexical_similarity_matrix(responses)
    loss = snne(
        similarity_matrix=sim,
        labels=semantic_ids,
        variant=args.snne_variant,
        temperature=args.snne_temperature,
        exclude_diagonal=not args.snne_self_similarity,
    )

    return {
        "semantic_ids": semantic_ids,
        "snne": float(loss.detach().cpu().item()),
        "snne_similarity": "lexical_rouge_l",
        "snne_variant": args.snne_variant,
        "snne_temperature": float(args.snne_temperature),
        "snne_self_similarity": bool(args.snne_self_similarity),
    }
