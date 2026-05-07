import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from path_config import model_path


def _normalize_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return " ".join(text.strip().lower().split())


def exact_match_confidence(original, candidates):
    if not candidates:
        return 1.0

    original_norm = _normalize_text(original)
    matches = 0
    for cand in candidates:
        if _normalize_text(cand) == original_norm:
            matches += 1
    return matches / len(candidates)


class _EmbeddingEncoder:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy()


class ConsistencyScorer:
    def __init__(self, device="auto"):
        self.cosine_encoder = _EmbeddingEncoder(model_path('models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf'), device)
        self.bert_encoder = _EmbeddingEncoder(model_path('bert-uncased'), device)

    def cosine_similarity_confidence(self, original, candidates):
        if not candidates:
            return 1.0

        texts = [original] + candidates
        embeddings = self.cosine_encoder.encode(texts)
        anchor = embeddings[0]
        cands = embeddings[1:]
        sims = cands @ anchor
        return float(np.mean((sims + 1.0) / 2.0))

    def bert_score_confidence(self, original, candidates):
        if not candidates:
            return 1.0

        texts = [original] + candidates
        embeddings = self.bert_encoder.encode(texts)
        anchor = embeddings[0]
        cands = embeddings[1:]
        sims = cands @ anchor
        return float(np.mean((sims + 1.0) / 2.0))

    def score_responses(self, responses):
        if not responses:
            return {
                "exact_match": 0.0,
                "bert_score": 0.0,
                "cosine_sim": 0.0,
            }

        original = responses[0]
        candidates = responses[1:] if len(responses) > 1 else []

        return {
            "exact_match": exact_match_confidence(original, candidates),
            "bert_score": self.bert_score_confidence(original, candidates),
            "cosine_sim": self.cosine_similarity_confidence(original, candidates),
        }
