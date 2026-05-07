import json
import torch
import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from path_config import model_path
from uq_methods.dse import *
from uq_methods.gwc import *
from uq_methods.kle import kle
from uq_methods.consistency import ConsistencyScorer
from uq_methods.snne import compute_discrete_snne
from uq_methods.luq import compute_discrete_luq_pair
from uq_methods.continuous_suq import (
    compute_continuous_luq,
    compute_continuous_snne,
    compute_continuous_semantic_density,
    compute_continuous_gwc,
)
from uq_methods.semantic_energy import compute_semantic_energy
from uq_methods.semantic_density import compute_discrete_semantic_density
import warnings
from transformers import logging as transformers_logging

# Suppress duplicated transformers max_length/max_new_tokens warnings.
transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='.*max_new_tokens.*max_length.*')

def get_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def pipeline_instance(args):
    model_dir = model_path('Meta-Llama-3-8B-Instruct_hf', args=args)
    gen_pipeline = pipeline(
        "text-generation",
        model=model_dir,
        device_map=args.device,
        torch_dtype=torch.float16  # Use half precision.
    )
    return gen_pipeline


class CacheOnlyNLI:
    """Cache-backed NLI that never loads a model and fails on cache miss."""

    def __init__(self, cache_path: str):
        self.cache = {}
        self.cache_path = cache_path
        loaded = self.load_cache(cache_path)
        print(f"[NLI cache-only] loaded {loaded} entries from {cache_path}")

    def _cache_key(self, text1, text2, question=None):
        q = "" if question is None else str(question)
        return (q, str(text1), str(text2))

    def check_implication(self, text1, text2, *args, **kwargs):
        question = args[0] if len(args) > 0 else kwargs.get("question", None)
        key = self._cache_key(text1, text2, question)

        # Identical answers should be treated as confident entailment even if
        # this pair was not precomputed in cache.
        if str(text1) == str(text2):
            self.cache[key] = (2, 0.999)
            return self.cache[key]

        if key in self.cache:
            return self.cache[key]
        raise KeyError(
            f"NLI cache miss for question={question!r}. "
            "Cache-only mode is enabled, so no model inference is performed."
        )

    def load_cache(self, cache_path):
        if not cache_path or (not os.path.exists(cache_path)):
            return 0
        with open(cache_path, 'rb') as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            self.cache.update(loaded)
            return len(loaded)
        return 0

    def save_cache(self, cache_path):
        if not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.cache, f)



def main(args):
    answer_set = get_questions(args.answer_set_path)

    needs_nli = args.uq_method in {
        "discrete_semantic_uncertainty",
        "discrete_gwc",
        "continuous_gwc",
        "kle",
        "discrete_snne",
        "discrete_luq_pair",
        "semantic_energy",
        "discrete_semantic_density",
        "continuous_luq",
        "continuous_snne",
        "semantic_density_continuous",
    }
    nli_model = None
    consistency_scorer = None

    if needs_nli:
        # If cache file exists, run in cache-only mode.
        # If cache path is provided but file is missing, build cache from scratch.
        use_cache_only = bool(args.nli_cache_path and os.path.exists(args.nli_cache_path))
        if args.nli_cache_path and not use_cache_only:
            print(
                f"[NLI cache] missing file at {args.nli_cache_path}, "
                "will build from scratch and save during this run"
            )
        if use_cache_only:
            nli_model = CacheOnlyNLI(args.nli_cache_path)
        else:
            if args.nli_model == "deberta":
                nli_model = EntailmentDeberta(args.device)
            else:
                qa_pipeline = pipeline_instance(args)
                nli_model = EntailmentLLM(qa_pipeline)
            if args.nli_cache_path:
                loaded = nli_model.load_cache(args.nli_cache_path)
                print(f"[NLI cache] loaded {loaded} entries from {args.nli_cache_path}")
    elif args.uq_method == "consistency_triplet":
        consistency_scorer = ConsistencyScorer(device=args.device)

    answer_set = answer_set['answers']
    result = []
    for idx, i in enumerate(tqdm(answer_set), start=1):
        if args.uq_method == "discrete_semantic_uncertainty":
            entropy = discrete_semantic_entropy(i['answers'], nli_model, i['question'])
            uq = entropy
            result.append(
                {
                    'qid': i['qid'],
                    'discrete_semantic_uncertainty': uq
                }
            )
        elif args.uq_method == "discrete_gwc":
            W = compute_weighted_matrix(nli_model, i['answers'], i['question'])
            L = compute_laplacian_matrix(W)
            U_eigv = compute_U_eigv(L)
            U_deg, C_deg = compute_UC_deg(W)
            U_ecc, C_ecc = compute_UC_ecc(
                L,
                k=min(5, len(i['answers'])),
                mask_eigenvalue=args.eigv_threshold,
            )
            result.append(
                {
                    'qid': i['qid'],
                    'U_eigv': U_eigv,
                    'U_deg':U_deg,
                    'C_deg': C_deg.tolist() if isinstance(C_deg, np.ndarray) else C_deg,
                    'U_ecc': U_ecc,
                    'C_ecc': C_ecc.tolist() if isinstance(C_ecc, np.ndarray) else C_ecc
                }
            )
        elif args.uq_method == "continuous_gwc":
            cache_confidence_dict = getattr(nli_model, "cache", None)
            gwc_result = compute_continuous_gwc(
                i['answers'],
                nli_model,
                i['question'],
                cache_confidence_dict=cache_confidence_dict,
                bidirectional=True,
                mask_eigenvalue=args.eigv_threshold,
            )
            gwc_result['qid'] = i['qid']
            result.append(gwc_result)
        elif args.uq_method == "kle":
            result_dict = kle(i['answers'], nli_model, i['question'], args)
            result_dict['qid'] = i['qid']
            result.append(result_dict)
        elif args.uq_method == "consistency_triplet":
            scores = consistency_scorer.score_responses(i['answers'])
            result.append(
                {
                    'qid': i['qid'],
                    'exact_match': scores['exact_match'],
                    'bert_score': scores['bert_score'],
                    'cosine_sim': scores['cosine_sim'],
                }
            )
        elif args.uq_method == "discrete_snne":
            snne_result = compute_discrete_snne(i['answers'], nli_model, i['question'], args)
            snne_result['qid'] = i['qid']
            result.append(snne_result)
        elif args.uq_method == "discrete_luq_pair":
            luq_result = compute_discrete_luq_pair(i['answers'], nli_model, i['question'], args)
            luq_result['qid'] = i['qid']
            result.append(luq_result)
        elif args.uq_method == "continuous_luq":
            cache_confidence_dict = getattr(nli_model, "cache", None)
            luq_result = compute_continuous_luq(
                i['answers'],
                i['question'],
                cache_confidence_dict=cache_confidence_dict,
                bidirectional=bool(getattr(args, "luq_bidirectional", False)),
                model=nli_model,
            )
            luq_result['qid'] = i['qid']
            result.append(luq_result)
        elif args.uq_method == "continuous_snne":
            cache_confidence_dict = getattr(nli_model, "cache", None)
            snne_result = compute_continuous_snne(
                i['answers'],
                i['question'],
                cache_confidence_dict=cache_confidence_dict,
                temperature=args.snne_temperature,
                exclude_diagonal=args.snne_self_similarity,
                variant=args.snne_variant,
                bidirectional=True,
                model=nli_model,
            )
            snne_result['qid'] = i['qid']
            result.append(snne_result)
        elif args.uq_method == "semantic_energy":
            se_result = compute_semantic_energy(i, nli_model, args)
            se_result['qid'] = i['qid']
            result.append(se_result)
        elif args.uq_method == "discrete_semantic_density":
            sd_result = compute_discrete_semantic_density(i['answers'], nli_model, i['question'], args)
            sd_result['qid'] = i['qid']
            result.append(sd_result)
        elif args.uq_method == "continuous_semantic_density":
            cache_confidence_dict = getattr(nli_model, "cache", None)
            sd_result = compute_continuous_semantic_density(
                i['answers'],
                nli_model,
                i['question'],
                cache_confidence_dict=cache_confidence_dict,
                bidirectional=True,
            )
            sd_result['qid'] = i['qid']
            result.append(sd_result)
        else:
            raise ValueError(f"Unknown uq_method: {args.uq_method}")

        if needs_nli and args.nli_cache_path and args.nli_cache_autosave_interval > 0:
            if idx % args.nli_cache_autosave_interval == 0:
                nli_model.save_cache(args.nli_cache_path)
                print(f"[NLI cache] autosaved at {idx} samples -> {args.nli_cache_path}")

    json.dump(result, open(args.result_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'结果已保存到 {args.result_file}')
    if needs_nli and args.nli_cache_path:
        nli_model.save_cache(args.nli_cache_path)
        print(f"[NLI cache] saved {len(nli_model.cache)} entries to {args.nli_cache_path}")
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nli_model', default='llama-8b', help="""选择用来NLI的模型: llama-8b,  qwen-1.5b, llama-70b, deberta""")
    parser.add_argument('--model_root', default='', help='Model root directory. If empty, use SUREBENCH_MODEL_ROOT or ./models')
    parser.add_argument('--answer_set_path', default='output/trivia_wiki_llama8.json', help='大模型答案路径')
    parser.add_argument('--strict_entailment', default='False', help='默认false')
    parser.add_argument('--result_file', default='uq_output/trivia_wiki_llama8_kle.json', help='uncertianty计算保存路径')
    parser.add_argument('--device', default="cuda:0", help='使用的设备')
    parser.add_argument('--uq_method', default="kle", help='discrete_semantic_uncertainty, discrete_gwc, continuous_gwc, kle, consistency_triplet, discrete_snne, discrete_luq_pair, continuous_luq, continuous_snne, semantic_energy, discrete_semantic_density, semantic_density_continuous')
    parser.add_argument('--nli_type', default="llm", help='选择NLI类型: nlp或llm')
    parser.add_argument('--eigv_threshold', type=float, default=0.9, help='gwc中ecc特征向量阈值, 保留eig<阈值')
    parser.add_argument('--snne_variant', default='full', choices=['full', 'only_num', 'only_denom', 'num_minus_denom'], help='SNNE损失变体')
    parser.add_argument('--snne_temperature', type=float, default=1.0, help='SNNE温度参数')
    parser.add_argument('--snne_self_similarity', action=argparse.BooleanOptionalAction, default=True, help='SNNE是否保留对角线自相似度')
    parser.add_argument('--luq_bidirectional', action=argparse.BooleanOptionalAction, default=False, help='LUQ相似度是否双向平均')
    parser.add_argument('--luq_include_neutral', action=argparse.BooleanOptionalAction, default=False, help='LUQ相似度是否计入neutral=0.5')
    parser.add_argument('--nli_cache_path', default='', help='NLI结果缓存文件(.pkl)，可跨方法复用以减少重复调用')
    parser.add_argument('--nli_cache_autosave_interval', type=int, default=200, help='每处理多少条样本自动保存一次NLI缓存，<=0表示不自动保存')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    