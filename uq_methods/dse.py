# implementation of Discrete Semantic Entropy (DSE) method
import os
import json
import pickle
import torch
from path_config import model_path
import numpy as np
from scipy.stats import entropy
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentDeberta:
    def __init__(self, device):
        model_dir = model_path('deberta-large-mnli')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.cache = {}

    def _cache_key(self, text1, text2, question=None):
        # Include question in the cache key to avoid accidental cross-question reuse.
        q = "" if question is None else str(question)
        return (q, str(text1), str(text2))

    def check_implication(self, text1, text2, *args, **kwargs):
        question = args[0] if len(args) > 0 else kwargs.get("question", None)
        key = self._cache_key(text1, text2, question)
        if key in self.cache:
            return self.cache[key]
        else:
            inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
            # The model checks if text1 -> text2, i.e. if text2 follows from text1.
            # check_implication('The weather is good', 'The weather is good and I like you') --> 1
            # check_implication('The weather is good and I like you', 'The weather is good') --> 2
            with torch.no_grad():

                outputs = self.model(**inputs)
                logits = outputs.logits
                # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
                activations = F.softmax(logits, dim=1)
                largest_index = torch.argmax(activations)  # pylint: disable=no-member
                confidence = torch.max(activations)
                prediction = largest_index.cpu().item()
                self.cache[key] = (prediction, confidence.cpu().item())
                # Clean up GPU memory by detaching and moving tensors to CPU
                del inputs, outputs
                return prediction, confidence
    
    def implication_logits(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        return logits.cpu().item()
    
    def clear_cache(self):
        """Clear the cache for every new question."""
        self.cache = {}

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


class EntailmentLLM:
    def __init__(self, pipeline):
        self.model = pipeline
        self.cache = {}

    def _cache_key(self, text1, text2, question=None):
        q = "" if question is None else str(question)
        return (q, str(text1), str(text2))

    
    def check_implication(self, text1, text2, question):
        content = f"""
        We are evaluating answers to the question \"{question}\"\n
        Here are two possible answers:\n
        Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n
        Does Possible Answer 1 semantically entail Possible Answer 2? 
        Respond with ONLY entailment, contradiction, or neutral.
        """
        key = self._cache_key(text1, text2, question)
        if key in self.cache:
            return self.cache[key]
        else:
            message = [{"role": "user", "content": content}]
            prompt = self.model.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            if self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>") is None:
                eos_pair = [self.model.tokenizer.eos_token_id, self.model.tokenizer.convert_tokens_to_ids("<|endoftext|>")] # qwen2.5-1.5b
            else:
                eos_pair = [self.model.tokenizer.eos_token_id, self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")] # llama
            output = self.model(prompt, max_new_tokens=128,
                                    eos_token_id=eos_pair,
                                    do_sample=True,
                                    temperature=0.1, top_p=0.9, pad_token_id=self.model.tokenizer.eos_token_id)
            result = output[0]["generated_text"][len(prompt):].lower()
            if 'entailment' in result:
                self.cache[key] = (2, 1)
                return 2,1
            elif 'neutral' in result:
                self.cache[key] = (1, 1)
                return 1,1
            elif 'contradiction' in result:
                self.cache[key] = (0, 1)
                return 0,1
            else:
                print('MANUAL NEUTRAL!')
                self.cache[key] = (1, 1)
                return 1,1
            
    def clear_cache(self):
        """Clear the cache for every new question."""
        self.cache = {}

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



def get_semantic_ids(strings_list, model, question, strict_entailment=False):
    """
    Group list of predictions into semantic meaning.
    Strings_list save all possible response.
    Compare each string using model.
    """

    def are_equivalent(model, text1, text2, question):
        # check whether text1 and text2 have the same meaning
        implication_1, _ = model.check_implication(text1, text2, question)
        implication_2, _ = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            """
            no zero and not all equal to 1
            """
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            # means never found equivalent response before, create a new group next_id
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(model, string1, strings_list[j], question):
                    # same with string1, give j string string1's id
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids  # make sure all responses has been assigned valid id

    return semantic_set_ids


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)  # number of different clusters
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def discrete_semantic_entropy(responses, model, question):
    semantic_ids = get_semantic_ids(responses, model, question)
    return cluster_assignment_entropy(semantic_ids)



if __name__ == '__main__':
    None
