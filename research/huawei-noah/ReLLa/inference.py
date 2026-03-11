import argparse
import json
import re


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

DEFAULT_SYSTEM_PROMPT_ZH = """\
你是一个乐于助人、尊重他人且诚实的助手。请尽可能提供有帮助且安全的回答。你的回答不应包含任何有害、不道德、带有偏见、危险或违法的内容。请确保你的回答在社会层面保持客观、公正，并传递积极的信息。

如果问题本身没有意义，或者与事实不符，请解释原因，而不是给出不正确的回答。如果你不知道答案，请直接说明，而不要提供虚假信息。"""


def _contains_cjk(text):
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def _get_system_prompt(pair):
    if "system_prompt" in pair:
        return pair["system_prompt"]
    if "system" in pair:
        return pair["system"]
    if _contains_cjk(pair["input"]):
        return DEFAULT_SYSTEM_PROMPT_ZH
    return DEFAULT_SYSTEM_PROMPT


def get_prompt(pair):
    B_INST, E_INST = "<s>[INST]", "[/INST] "
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    system_prompt = B_SYS + _get_system_prompt(pair) + E_SYS
    prompt_template = B_INST + system_prompt + pair["input"] + E_INST + pair["output"]
    return prompt_template



def preprocess_logits_for_metrics(logits, labels):
    """
    labels: (N, seq_len), logits: (N, seq_len, 32000)
    """
    import mindspore
    import mindspore.ops as ops
    from mindspore import Tensor

    labels_index = ops.nonzero(ops.bitwise_or(labels == 3869, labels == 1939))

    gold = ops.select(labels[labels_index[:, 0], labels_index[:, 1]] == 1939,
                      Tensor(0, mindspore.int32), Tensor(1, mindspore.int32))

    labels_index[:, 1] = labels_index[:, 1] - 1
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [1939, 3869]]
    prob = ops.Softmax(axis=-1)(logits)

    return prob[:, 1], gold


def compute_metrics(prob, gold):
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    auc = roc_auc_score(gold, prob)
    ll = log_loss(gold, prob)
    acc = accuracy_score(gold, prob > 0.5)
    return {
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    }


def infer(args):
    import mindspore.numpy as mnp
    from mindformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, MindFormerConfig, init_context
    from tqdm import trange

    dataset = json.load(open(args.data_dir))
    dataset = [get_prompt(pair) for pair in dataset]

    llama_config = MindFormerConfig("run_llama2_7b_910b.yaml")

    init_context(use_parallel=llama_config.use_parallel,
                    context_config=llama_config.context,
                    parallel_config=llama_config.parallel)

    model_config = LlamaConfig(llama_config.model.model_config)
    model_config.use_past = True
    model_config.seq_length = 2048
    model_config.checkpoint_name_or_path = args.ckpt_dir

    tokenizer = LlamaTokenizer.from_pretrained("llama2_7b")
    model = LlamaForCausalLM(model_config)
    model.set_train(False)


    prob, gold = [], []

    for i in trange(0, len(dataset), args.eval_batch_size):
        cur_batch = dataset[i:i + args.eval_batch_size]
        inputs = tokenizer(cur_batch)
        for k in inputs:
            inputs[k] = inputs[k].cuda()
        logits, _, _ = model.construct(**inputs)
        cur_prob, cur_labels = preprocess_logits_for_metrics(logits, inputs['input_ids'])
        prob.append(cur_prob)
        gold.append(cur_labels)
    
    prob = mnp.concatenate(prob, axis=0)
    gold = mnp.concatenate(gold, axis=0)
    metrics = compute_metrics(prob, gold)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str)
    args = parser.parse_args()
    infer(args)
