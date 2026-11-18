"""Debug: next-token probabilities after a forced digit, with cache reuse vs full re-encoding.

Usage: python code/debug_cache_scoring.py MODEL_ID INPUT_MODE SOURCE_SUBSTRING STATEMENT_TYPE
"""
import copy
import sys

import torch

import feed_experiment as fe

model_id, mode, source, statement_type = sys.argv[1:5]
runner = fe.resolve_runner(model_id, "auto")
statements = fe.load_statements(fe.STATEMENT_FILE)
jobs = fe.validate_inputs(fe.load_jsonl(fe.SEQUENCE_INDEX), statements, statement_types=[statement_type],
                          allow_missing_statements=False)
sequence, _, statement = next(job for job in jobs if source in job[0]["source_video"])
content = fe.sequence_content(sequence, statement, mode)
model, processor = fe.load_model(model_id, runner)
captured = {}
fe.score_distribution = lambda model, tokenizer, inputs: captured.setdefault("inputs", inputs)
fe.GENERATION["score_distribution"] = True
fe.run_one_sequence(model, processor, content, runner, 8, False)
inputs, tokenizer = captured["inputs"], processor.tokenizer
digit = {d: tokenizer.encode(d, add_special_tokens=False)[0] for d in "0123456789"}
settings = dict(max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True)


def top(logits, k=6):
    p = torch.softmax(logits.float(), -1)
    values, ids = p.topk(k)
    return [(tokenizer.decode([i]), round(v.item(), 4)) for v, i in zip(values, ids)]


with torch.inference_mode():
    first = model.generate(**inputs, **settings)
    print("first token:", top(first.logits[0][0]))
    seq_len = inputs["input_ids"].shape[-1]
    print("cache type:", type(first.past_key_values).__name__, "cache length:",
          first.past_key_values.get_seq_length(), "prompt length:", seq_len)
    for prefix in ("1", "10", "0"):
        forced = torch.tensor([[digit[c] for c in prefix]], device=model.device)
        full = {k: (torch.cat([v, (forced if k == "input_ids" else torch.ones_like(forced) if "attention" in k
                                   else torch.zeros_like(forced)).to(v.dtype)], -1)
                    if torch.is_tensor(v) and v.dim() == 2 and v.shape[-1] == seq_len else v)
                for k, v in inputs.items()}
        recompute = model.generate(**full, **settings)
        cached_inputs = {k: v for k, v in full.items() if torch.is_tensor(v) and v.dim() == 2}
        cached = model.generate(**cached_inputs, past_key_values=copy.deepcopy(first.past_key_values), **settings)
        print(f"after {prefix!r}: full re-encode {top(recompute.logits[0][0])}")
        print(f"after {prefix!r}: cache reuse    {top(cached.logits[0][0])}")
