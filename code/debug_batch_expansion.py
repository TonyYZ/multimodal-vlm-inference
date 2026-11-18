"""Debug: do num_return_sequences copies see the same input as a single sequence?

Compares first-token probabilities for one sequence with those of each of 3 sampled copies.
Usage: python code/debug_batch_expansion.py MODEL_ID INPUT_MODE SOURCE_SUBSTRING STATEMENT_TYPE
"""
import sys

import torch

import feed_experiment as fe

model_id, mode, source, statement_type = sys.argv[1:5]
runner = fe.resolve_runner(model_id, "auto")
jobs = fe.validate_inputs(fe.load_jsonl(fe.SEQUENCE_INDEX), fe.load_statements(fe.STATEMENT_FILE),
                          statement_types=[statement_type], allow_missing_statements=False)
sequence, _, statement = next(job for job in jobs if source in job[0]["source_video"])
content = fe.sequence_content(sequence, statement, mode)
model, processor = fe.load_model(model_id, runner)
captured = {}
fe.score_distribution = lambda model, tokenizer, inputs: captured.setdefault("inputs", inputs)
fe.GENERATION["score_distribution"] = True
fe.run_one_sequence(model, processor, content, runner, 8, False)
inputs, tokenizer = captured["inputs"], processor.tokenizer
print({k: tuple(v.shape) for k, v in inputs.items() if torch.is_tensor(v)})


def top(logits, k=5):
    p = torch.softmax(logits.float(), -1)
    values, ids = p.topk(k)
    return [(tokenizer.decode([i]), round(v.item(), 3)) for v, i in zip(values, ids)]


common = dict(max_new_tokens=1, output_logits=True, return_dict_in_generate=True, do_sample=True,
              temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
with torch.inference_mode():
    single = model.generate(**inputs, num_return_sequences=1, **common)
    batch = model.generate(**inputs, num_return_sequences=3, **common)
print("single  :", top(single.logits[0][0]))
for i in range(3):
    print(f"copy {i}  :", top(batch.logits[0][i]))
