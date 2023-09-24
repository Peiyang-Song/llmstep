from flask import Flask, request, jsonify

import argparse
import transformers
import torch
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--hf-model', type=str, default='kaiyuy/leandojo-lean4-tacgen-byt5-small')
parser.add_argument('--temperatures', type=float, nargs='+', default=[1.0])
parser.add_argument('--num-samples', type=int, default=8)
args = parser.parse_args()


def load_hf(hf_model):
    print("Loading model...")
    if 'kaiyuy/leandojo-lean4-tacgen-byt5-small' in hf_model:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.hf_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.hf_model)
    else:
        raise NotImplementedError(hf_model)

    # We evaludate everything on CPU in the real time.
    # if torch.cuda.is_available():
    #     model.cuda()
    model.eval()
    print("Done.")
    return model, tokenizer


model, tokenizer = load_hf(args.hf_model)
app = Flask(__name__)

@app.route('/', methods=['POST'])
def process_request():
    start = time.time()
    data = request.get_json()

    # For fair comparison, we do not rely on tactic prefix to simplify the task.
    # tactic_state = data.get('tactic_state')
    # prefix = data.get('prefix')

    state = data.get('tactic_state')
    tokenized_state = tokenizer(state, return_tensors="pt")

    # ReProver uses a different prompt format.
    # prompt = """[GOAL]%s[PROOFSTEP]%s""" % (tactic_state, prefix)

    # texts = generate(prompt, args.temperatures, args.num_samples)
    # texts = [prefix + text for text in texts]

    # response = {"suggestions": texts}
    # end = time.time()
    # print("%d suggestions (%.3fs)" % (len(texts), (end-start)))

    tactic_candidates_ids = model.generate(
        tokenized_state.input_ids,
        max_length=1024,
        num_beams=4,
        length_penalty=0.0,
        do_sample=False,
        num_return_sequences=4,
        early_stopping=False,
    )
    tactic_candidates = tokenizer.batch_decode(
        tactic_candidates_ids, skip_special_tokens=True
    )

    response = {"suggestions": tactic_candidates}
    end = time.time()
    print("%d suggestions (%.3fs)" % (len(tactic_candidates), (end-start)))

    return jsonify(response)


if __name__ == '__main__':
    port = os.environ.get('LLMSTEP_PORT', 5000)
    app.run(use_reloader=False, host='0.0.0.0', port=port)

# def generate(prompt, temperatures, num_samples):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
#     texts = []
#     for temp in temperatures:
#         out = model.generate(
#             input_ids,
#             max_new_tokens=128,
#             do_sample=temp > 0,
#             temperature=temp,
#             pad_token_id=tokenizer.eos_token_id,
#             num_return_sequences=num_samples if temp > 0 else 1
#         )
#         output_tokens = out[:, input_ids.shape[1]:]
#         texts.extend(tokenizer.batch_decode(
#             output_tokens,
#             skip_special_tokens=True
#         ))
#     texts = list(set(texts))
#     return texts
