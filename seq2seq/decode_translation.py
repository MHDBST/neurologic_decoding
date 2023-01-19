import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import MarianTokenizer, MarianMTModel
# FSMTForConditionalGeneration, FSMTTokenizer, T5ForConditionalGeneration

from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
from unilm import utils_seq2seq
from lexical_constraints import init_batch

from generate import generate
#from zero_shot.generate_baseline import generate
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def my_encode(batch,tokenizer,input_len,EOS_ID,device):
    sample_output = []
    sample_mask = []
    shod=0
    for current_sent in batch:
        try:
            import nltk
            current_pieces = nltk.sent_tokenize(current_sent)
        except Exception as e:
            print(e)
        sample = tokenizer.batch_encode_plus(current_pieces, add_special_tokens=False)#,pad_to_max_length=True,padding='max_length',\
           
        current_input = sample['input_ids']
        used_input_temp = []
        used_input = ''
        idd = 0
        while True :
            used_input_temp += current_input[idd]
            if len(used_input_temp) >= input_len-5:
                break
            used_input = used_input_temp.copy()
            idd += 1
            if idd == len(current_input):
                shod += 1
                break
        if len(used_input) < 2:
            print('input sentence is long')
            continue
        padded_input = [21603,    10]+used_input+ [EOS_ID]+[0]*(input_len-len(used_input)-3)#+ EOS_ID#+[0]*(input_len-len(used_inputs)-3)

        input_attention_mask = [1] * (3+len(used_input)) + [0]*(input_len-len(used_input)-3) #+ [0]*(input_len-len(used_inputs)-3)
        assert len(padded_input) == len(input_attention_mask) == input_len
        
        sample_output.append(padded_input)
        
        sample_mask.append(input_attention_mask)

    torch.cuda.empty_cache()
    # exit()
    print('completed:', shod)
    print('all succedded:' , len(sample_output))
    return torch.tensor(sample_output).to(device),torch.tensor(sample_mask).to(device)
def generate_summaries_or_translations(
    args,
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    constraints_list=None,
    label=False,
    **gen_kwargs,
):

    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    print(f'Decode with {model_name}')
    print(f'device is {device}')
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    with torch.no_grad():
        model = MarianMTModel.from_pretrained(args.model_name).to(device)
    # tokenizer = FSMTTokenizer.from_pretrained(model_name)
    # model = FSMTForConditionalGeneration.from_pretrained(model_name).to(device)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    # if fp16:
    #     model = model.half()
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    # except:
    #     tokenizer = AutoTokenizer.from_pretrained('t5-large')
    print('tokenizer is:',tokenizer)
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    if "bart" in args.model_name.lower():
        period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    constraints_list = utils_seq2seq.tokenize_constraints(tokenizer, constraints_list)
    # update config with summarization specific params
    use_task_specific_params(model, task)
    # max_input_len=512
    for batch, cons in tqdm(zip(list(chunks(examples, batch_size)), list(chunks(constraints_list, batch_size)))):
        constraints = init_batch(raw_constraints=cons,
                                 beam_size=args.beam_size,
                                 eos_id=eos_ids)
        # print('constraints0',constraints[0])
        # print('constraints1',constraints[1])
        # exit()
        if "t5" in model_name:
            batch = ['generate a sentence with: ' + text + ' </s>' for text in batch]
            # batch = ['summarize: ' + text + ' </s>' for text in batch]
        # input_ids, attention_mask =my_encode(batch,tokenizer,max_input_len,tokenizer.eos_token_id,device)
        batch = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        # print(batch)
        input_ids, attention_mask = trim_batch(batch['input_ids'], pad_token_id=tokenizer.pad_token_id, \
            attention_mask=batch['attention_mask'])
        # print('input_ids in decode translation:',input_ids)
        # exit()

        # input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
        summaries = generate(self=model,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             decoder_start_token_id=tokenizer.bos_token_id,
                             min_length=args.min_tgt_length,
                             max_length=args.max_tgt_length,
                             num_beams=args.beam_size,
                             no_repeat_ngram_size=args.ngram_size,
                             length_penalty=args.length_penalty,
                             constraints=constraints,
                             prune_factor=args.prune_factor,
                             sat_tolerance=args.sat_tolerance,
                             beta=args.beta,
                             early_stop=args.early_stop,
                             tokenizer=tokenizer)
        sst = True
        if "bart" in args.model_name.lower():
            sst = True
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=sst, clean_up_tokenization_spaces=True)
        for hypothesis in dec:
            fout.write(hypothesis.strip() + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("--save_path", type=str, help="where to save summaries")
    parser.add_argument("--constraint_file", type=str, help="constraint file")

    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path", type=str, required=False, help="where to save the rouge score in json format")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization", help="typically translation or summarization")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument('--beam_size', type=int, default=10, help="Beam size for searching")
    parser.add_argument('--ngram_size', type=int, default=3, help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.1, help="length penalty for beam search")
    parser.add_argument('--min_tgt_length', type=int, default=0, help="minimum length of target sequence")
    parser.add_argument('--max_tgt_length', type=int, default=128, help="maximum length of target sequence")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument('--prune_factor', type=int, default=50, help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2, help="minimum satisfied clause of valid candidates")
    parser.add_argument('--beta', type=float, default=0., help="reward factor for in progress constraint")
    parser.add_argument('--early_stop', type=float, default=None, help="optional early stop if all constraints are satisfied")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--index', type=int, default=0, help="index to start processing")

    args = parser.parse_args()
    print('input path',args.input_path)
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]

    def lemma(x):
        if "bart" in args.model_name:
            return f' {x}'
        return x

    with open(args.constraint_file, 'r') as f:
        constraints_list = []
        for line in f:
            constraints = []
            for concept in json.loads(line):
                constraints.append([lemma(c) for c in concept])
            constraints_list.append(constraints)

    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    index=args.index
    examples=examples[index:]
    constraints_list=constraints_list[index:]
    generate_summaries_or_translations(
        args,
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        constraints_list=constraints_list,
        decoder_start_token_id=args.decoder_start_token_id,
    )
    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_bleu_score if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    reference_lns=reference_lns[index:]
    scores: dict = score_fn(output_lns, reference_lns)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return scores


if __name__ == "__main__":
    run_generate()
