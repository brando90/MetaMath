import argparse
import json
import pdb
import jsonlines
from pathlib import Path
import datetime
import wandb
import fire
import os

import util
from vllm import LLM, SamplingParams
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

# def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
def test_hendrycks_math(
        model: str = 'mistralai/Mistral-7B-Instruct-v0.1', 
        # data_path: str = '~/putnam-math/data/Putnam_MATH_original_static2.jsonl', 
        data_path: str = '~/putnam-math/data/MATH_test.jsonl', 
        start=0, end=MAX_INT, 
        batch_size=5000, 
        tensor_parallel_size=1,
        mode: str = 'dryrun',  # 'dryrun' or 'online'
        # mode: str = 'online',  # 'dryrun' or 'online'
        ):
    # - Start wandb run
    path_2_eval_dataset = data_path
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    current_tmux_session = os.environ.get("TMUX", "").split(",")[-1]
    today = datetime.datetime.now().strftime('%Y-m%m-d%d-t%Hh_%Mm_%Ss')
    run_name = f'putnam-math ({today=} {model} {path_2_eval_dataset} {CUDA_VISIBLE_DEVICES=} {current_tmux_session=})'
    run = wandb.init(mode=mode, project="putnam-math", name=run_name, save_code=True)
    print(f'{run.url=}')
    output_dir = Path(f'~/data/results_{today}/').expanduser() 
    output_dir.mkdir(parents=True, exist_ok=True)

    # - Get eval data
    data_path: Path = Path(data_path).expanduser()
    print(f'{model=}\n, {data_path=}\n, {start=}\n, {end=}\n, {batch_size=}\n, {tensor_parallel_size=}\n')
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # temp_instr = problem_prompt.format(instruction=item["instruction"])
            temp_instr = problem_prompt.format(instruction=item["problem"])
            hendrycks_math_ins.append(temp_instr)
            # solution = item['output']
            solution = item['solution']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)

    # - End run
    prompt_template = prompt_template
    prompt_gen_func = 'meta_math_TODO_what_this_is_ignore_for_now_since_they_use_their_own_thing'
    wandb.config.update(dict(prompt_gen_func=str(prompt_gen_func), prompt_template=prompt_template, model=model, path_2_eval_dataset=path_2_eval_dataset, output_dir=output_dir))
    print(f'{wandb.config=}')
    run.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    import time
    start = time.time()
    # args = parse_args()
    # test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
     
    # test_hendrycks_math()  # python ~/MetaMath/eval_math.py
    
    fire.Fire(test_hendrycks_math) # python ~/MetaMath/eval_math.py 
    print(f"Done!\a Time: {time.time()-start:.2f} sec, {(time.time()-start)/60:.2f} min, {(time.time()-start)/3600:.2f} hr\a")
