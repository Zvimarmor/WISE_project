import os.path
import sys
import json
import argparse

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--data_type', required=True, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=3, type=int)
    parser.add_argument('--sequential_edit', action="store_true")
    parser.add_argument('--track_first_edit', action="store_true", help="Track the performance of the first edit throughout the sequential process.")
    parser.add_argument('--eval_steps', default=10, type=int, help="Frequency of evaluation steps during sequential editing.")
    parser.add_argument('--vanilla_generation', action="store_true", help="Use vanilla generation for evaluation.")
    parser.add_argument('--evaluation_type', type=str, default='', help="Custom evaluation type (e.g., generate-text).")

    args = parser.parse_args()

    # ... (Hparams selection remains same) ...
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size

    # ... (Data loading remains same) ...
    if args.data_type == 'ZsRE':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'temporal':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['ood_rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )
    
    metric_output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_FirstEditHistory.json'
    )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',
        'hallucination': 'ppl',
        'temporal': 'ood_ppl'
    }

    editor = BaseEditor.from_hparams(hparams)
    
    # Updated call with Unpacking
    metrics_bundle = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric=eval_metric[args.data_type],
        track_first_edit=args.track_first_edit,  # Pass the flag
        eval_steps=args.eval_steps,  # Pass the frequency
        test_generation=args.vanilla_generation # Pass the new flag
    )
    
    # Handle return values carefully
    if len(metrics_bundle) == 4:
         metrics, edited_model, weights_copy, first_edit_history = metrics_bundle
    else:
         metrics, edited_model, weights_copy = metrics_bundle[0], metrics_bundle[1], metrics_bundle[2]
         first_edit_history = []

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if args.track_first_edit:
        with open(metric_output_file, 'w') as f:
            json.dump(first_edit_history, f, indent=4)
        print("First Edit History saved at: ", metric_output_file)

    if len(metrics) > 0:
        summary_metrics(metrics)


