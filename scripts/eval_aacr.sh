#! /bin/bash

uv run src/eval_aacr.py \
    --input_path data/exp-hallu/test_retrieved/triviaqa-dev.jsonl \
    --model_name gemma-2-2b-it \
    --prober_path output/binary_classifiers/gemma-2-2b-it-irrelevant__gemma-2-2b-it-relevant/binary_clf_gemma-2-2b-it-irrelevant__gemma-2-2b-it-relevant_seed_42/classifiers/layer_15_classifier.pt \
    --optimal_layer 15
