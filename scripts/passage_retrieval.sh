#!/bin/bash
# NOTE: use faiss-gpu will take approximately 70GB GPU memory

uv run src/retrieval/passage_retrieval.py \
    --model_name_or_path facebook/dragon-plus-query-encoder \
    --passages corpora/wiki/enwiki-contriver-2018/psgs_w100.tsv \
    --passages_embeddings "corpora/wiki/enwiki-contriver-2018/dragon-embedding_enwiki-dec2018" \
    --data "data/train_10k.jsonl" \
    --n_docs 30 \
    --validate_retrieval \
    --per_gpu_batch_size 128 \
    --output_dir "data/retrieved"
