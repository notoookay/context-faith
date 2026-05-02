## Setup

Use `uv` for package management.
```bash
uv sync
uv sync --extra compile # to install flash attention
```

## Running Scripts

To get the model generation without direction intervention, run:
```bash
bash scripts/model_generate.sh
```

To get the model generation with direction intervention, run:
```bash
bash scripts/apply_direction_generation.sh
```

Before getting the `difference-in-means` direction, we need to first get the model activations for the relevant and irrelevant passages.
```bash
bash scripts/get_model_activations.sh
```

Then, we can get the `difference-in-means` direction.
```bash
bash scripts/extract_diff_in_mean.sh
```

The context utility can be evaluated by running:
```bash
bash scripts/context_utility_evaluate.sh
```

The corpora can be downloaded from [DPR](https://github.com/facebookresearch/DPR) by:
```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

AACR can be evaluated by running:
```bash
bash scripts/eval_aacr.sh
```

## Data Format

The pipeline expects JSONL files where each line is one QA item paired with a single retrieved passage. The schema follows the [DPR](https://github.com/facebookresearch/DPR) retrieval output format:

```json
{
    "id": "nq_train_28074",
    "question": "when did colorado become part of the united states",
    "answers": ["August 1, 1876"],
    "ctxs": [
        {
            "id": "66490",
            "title": "Colorado",
            "text": "southwestern Colorado. The Ute people were removed ...",
            "score": "378.82068",
            "hasanswer": true
        }
    ]
}
```

Field reference:
- `id` (str): unique item identifier (we use `nq_train_*`, `nq_validation_*`, `tqa_*`, etc.).
- `question` (str): the user query.
- `answers` (list[str]): one or more gold answers (used for accuracy evaluation via exact-match / inclusion).
- `ctxs` (list[dict]): retrieved passages. Most scripts read `ctxs[0]` because each line carries a single relevant *or* irrelevant passage (see below). Each passage has:
    - `id` (str): passage id from the corpus.
    - `title` (str): passage title.
    - `text` (str): passage text (≤100 words in our setup, matching the DPR Wikipedia chunking).
    - `score` (str|float): retrieval score (informational; not consumed by the pipeline).
    - `hasanswer` (bool): `true` if the gold answer string appears in `text` (relevant passage), `false` otherwise (irrelevant passage). The activation-extraction and direction-evaluation scripts rely on this label to split items into the relevant (`D_rel`) and irrelevant (`D_irrel`) groups described in the paper.

Put your data under `data/`.
