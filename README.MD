# Polymer Information Extraction

This repo contains code for the paper 'A general purpose material property extraction
pipeline from large polymer corpora using natural language processing'[[1]](https://www.nature.com/articles/s41524-023-01003-w).

## Requirements and Setup

- Python 3.7
- Pytorch (version 1.10.0)
- Transformers (version 4.17.0)

You can install all required Python packages using the provided environment.yml file using `conda env create -f environment.yml`

## Running the code

Example scripts and parameters for running training of the NER model is provided in the file run_ner.sh.

The script for fine-tuning of the masked language model can be run by using the following command:

```bash
python run_mlm.py \
    --model_name_or_path bert-base \
    --train_file /path/to/train/file \
    --do_train \
    --do_eval \
    --output_dir /output
```

Use python data_extraction.py to combine NER predictions using heuristic rules.

Please cite our paper if you use the code or data in this repo

```latex
@article{materialsbert,
  title={A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing},
  author={Shetty, Pranav and Rajan, Arunkumar Chitteth and Kuenneth, Chris and Gupta, Sonakshi and Panchumarti, Lakshmi Prerana and Holm, Lauren and Zhang, Chao and Ramprasad, Rampi},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={52},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## References

[1] Shetty, P., Rajan, A., Kuenneth, C., Gupta, S., Panchumarti, L., Holm, L., Zhang, C. & Ramprasad, R.  A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing. npj Computational Materials 9, 52 (2023) 
