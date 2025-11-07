<div align="center">

# Representation-Based Exploration for Language Models: <br> From Test-Time to Post-Training

[📄 arXiv](https://arxiv.org/abs/2510.11686) &nbsp; &nbsp; [🌐 Website](https://rep-exp.github.io) &nbsp; &nbsp; [🐦 Twitter / X ](https://x.com/JensTuyls/status/1978244454617128993)

</div>

## Installation 🔌

Our algorithm doesn't require anything beyond the base verl installation, which you can find [here](https://verl.readthedocs.io/en/latest/start/install.html).

## Running the Experiments 🚀

You can reproduce or extend our experiments by running the following commands:

```bash
# General format
sh recipe/rep_exp/train_elliptical.sh $TASK $SPARSE_DIM $BETA $SEED

# MATH
sh recipe/rep_exp/train_elliptical.sh math 32 0.01 42

# GSM8K
sh recipe/rep_exp/train_elliptical.sh gsm8k 32 0.01 42

# DAPO-WITH-AIME
sh recipe/rep_exp/train_elliptical.sh dapo-with-aime24 128 0.01 42
```
where `$TASK` is the task name, `$SPARSE_DIM` is the sparse dimension, `$BETA` is the beta parameter, and `$SEED` is the seed.

## Evaluation 📊



## Citation 📝

```bibtex
@article{tuyls2025representation,
  title={Representation-Based Exploration for Language Models: From Test-Time to Post-Training},
  author={Tuyls, Jens and Foster, Dylan J and Krishnamurthy, Akshay and Ash, Jordan T},
  journal={arXiv preprint arXiv:2510.11686},
  year={2025}
}
```

## Contact 📬

If you have any questions or suggestions, feel free to reach out to us at [jtuyls@princeton.edu](mailto:jtuyls@princeton.edu).