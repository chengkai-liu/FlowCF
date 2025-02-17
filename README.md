# Flow Matching for Collaborative Filtering

> **Flow Matching for Collaborative Filtering**\
> Chengkai Liu, Yangtian Zhang, Jianling Wang, Rex Ying, James Caverlee\
> Paper: https://arxiv.org/abs/2502.07303

![framework](/Users/lck/Desktop/FlowCF/asset/framework.png)

## Requirements

* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+
* Install RecBole:
  * `pip install recbole`
  * After installing RecBole, you need to modify the `register_table` in the `get_dataloader` function in `recbole.data.utils`. Add new models' names (e.g. `FlowCF`) to `register_table`. There is a shortcut to modify `get_dataloader` in the first line of `run.py`. You can copy the provided [utils.py](./utils.py) file to the `recbole.data.utils` directory and modify the `get_dataloader` function within this copy. The key changes are in line 251 of the `get_dataloader` function. Make sure to implement these modifications correctly.
  


## Run

```python run.py```

Please config the model, dataset, and hyperparameters in `run.py` and  `flowcf.yaml` before running.

## Citation
```bibtex
@article{liu2025flow,
  title={Flow Matching for Collaborative Filtering},
  author={Liu, Chengkai and Zhang, Yangtian and Wang, Jianling and Ying, Rex and Caverlee, James},
  journal={arXiv preprint arXiv:2502.07303},
  year={2025}
}
```
