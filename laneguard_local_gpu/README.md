# Laneguard Local GPU Training

This folder contains standalone scripts for training the Roboflow `laneguard_s` dataset on a local NVIDIA GPU and plotting the resulting train/test curves.

## Files

- `run_train_100e.sh` launches 100-epoch RT-DETRv2 training on `cuda:0`.
- `plot_curves.py` builds a training/test graph from the trainer-written `log.txt`.

## Usage

```bash
cd /home/parvezdev/fydp/laneguard_local_gpu
bash run_train_100e.sh
python plot_curves.py
```

The scripts use:

- dataset: `/home/parvezdev/fydp/laneguard_s-3`
- output dir: `/home/parvezdev/fydp/outputs/laneguard_local_gpu_100e`
- model: RT-DETRv2 R18
- epoch log: `/home/parvezdev/fydp/outputs/laneguard_local_gpu_100e/log.txt`
