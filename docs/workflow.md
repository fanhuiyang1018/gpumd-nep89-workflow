# Workflow Notes

## Logical flow

1. Compile and configure GPUMD and GPUMDkit.
2. Run a small GPUMD system with the base NEP89 potential to generate candidate structures.
3. Select representative structures using PCA or another dimensionality-reduction method.
4. Convert selected structures to DFT single-point input folders.
5. Assemble the completed DFT labels into `train.xyz` with GPUMDkit.
6. Run `nep` in prediction mode to evaluate the base NEP89 model.
7. Fine-tune the NEP model if the prediction error is unacceptable.
8. Use the fine-tuned potential for production GPUMD simulations.

## Important file roles

- `model.xyz`: initial atomic structure.
- `run.in`: GPUMD simulation input.
- `nep.in`: NEP prediction or training input.
- `nep.txt`: NEP potential file.
- `train.xyz`: labeled training or validation dataset.
- `dump.xyz`: sampled trajectory from GPUMD.
- `select_100.xyz`: representative structures selected for DFT labeling.
