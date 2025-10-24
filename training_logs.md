
================================================================================
# ImageNet-1K Training Session Started: 2025-10-24 18:46:33
================================================================================

## ResNet-50 ImageNet-1K Training Configuration
✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]
✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)
✅ **MaxPool**: Included after initial conv (required for ImageNet)
✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8
✅ **Label Smoothing**: 0.1 (reduces overfitting)
✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs
   └─ Target: 83% top-1 accuracy in <150 epochs
Expected Impact: State-of-the-art ImageNet performance

usage: train.py [-h] [--dataset {test,full}] [--max-samples MAX_SAMPLES]
                [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--resume RESUME]
                [--cache-dir CACHE_DIR]

Train ResNet-50 on ImageNet-1K

options:
  -h, --help            show this help message and exit
  --dataset {test,full}
                        Dataset to use: test (10k train + 1k val) or full
                        (1.2M train + 50k val)
  --max-samples MAX_SAMPLES
                        Maximum number of samples to use (overrides dataset
                        choice)
  --batch-size BATCH_SIZE
                        Batch size (overrides default based on CUDA
                        availability)
  --epochs EPOCHS       Number of epochs to train
  --resume RESUME       Path to checkpoint to resume training from
  --cache-dir CACHE_DIR
                        Custom cache directory for dataset

================================================================================
# ImageNet-1K Training Session Started: 2025-10-24 18:46:57
================================================================================

## ResNet-50 ImageNet-1K Training Configuration
✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]
✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)
✅ **MaxPool**: Included after initial conv (required for ImageNet)
✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8
✅ **Label Smoothing**: 0.1 (reduces overfitting)
✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs
   └─ Target: 83% top-1 accuracy in <150 epochs
Expected Impact: State-of-the-art ImageNet performance


================================================================================
ResNet-50 Training Configuration
================================================================================
Dataset: Test (10k train + 1k val)
Max samples: 100
Epochs: 1
Batch size: Auto (64 CUDA, 32 CPU)
Cache directory: Auto
================================================================================

CUDA Available? False
Creating dataloaders...

================================================================================
Loading ImageNet-1K (train) from Hugging Face...
Dataset: ILSVRC/imagenet-1k
Cache directory: /teamspace/studios/this_studio/datasets/imagenet_test_cache
Mode: Test
Streaming mode: False
Max samples: 100
================================================================================

❌ Error loading ImageNet-1K dataset: Dataset 'ILSVRC/imagenet-1k' is a gated dataset on the Hub. You must be authenticated to access it.

================================================================================
# ImageNet-1K Training Session Started: 2025-10-24 18:55:15
================================================================================

## ResNet-50 ImageNet-1K Training Configuration
✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]
✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)
✅ **MaxPool**: Included after initial conv (required for ImageNet)
✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8
✅ **Label Smoothing**: 0.1 (reduces overfitting)
✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs
   └─ Target: 83% top-1 accuracy in <150 epochs
Expected Impact: State-of-the-art ImageNet performance


================================================================================
ResNet-50 Training Configuration
================================================================================
Dataset: Test (10k train + 1k val)
Max samples: All
Epochs: 150
Batch size: Auto (64 CUDA, 32 CPU)
Cache directory: Auto
================================================================================

CUDA Available? False
Creating dataloaders...

================================================================================
Loading ImageNet-1K (train) from Hugging Face...
Dataset: ILSVRC/imagenet-1k
Cache directory: /teamspace/studios/this_studio/datasets/imagenet_test_cache
Mode: Test
Streaming mode: False
Max samples: 10000
================================================================================


================================================================================
# ImageNet-1K Training Session Started: 2025-10-24 19:03:46
================================================================================

## ResNet-50 ImageNet-1K Training Configuration
✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]
✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)
✅ **MaxPool**: Included after initial conv (required for ImageNet)
✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8
✅ **Label Smoothing**: 0.1 (reduces overfitting)
✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs
   └─ Target: 83% top-1 accuracy in <150 epochs
Expected Impact: State-of-the-art ImageNet performance


================================================================================
ResNet-50 Training Configuration
================================================================================
Dataset: Test (10k train + 1k val)
Max samples: 100
Epochs: 1
Batch size: 8
Cache directory: Auto
================================================================================

CUDA Available? True
cuDNN benchmark enabled for maximum speed
Creating dataloaders...

================================================================================
Loading ImageNet-1K (train) from Hugging Face...
Dataset: ILSVRC/imagenet-1k
Cache directory: /teamspace/studios/this_studio/datasets/imagenet_test_cache
Mode: Test
Streaming mode: False
Max samples: 100
================================================================================

❌ Error loading ImageNet-1K dataset: [Errno 2] No such file or directory: '/teamspace/studios/this_studio/datasets/imagenet_test_cache/ILSVRC___imagenet-1k/default/0.0.0/49e2ee26f3810fb5a7536bbf732a7b07389a47b5.incomplete/dataset_info.json'

================================================================================
# ImageNet-1K Training Session Started: 2025-10-24 19:12:58
================================================================================

## ResNet-50 ImageNet-1K Training Configuration
✅ **Architecture**: ResNet-50 with Bottleneck blocks [3,4,6,3]
✅ **Dataset**: ImageNet-1K (1000 classes, 224x224 input)
✅ **MaxPool**: Included after initial conv (required for ImageNet)
✅ **Dropout**: 0.5 in final FC layer (standard for ImageNet)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **MixUp Augmentation**: Enabled with prob=0.3, alpha=0.8
✅ **Label Smoothing**: 0.1 (reduces overfitting)
✅ **Optimized LR Schedule**: OneCycleLR for 150 epochs
   └─ Target: 83% top-1 accuracy in <150 epochs
Expected Impact: State-of-the-art ImageNet performance


================================================================================
ResNet-50 Training Configuration
================================================================================
Dataset: Test (10k train + 1k val)
Max samples: 20
Epochs: 1
Batch size: 4
Cache directory: Auto
================================================================================

CUDA Available? True
cuDNN benchmark enabled for maximum speed
Creating dataloaders...

================================================================================
Loading ImageNet-1K (train) from Hugging Face...
Dataset: ILSVRC/imagenet-1k
Cache directory: /teamspace/studios/this_studio/datasets/imagenet_test_cache
Mode: Test
Streaming mode: False
Max samples: 20
================================================================================

