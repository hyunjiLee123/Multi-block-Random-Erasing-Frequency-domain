# Multi block + Random Erasing in the Frequency domain
===============================================================

### This code has the source code for the paper </br> "[멀티 블록 기반 Random Erasing in the Frequency Domain](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003125682)".

### Installation

Requirements for Pytorch （see [Pytorch](http://pytorch.org/) installation instructions）

### Examples:

#### CIFAR10

ResNet-20 baseline on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --p 0.5
    ```

ResNet-20 + REF on CIFAR10：
    ```
    python cifar.py --dataset cifar10 --arch resnet --depth 20 --p 0.0 --p2 0.5
    ```

#### CIFAR100

ResNet-20 baseline on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20 --p 0.5
    ```

ResNet-20 + REF on CIFAR100：
    ```
    python cifar.py --dataset cifar100 --arch resnet --depth 20 --p 0.0 --p2 0.5
    ```

#### Fashion-MNIST


ResNet-20 baseline on Fashion-MNIST：
    ```
    python fashionmnist.py --dataset fashionmnist --arch resnet --depth 20
    ```
    
ResNet-20 + Random Erasing on Fashion-MNIST：
    ```
    python fashionmnist.py --dataset fashionmnist --arch resnet --depth 20 --p 0.5
    ```


### Other architectures

For ResNet： 
    ```
    --arch resnet --depth (20， 32， 44， 56， 110)
    ```

For WRN：
    ```
    --arch wrn --depth 28 --widen-factor 10
    ```

===============================================================
#### Corruption data preparation

### CIFAR-10-C

Download CIFAR10-C dataset

https://paperswithcode.com/dataset/cifar-10c

### CIFAR-100-C

Download CIFAR100-C dataset

https://zenodo.org/records/3555552


### Evaluation
Clean Error and Corruption Error(Mean)

CIFAR10
    ```
    python cifar.py --resume [model_best.pth.tar path] --evaluate
    ```

CIFAR100
    ```
    python cifar.py --resume [model_best.pth.tar path] --evaluate --dataset cifar100
    ```

===============================================================

