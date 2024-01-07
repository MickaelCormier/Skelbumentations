# Skelbumentations

Official Repository for the Paper "Enhancing Skeleton-Based Action Recognition in Real-World Scenarios through Realistic Data Augmentation" presented at 4th Real-World Surveillance: Applications and Challenges Workshop at WACV2024.

skelbumentations is a Python library for pose sequence augmentation. The architecture and code is based on [Albumentations](https://github.com/albumentations-team/albumentations)

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [A simple example](#a-simple-example)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Citing](#citing)

## Authors

[**Mickael Cormier** — Research Assistant at Fraunhofer IOSB](https://www.linkedin.com/in/mickaelcormier/)

[**Yannik Schmid**](https://www.linkedin.com/in/yannik-schmid/) 

## Installation

Skelbumentations requires Python 3.7 or higher.

To install the library on your system run the following command in the repository directory

```
pip install .
```

and import the library with

```python
import skelbumentations as S
```


## A simple example


```python
import skelbumentation as S

# Declare the augmentation pipeline
augment = S.Compose(
    [
        S.SelectFrames(
            [
                A.RandomOcclusion(chance=0.4)
            ], 
            frames=[3, 4, 5]
        ),
        S.SelectHighMovement(
            [
                A.WholeOcclusion()
            ], 
        ),
        S.SelectRandomFrames(
            [
                S.OneOf(
                    [
                        S.SwapPerturbation(),
                        S.MirrorPerturbation(opposite_points=opposite_points),
                    ]
                )
            ],
            min_num=5,
            max_num=10,
        ),
        S.MovePerturbation(variance=0.1),
    ]
)

# Get a keypoint sequence
keypoints = ...

# Augment the keypoint sequence
augmented = augment(keypoints=keypoints)
augmented_keypoints = augmented["keypoints"]
```

## Getting started

There a three different types of classes in this library:
- Compose: These classes are used to compose different transforms and selects. (`Compose, OneOf, SomeOf, OneOrOther, Sequential, NoOp`)
- Transform: These classes are used to apply perturbations or occlusions to the pose sequence. (`MovePerturbation, SwapPerturbation, MirrorPerturbation, RandomOcclusion, WholeOcclusion, SpecificOcclusion, InterpolateOcclusions`)
- Select: These classes are used to select certain frames from the pose sequence on which further transforms or compositions are applied. (`SelectRandomFrames, SelectFrames. SelectHighMovement, SelectRandomWithBorder`)

Each of the above classes feature a docstring explaining its function.

## Usage

An augmentation pipeline always starts with the `Compose` class. It accepts pose sequences as a named argument `keypoints=...`. The pose sequence has to be a numpy array with the format (T,V,C). With T being the timesteps/frames, V the keypoints and C the channels/dimensions. Furthermore it accepts the optional `invalid` argument which is a boolean map with the shape (T,V). It indicates wether a keypoint is invalid and therefore is ignored by Pertubations. The Compose returns a dict with both `keypoints` and `invalid` again. `keypoints` is the augmented skeleton sequence and `invalid` indicates occluded and therefore invalid keypoints. By default invalid keypoints are set to zero before being returned by `Compose`. This can be turned of with the `set_invalid_to_zero=False` argument during initialising. Then the user has to handle and modify invalid joints by himself with the help of the invalid map.

### Citing

```
@inproceedings{cormier2024skelbumentations,
  title={Enhancing Skeleton-Based Action Recognition in Real-World Scenarios through Realistic Data Augmentation},
  author={Cormier, Mickael and Schmid, Yannik and Beyerer, J{\"u}rgen},
  booktitle={2024 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  year={2024}
}
```
