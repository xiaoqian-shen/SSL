# contrastive learning

## Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning

[paper](http://arxiv.org/abs/2011.10043) | [code](https://github.com/zdaxie/PixPro) CVPR 2021

#### problem

current methods are trained only on instance-level pretext tasks

#### solution

add a Pixel-to-Propagation Module in pix-to-pix level

![image-20210912195859868](F:\paper\imgs\image-20210912195859868-16314479567121.png)

## Dual Contrastive Learning for Unsupervised Image-to-Image Translation

[paper](http://arxiv.org/abs/2104.07689) | [code](https://github.com/JunlinHan/DCLGAN)

#### problem

Cycle-consistency ensures that the translated images have similar texture information to the target domain, failing to perform geometry changes.

Cycle-consistency forces relationship between two domains to be a bijection, which is too constrained losing image diversity.

#### solution

BREAK THE CYCLE

Contrastive learning for unpaired image-to-image translation with a patch-based, multi-layer PatchNCE loss to maximize the mutual information

no reconstruction

![image-20210912093806141](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912093806141.png)

<img src="C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912094551436.png" alt="image-20210912094551436" style="zoom:67%;" />

SimDCL=DCLGAN+similarity loss, which is used to address mode collapse

## What Should Not Be Contrastive In Contrastive Learning

[paper](http://arxiv.org/abs/2008.05659) ICLR 2021

#### problem

existing SSL contrastive methods is invariant to specific transformation, but perform poorly if a downstream task requires discriminative ability to this variant. ex: colorization in augmentation, while required to distinguish color

#### solution

LEAVE ONE OUT

construct separate embedding sub-spaces sensitive to a specific augmentation while invariant to others, in order to learn general representation that can be applied to different downstream tasks

![image-20210912085558133](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912085558133.png)

Hence, each embedding sub-space is specialized to a single augmentation, and the shared layers will contain both augmentation-varying and invariant information.

training data: ImageNet

downstream: adopt linear classification protocol by training a supervised linear classifier on frozen features of feature space V for LooC, or concatenated feature spaces Z for LooC++ (iNaturalist 2019, CaltechUCSD Birds 2011,VGG Flowers, ImageNet)

![image-20210912090703624](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912090703624.png)

![image-20210912091126228](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912091126228.png)

## DetCo: Unsupervised Contrastive Learning for Object Detection

[paper](http://arxiv.org/abs/2102.04803) | [code](https://github.com/open-mmlab/OpenSelfSup) ICCV2021

#### problem

some methods only transfer well on object detection but sacrifice image classification performance

#### solution

assumption: image classification recognizes global instance from a single high-level feature map, while object detection recognizes local instance from multi-level feature pyramids

contrastive learning between global image and local patches

<img src="F:\paper\imgs\image-20210913083028664.png" alt="image-20210913083028664" style="zoom:67%;" />

performance on segmentation ImageNet

<img src="F:\paper\imgs\image-20210913083534364.png" alt="image-20210913083534364" style="zoom:67%;" />

## Debiased Contrastive Learning

[paper](http://arxiv.org/abs/2007.00224) | [code](https://github.com/chingyaoc/DCL) NeurIPS 2020

#### problem

sampling bias

negative samples are randomly sampled, which causes that some are actually have the same label as the anchor

#### solution

debiased contrastive loss

indirectly approximate the distribution of negative examples

证明过程略。。。只看结论

$$\tau^+$$ is a hyperparameter

![image-20210911222330290](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911222330290.png)

![image-20210911222359220](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911222359220.png)

result

CIFAR10, STL10

SimCLR (encoder: ResNet-50) optimizer: Adam; learning rate: 0.001

<img src="C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911231459544.png" alt="image-20210911231459544" style="zoom: 50%;" />

ImageNet-100: 

<img src="C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911224852609.png" alt="image-20210911224852609" style="zoom:50%;" />

## Self-Damaging Contrastive Learning

[paper](http://arxiv.org/abs/2106.02990) | [code](https: //github.com/VITA-Group/SDCLR) ICML 2021

#### problem

real data is imbalanced and shows a long-tail distribution, and contrastive learning is vulnerable to such data

deep models have difficulty in memorizing samples

#### solution

Based on the observation that network pruning, which usually removes the smallest magnitude weights, which is effective tool to spot forgotten samples (expose model's weakness)

Contrasting between pruned and non-pruned models will boost the small magnitude weight and lead to re-balancing

![image-20210912102143233](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912102143233.png)

two pruning stategy

+ magnitude pruning
+ random pruning

def magnitudePruning(self, magnitudePruneFraction, randomPruneFraction=0):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

```python
    # only support one time pruning
    self.reset()
    prunableTensors = []
    for name, module in self.model.named_modules():
        if hasattr(module, "prune_mask"):
            prunableTensors.append(module.prune_mask.detach())

    number_of_remaining_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors])).cpu().numpy()
    number_of_weights_to_prune_magnitude = np.ceil(magnitudePruneFraction * number_of_remaining_weights).astype(int)
    number_of_weights_to_prune_random = np.ceil(randomPruneFraction * number_of_remaining_weights).astype(int)
    random_prune_prob = number_of_weights_to_prune_random / (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

    # Create a vector of all the unpruned weights in the model.
    weight_vector = np.concatenate([v.flatten() for v in weights])
    threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

    # apply the mask
    for name, module in self.model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask = (torch.abs(module.weight) >= threshold).float()
            # random weights been pruned
            module.prune_mask[torch.rand_like(module.prune_mask) < random_prune_prob] = 0
```

result

long-tail datasets:

![image-20210912103251010](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912103251010.png)

and SDCLR' accuracy is even better in balanced CIFAR10/100 than SimCLR

## Hard Negative Mixing for Contrastive Learning

[paper](http://arxiv.org/abs/2010.01028) | [code](https://europe.naverlabs.com/research/code/) NeurIPS 2020

#### problem

More negative samples is not equivalent to effectiveness

Some easy negative samples are too far to contribute to the contrastive loss

#### solution

mix pairs of the hardest existing negatives

mix the hardest negatives with the query itself

![image-20210912110057487](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912110057487.png)

## Unsupervised Representation Learning by Invariance Propagation

[paper](http://arxiv.org/abs/2010.11694) NeurIPS 2020

#### problems

+ old ones: Most contrastive learning methods learn representation invariant to different views of the same instance

  ignore the relation between different instances, which are the key to reflect the global semantic

+ new ones: optimizing $$-log(P_{v_i}(N(i)))$$ tends to maximize some easy optimized similarities

  $$ v_i$$ should be similar to all discovered positive samples to capture abstract invariance effectively

#### solution

**hard positive samples**

positive samples with low similarities to the anchor sample

this method concentrates on maximizing the agreement between the anchor sample and its hard positive samples

**Invariance Propagation**

semantically consistent images are concentrated, while semantically inconsistent images are separated
$$
P_{v_i}(j)=\frac {{\rm exp} (\bar v_j \cdot v_i /\tau)} {\sum_{k=1}^n {\rm exp}(\bar v_k \cdot vi/ \tau)}
$$
using memory bank to update $${\bar v_i}$$ as an exponential moving average of $$v_i$$

**Positive Sample Discovery**

samples whose graph distances from vi are less or equal than l are added to positive sample set in each step

> smoothness assumption: if two points v1 and v2 in a high-density region are close, then their semantic information should be similar

although C is further, it relates more to A, while the nearer B is not the same category as A

![image-20210906065711542](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210906065711542.png)

<img src="C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210906073721446.png" alt="image-20210906073721446" style="zoom:80%;" />

Compared with KNN, this method uses Euclidean distance as the local metric and graph distance as the global metric

**Hard Sampling Strategy**

+ pos

  select P samples with the lowest similarity to construct the **hard positive sample** set $$ \mathcal N^h(i)$$

+ neg

  These hard positive samples deviate far from the anchor sample such that they provide more intra-class variations, which is beneficial to learn more abstract invariance

  M nearest neighbors of $$ v_i$$ as $$ \mathcal N_M(i)$$ and $$ \mathcal N_{neg}(i)=\mathcal N_M(i)-\mathcal N(i)$$ is the **hard negative sample** set

  set $$ B(i)=\mathcal N_{neg}(i) \cup \mathcal N^h(i)$$

  keep the ambiguous negative samples further

$$
\begin{align}
\mathcal L_{inv}(x_i)=& -log \, P_{v_i}(\mathcal N^h(i) | B(i))\\
=& -log \, \frac {{\sum_{p \in \mathcal N^h(i)} {\rm exp}(\bar v_p \cdot v_i/ \tau)}}
{{\sum_{n \in B(i)} {\rm exp}(\bar v_n \cdot v_i/ \tau)}}
\end{align}
$$

+ loss

  At the beginning, the discovered positive samples are not reliable due to the random network initialization
  $$
  \mathcal L_{ins}(x_i)=-log \, P_{v_i}(i | \mathcal N_M(i) \cup \{i\})
  $$

  $$
  \mathcal L(x_i) = \mathcal L_{ins}(x_i)+\lambda_{inv} \cdot \omega(t) \cdot \mathcal L_{inv}(x_i)\\
  w(t)=
  \begin{cases}
  0,& t \le T\\
  1,& t > T
  \end{cases}
  $$

  ![image-20210912105244274](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210912105244274.png)

# Few shot

## Recurrent Mask Refinement for Few-Shot Medical Image Segmentation

[paper](http://arxiv.org/abs/2108.00622) | [code](https://github.com/uci-cbcl/RP-Net)

difficult to generalize to unseen classes in medical image segmentation

The model is trained to distill knowledge about a semantic class from the support set (xs, ys) and then apply this knowledge to segment query set xq

![image-20210912232629522](F:\paper\imgs\image-20210912232629522.png)

context relation encoder to enhance context features and force the model to focus on the shape and context of the region of interest rather than pixels themselves

## Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer

[paper](http://arxiv.org/abs/2108.03032) | [code](https://github.com/zhiheLu/CWTfor-FSS)

#### Problems

existing semantic segmentation methods have poor scalability to new classes, and general applicability is hindered

existing methods typically meta-learn the entire model after the encoder is pre-trained on ImageNet. Once trained, given a new class with annotated support set images and query images for test, the model is expected to adapt all three parts to the new class. With only few annotated support set images and the complex and interconnected three parts, this adaptation is often sub-optimal.

#### Methods

<img src="F:\paper\imgs\image-20210908145019427.png" alt="image-20210908145019427" style="zoom:150%;" />

episodic training with two loops

1. the support set is used to construct a classifier
2. the query set is utilized to adapt the classifier with a Classifier Weight Transformer

#### stage 1

pre-training to learn feature representations

#### stage2

Assumption: after seeing sufficient diverse classes, it can work well when task is to separate foreground and background pixels

given an episode we froze the encoder and decoder, initialize the classifier on the support set, and meta-learn a Classifier Weight Transformer (CWT) to update the classifier for each query image.

project the classifier weights and query feature to a $$d_a{\rm -D}$$ latent space

classifier-to-query-image attention mechanism is to adapt the classifier weights to the query image

<img src="F:\paper\imgs\image-20210908144724354.png" alt="image-20210908144724354" style="zoom: 67%;" />

Residual connection is for more stable model convergence

This attentive learning would reinforce this desired proximity and adjust

![image-20210912232201895](F:\paper\imgs\image-20210912232201895.png)

## Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning

[paper](http://arxiv.org/abs/2003.12862) | [code](https://github.com/TAMU-VITA/ Adv-SS-Pretraining) CVPR2020

#### problem

gaining robustness from pretraining is left unexplored in SSL

robust training methods have higher sample complexity

#### extensive experiments to answer some questions

robust pretrained models leveraged for adversarial fine-tuning result in a large performance gain

robust pretraining mainly speeds up adversarial finetuning

pretrained models resulting from different self-supervised tasks have diverse adversarial vulnerabilities

a more significant robustness improvement is obtained by adversarial fine-tuning

and full fine-tuning outperforms that for partial fine-tuning

![image-20210912231236059](F:\paper\imgs\image-20210912231236059.png)

# adversarial

## Adversarial Self-Supervised Contrastive Learning

[paper](http://arxiv.org/abs/2006.07589) | [code](https://github.com/Kim-Minseon/RoCL) NeurIPS 2020

#### problem

adversarial learning use class labels to generate adversarial samples that lead to incorrect predictions, in order to augment robustness

#### solution

generate adversarial samples by perturbing augmented data

utilize contrastive learning to maximize the similarity between augmented samples and their adversarial perturbations

like this:<img src="C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911230144900.png" alt="image-20210911230144900" style="zoom:50%;" />

and this: ![image-20210911230309811](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911230309811.png)

result: CIFAR10

white-box attack

![image-20210911231951425](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911231951425.png)

black-box attack

![image-20210911232252654](C:\Users\26952\AppData\Roaming\Typora\typora-user-images\image-20210911232252654.png)

# Others

## Labels4Free: Unsupervised Segmentation using StyleGAN

[paper](http://arxiv.org/abs/2103.14968)  | [code](https:/rameenabdal.github.io/Labels4Free) CVPR

To generate segmentation masks that enable foreground and background separation

without supervision for ground truth masks

<img src="F:\paper\imgs\image-20210912225703955.png" alt="image-20210912225703955" style="zoom:67%;" />

generate background and foreground images from different branches, and then mix them up and send it into discriminator D

Alpha network is used to learn the alpha mask for the foreground
