#import "notes_template.typ": *

#show: dvdtyp.with(
  title: "Deep Learning Notes", subtitle: [], author: "Howard Nguyen-Huu",
)

#outline()

#pagebreak()

= Structure-aware Mamba-transformer hybrid model for hyperspectral image classification

- objective predict the category of each pixel in hyperspectral image
- this is not a segmentation problem, each pixel is a specific classification,
  each pixel has hundreds of chneels
- we split images in to test train

- mambe is a kind of SSM which

- in SSM, an input sequence $x(t) in RR$ is transformed into an output sequence $y(t) in RR$ through
  a state variable $h(t) in CC^N$ Here $t>0$ represents the time indesx and $N$ indicates
  the dimension of the state veriable

- zeor-order hold discrization is used
given a timescale $Delta$ which repres the interval betw discreet time steps

- real world proceccess often change over time and cannot be accurately describe
  by a LTI system
- in mamba they are made time aware

- why mambe?
  - linear time complexity,
    - this can lower comp. burden caused by hundreds of bands in HSI
  - efficient long-range dependency modelling

- why structure-aware state fusion?
  - while mamba performs well on sequential data, its original formulation is
    primarily designed for 1D dequences, appying it to Cd or 3d image data typically
    requires flattedning which disrupts the spatial structure of the image

  - many methods use different "paths" through the image

  - to address this, a novel structure arware state fusion is used
  - there are 2 kinds of fusion
    - spatial: dilated convolution
    - spectral: cov1d

- why mamba-tf hybrid model
- combine efficient dequence modeling capabilities with the tf's global attn

= Flow and Diffusion Models

== Lecture 1: Generative AI with SDEs

=== Introduction

- *Goal*: formalize what it means to generate something

- How do we represent images/videos/proteins as vectors:
  - images: $z in RR^(H times W times 3)$
  - videos: $z in RR^(T times H times W times 3)$
  - molecules: $z in RR^(N times 3)$

- What does it mean to successfully generate something?
  - we can frame the success of our generated sample as how likely it was that it
    came from our desired probability distribution
  - thus, we solve this problem by learning a data distribution from a dataset
  - then, generation will be as easy as sampling from our learned data distribution

- Data distribution: distr. of objs that we want to generate

- Probability density: $p_"data": RR^d arrow.r RR gt.eq 0$

- to train our algorithms, we need a dataset

- a dataset consists of a finite number of samples from the data distribution:
$
  z_1, ... z_N ~ p_"data"
$

- conditional generation means sampling the conditional data distr, this allows us
  to have some control of what we generate
$
  z~p_"data" (dot | y)
$

- a generative treis to generate samples from the distribution
- at first we have an initial distr, $p_"init"$ which is often $cal(N)(0, I_d)$
- a generative converts samples from an initial distr (eg gaussian) into samples
  from a data distribution, in this course this is done via Flow and Diffusion
  models

=== Flow Models

#definition("Trajectory")[
  A solution to an ODE is defined by a trajectory, a function with form:
  $
    X: [0, 1] arrow RR^d, t arrow.bar X_t
  $
]

#definition(
  "Vector Field",
)[
  Every ODE is defined by a vector field, $u$, i.e. a function of the form:
  $
    u: RR^d times [0, 1] arrow RR^d, (x, t) arrow.bar
  $
]

== VAE

=== VQ-VAE

- when using autoregressive models- if you are predicting discrete things you need
  a codebook

$
  L = log p(x|z_q(x)) + || "sg"[z_e (x)] - e||^2_2 + beta||
$

== GAN

- one of the issues with GAN loss is with the KL loss; when the distributions have
  no overlap the KL goes to infinity
- to solve this the Wasserstein loss was introduced, WGAN uses the optimal
  tansportation plan as the learning objective
- Wasserstein loss is kinda like moveing the means together instead of optimizing
  with SGD wrt KL divergence, but with Lipschitz constraints

- StyleGAN: able to control the generation and have very smooth head generation?
- they don't really pay attn to discriminator, but the generator they advance
- EG3D: uses a tri-plane?

= REPA-E

- instead of e2e diffusion+VAE training, guide the training with Dinov2
  supervision and partially train Diffusion
- traditionally, latent diffusion model 2 parts: train VAE, train diffusion on
  that
- if one were to train this e2e, this does not work at all, since it tries to
  learn more on the pixel space (local low level features) this makes it difficult
  to learn the latent features
- align with DINOv2 features while still training e2e
- they got better VAE performance when doing this as well

= diffuse and disperse: img generation w/ representation regularization

- proposes dispersive loss, gets rid of positive samples?

= control net

- zero convolutions

= LoRA

- basically train another branch which is added to a pretrained model

= AnimateDiff

- add a motion module to

= TokenFlow

- warping based

= CoDeF: content deformation fields for temporally consisten video processing

- instead of moving things

= LocLLM

- task: 2D human pose estimation -- input: image, output: keypoints
- motivation: LLMs for multimodal data LLaMA fine-tuned with Visual Instruction
  Tuning (Neurips 2023) -> LLaVA

