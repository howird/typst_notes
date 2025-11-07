#import "../styles/things.typ": challenge, hypothesis, question

= Normalizing Flows are Capable RL Models

== Overview

An overview of "Normalizing Flows are Capable Models for RL" by Ghugare and
Eysenbach (2025).

=== Challenges and Approaches

#challenge[
  Modern Reinforcement Learning (RL) algorithms often rely on powerful but
  complex probabilistic models like diffusion models or transformers, which can
  be computationally expensive, have high complexity, or are limited in their
  application (e.g., to discrete spaces).
][
  - *Alternative Solutions:*
    - *Diffusion Models:* Highly expressive but computationally intensive as
      they require solving differential equations for training, sampling, and
      likelihood evaluation.
    - *Autoregressive Transformers:* Scalable and expressive but typically
      require learning discrete representations for use in continuous spaces.
    - *Energy-Based Models (EBMs):* Offer unnormalized likelihoods but require
      computationally intensive Markov Chain Monte Carlo (MCMC) for sampling.
    - *Variational Autoencoders (VAEs):* Provide a lower bound on the likelihood
      and support sampling but are not suitable for direct variational
      inference.

  #hypothesis[
    The prevailing belief that Normalizing Flows (NFs) lack sufficient
    expressivity for complex RL tasks is a misconception. NFs can match the
    performance of more complex models while being simpler and more efficient.
  ]

  - *Proposed Approach:* Demonstrate that a single, simple NF architecture can
    seamlessly integrate into various RL algorithms, serving as a policy,
    Q-function, or occupancy measure. This is achieved by leveraging the key
    capabilities of NFs: exact likelihood computation, efficient sampling, and
    compatibility with both Maximum Likelihood Estimation (MLE) and Variational
    Inference (VI).
]

#challenge[
  In imitation learning (IL), achieving high performance often requires complex
  models like diffusion policies or transformers, which come with a high
  computational cost and many hyperparameters.
][
  #hypothesis[
    The extra compute and complexity of state-of-the-art generative models might
    not be necessary for high performance in IL. A simpler, expressive model
    like an NF should suffice.
  ]

  - *Proposed Approach:* Apply a simple NF architecture to Behavior Cloning
    (BC), termed NF-BC, and compare its performance, simplicity (fewer
    hyperparameters), and parameter count against complex baselines like
    Diffusion Policy and VQ-BeT.
]

#challenge[
  In offline Goal-Conditioned RL (GCRL), it is often argued that learning a
  value function is essential for strong performance, leading to more complex
  offline RL algorithms.
][
  #hypothesis[
    A sufficiently expressive policy model, like an NF, can often achieve strong
    performance in conditional imitation learning without needing an explicitly
    learned value function.
  ]

  - *Proposed Approach:* Use the NF architecture within a simple
    Goal-Conditioned Behavior Cloning (GCBC) framework (NF-GCBC). This approach
    relies only on MLE and is compared against both other BC methods and more
    complex offline RL algorithms that learn a value function.
]

#challenge[
  Integrating expressive policies into offline RL algorithms often requires
  auxiliary techniques like distillation or importance resampling, which adds
  complexity.
][
  #hypothesis[
    NF policies can be directly integrated with standard offline RL algorithms
    to achieve high performance without extra machinery, due to their inherent
    ability to be trained via both VI and MLE.
  ]

  - *Proposed Approach:* Integrate the NF policy directly into a minimal
    actor-critic algorithm (MaxEnt RL + BC). This method, NF-RLBC, allows for
    direct gradient backpropagation through the policy to optimize actions based
    on the Q-function, a capability not easily available in other expressive
    models.
]

#challenge[
  In unsupervised goal sampling (UGS), effective exploration requires estimating
  the density of the agent's goal coverage, which is often done with separate
  models or complex objectives.
][
  #hypothesis[
    A single NF model can be used to efficiently estimate both the
    goal-conditioned value function (Q-function) and the marginal goal coverage
    density, enabling better exploration.
  ]

  - *Proposed Approach:* Use one NF network to model both the conditional
    distribution $p(g|s, a)$ (the Q-function) and the marginal distribution
    $p(g)$ (the coverage density) using a masking scheme. This single model then
    supports a canonical UGS algorithm for exploration.
]

=== Proposed Component: A Simple Normalizing Flow Architecture

- *Description:* The paper proposes a single NF architecture that is used across
  all RL settings. It is built by stacking multiple identical blocks, where each
  block consists of two main components adapted from prior work: a *coupling
  network* and a *linear flow*.
  - The *coupling network* splits the input, applies an affine transformation to
    one half conditioned on the other, and is easily invertible. The paper uses
    fully connected networks with LayerNorm.
  - The *linear flow* applies an invertible linear transformation (generalized
    permutation) to the output of the coupling layer to allow all dimensions to
    influence each other.
- *Inputs:*
  - A data point $x in RR^d$ to be transformed (e.g., an action).
  - Optional conditioning information $y in RR^k$ (e.g., a state or state-goal
    pair).
- *Outputs:*
  - For the forward pass $f_theta(x)$, a sample $z$ from a simple prior
    distribution (e.g., Gaussian) and the log-determinant of the Jacobian, which
    together give the exact log-likelihood $log~p_theta(x|y)$.
  - For the inverse pass $f_theta^(-1)(z)$, a sample $x$ from the learned
    distribution $p_theta(x|y)$ is generated by transforming a sample $z$ from
    the prior.

=== Dependencies for Reproducibility

- *Environments/Tasks:*
  - *Imitation Learning:* PushT, UR3 BlockPush, Kitchen, Multimodal Ant.
  - *Conditional Imitation Learning:* 45 tasks from OGBench.
  - *Offline RL:* 30 tasks from prior work , including `puzzle-3x3-play`,
    `scene-play`, `antmaze-large-nav`, `cube-single-play`,
    `humanoidmaze-medium-nav`, and `antsoccer-arena-nav`.
  - *Goal-Conditioned RL:* `ant-u-maze`, `ant-big-maze`, `ant-hard-maze`.
- *Datasets:*
  - The experiments rely on pre-existing datasets associated with the benchmarks
    listed above, such as the D4RL datasets for offline RL.
- *Algorithms/Models (used as baselines):*
  - *BC Baselines:* Diffusion Policy (DiffPolicy-C, DiffPolicy-T) , VQ-BeT ,
    BeT.
  - *GCBC Baselines:* Standard GCBC with a Gaussian policy, FM-GCBC
    (Flow-Matching GCBC) , GCIQL , QRL , CRL.
  - *Offline RL Baselines:* BC with a Gaussian policy, ReBRAC , IDQL , IFQL ,
    FQL.
  - *GCRL Baselines:* CRL-oracle, CRL-uniform, CRL-minmax.
- *Architectural Components (from prior work):*
  - Coupling layers from RealNVP.
  - Linear flows (invertible 1x1 convolutions / generalized permutations) from
    GLOW.
  - Noising/denoising technique for MLE training from Zhai et al. (2024) .

=== Missing Perspectives from Abstract

- The abstract mentions NFs serve as a "policy, Q-function, and occupancy
  measure" but does not detail that a *single* NF model can perform multiple
  roles simultaneously (e.g., Q-function and occupancy measure via masking) as
  shown in the UGS experiments.
- The abstract highlights simplicity and performance but omits the key technical
  reason for this simplicity: NFs' native support for both MLE and VI allows
  them to be plugged into existing RL algorithms *without* the complex auxiliary
  machinery (like distillation or resampling) required by other expressive
  models.

=== Assumptions

- *Markov Decision Process (MDP):* The underlying problems are modeled as
  controlled Markov processes.
- *Infinite Horizon:* The GCRL and some offline RL formulations assume an
  infinite horizon setting ($H=infinity$) with discounted returns.
- *Universal Approximator:* The paper implicitly assumes that the chosen NF
  architecture is sufficiently expressive to be considered a universal density
  approximator for the tasks at hand, a property discussed in the broader NF
  literature.

== Problem Formulation

Of course. Here is a detailed outline of the problem formulation and the
implementation pipeline for the project described in the paper.

== Problem Formulation

The paper frames various reinforcement learning problems as probabilistic
modeling tasks, solvable with either *Maximum Likelihood Estimation (MLE)* or
*Variational Inference (VI)*. The core proposal is to use Normalizing Flows
(NFs) as a unified, expressive, and efficient model for these tasks.

=== 1. Normalizing Flows (NFs)

An NF learns an invertible, differentiable mapping $f_theta: RR^d -> RR^d$
between a complex data distribution $p(x)$ and a simple base (prior)
distribution $p_0(z)$ (e.g., a standard Gaussian).

- *Density Estimation:* The probability density of a data point $x$ is
  calculated using the change of variables formula:

$
  p_theta (x) = p_0 (f_theta (x)) lr(|det((diff f_theta (x))/(diff x))|) quad(E q . 3)
$

where $(partial f_theta(x))/( partial x )$ is the Jacobian of the
transformation. The architecture is designed such that the determinant of the
Jacobian is computationally efficient to calculate.

- *Sampling:* To generate a new sample $x$ from the learned distribution
  $p_theta(x)$, one samples a latent variable $z$ from the simple prior $p_0(z)$
  and applies the inverse transformation:

$
  x = f_theta^(-1)(z) quad(E q . 4)
$

=== 2. Training Objectives

NFs can be trained using two primary objectives, making them highly flexible.

- *Maximum Likelihood Estimation (MLE):* Given a dataset $cal(D)$ of samples
  from the true data distribution, MLE finds the parameters $theta$ that
  maximize the log-likelihood of the observed data.

$
  theta_"MLE" = arg max_theta EE_(x ~ cal(D)) [log p_theta (x)]
$

Substituting the NF density (Eq. 3) gives the practical objective:

$
  theta_"MLE" = arg max_theta EE_(x ~ cal(D)) [log p_0 (f_theta (x)) + log lr(|det((diff f_theta (x))/(diff x))|)] quad(E q . 5)
$

- *Variational Inference (VI):* When the target density $p(x)$ is only known up
  to a constant (i.e., we have an unnormalized density $tilde(p)(x)$), VI
  approximates $p(x)$ by minimizing the KL-divergence with a distribution
  $q_theta(x)$ from our model family.

$
  theta_"VI" = arg max_(q_theta in cal(Q)) EE_(x ~ q_theta (x)) [log tilde(p)(x) - log q_theta (x)]
$

For NFs, this is optimized by sampling from the base distribution
($x = f_theta^-1(z)$ where $z ~ p_0(z)$) and re-writing the objective as:

$
  theta_"VI" = arg max_theta EE_(z ~ p_0 (z)) [log tilde(p)(f_theta^(-1)(z)) - log p_0 (z) + log lr(|det((diff f_theta^(-1)(z))/(diff z))|)] quad(E q . 6)
$

=== 3. RL as Probabilistic Inference

The paper applies this framework to several RL settings:

- *Offline Imitation Learning (IL):* The goal is to learn a policy
  $pi_theta(a_t|s_t)$ that mimics expert behavior from a static dataset
  $cal(D)$. This is a direct application of *MLE*.
  - *Behavior Cloning (BC):*

$
  arg max_theta EE_((s_t, a_t) ~ cal(D)) [log pi_theta (a_t |s_t)] quad(E q . 7)
$

- *Goal-Conditioned BC (GCBC):*

$
  arg max_theta EE_(s_t, a_t) ~ cal(D), g ~ p(s_(t' > t)|s_t) [log pi_theta (a_t |s_t, g)] quad(E q . 8)
$

- *Offline RL:* The goal is to learn a policy that maximizes returns from a
  static dataset. The policy update combines *VI* (to maximize the Q-function)
  and *MLE* (to regularize towards the dataset policy).
  - *Q-Function (Critic) Update (TD-Learning):*

$
  arg min_(Q_phi.alt) EE_((s_t, a_t, r_t, s_(t + 1)) ~ cal(D)) [(Q_phi.alt (s_t, a_t) -(r_t + gamma Q_macron(phi.alt) (s_(t + 1), a_(t + 1))))^2 ] quad(E q . 9)
$

- *Policy (Actor) Update:*

$
  arg max_theta underbrace(
    EE_(s_t ~ cal(D), a_t ~ pi_theta(dot|s_t)) [Q_phi(s_t, a_t) - lambda log pi_theta(a_t|s_t)], "VI Part",
  ) + underbrace(alpha EE_((s_t, a_t) ~ cal(D))[log pi_theta(a_t|s_t)], "MLE Part") quad (E q. 10)
$

- *Goal-Conditioned RL (GCRL):*
  - *Q-Function Estimation:* Learning the Q-function $Q(s, a, g)$ is framed as
    *MLE*, where the Q-value is equivalent to the discounted future state
    occupancy probability $p(g|s,a)$.

$
  arg max_theta EE_( (s_t, a_t) ~ cal(D), g ~ p(s_(t' > t)|s_t, a_t) ) [log p_theta (g|s_t, a_t)] quad(E q . 11)
$

- *Unsupervised Goal Sampling (UGS):* Effective exploration requires modeling
  the marginal density of reachable goals $p(g)$, which is also an *MLE*
  problem.

$
  arg max_theta EE_( g ~ cal(D) )[log p_theta (g)] quad (E q. 12)
$

== Pipeline

This pipeline outlines how the proposed NF architecture is implemented and
trained for the various RL tasks.

=== Stage 1: Model Definition
This stage defines the core NF architecture.

- *Description:* The model $f_theta$ is constructed by stacking $T$ identical
  blocks. Each block performs an invertible transformation. A separate MLP
  encoder processes the conditioning variable $y$.
  1. *Conditioning Encoder:* An MLP encodes the conditioning variable $y$ (e.g.,
    state $s_t$) into an embedding.
  2. *NF Block:* For each block $t=1...T$:
    - The input $x^t-1$ is split into two halves: $x_1^t-1, x_2^t-1$.
    - A *Coupling Network* (an MLP) takes $x_1^t-1$ and the encoded condition
      $y$ as input and outputs a scale $S$ and shift $A$.
    - An affine transformation is applied:
      $tilde(x)_2^t = (x_2^t-1 + A) dot.circle exp(-S)$.
    - The halves are concatenated:
      $tilde(x)^t = "concat"(x_1^t-1, tilde(x)_2^t)$.
    - A *Linear Flow* applies an invertible linear transformation (PLU
      decomposition) to $tilde(x)^t$ to produce the block's output $x^t$. This
      ensures information mixing between the two halves across blocks.
- *Inputs:*
  - `x`: The data to be transformed (e.g., actions). Tensor shape:
    `(B, D_action)`.
  - `y`: The conditioning variable (e.g., states). Tensor shape: `(B, D_state)`.
  - `B` is batch size.
- *Outputs:*
  - `z`: The transformed data in the latent space. Tensor shape:
    `(B, D_action)`.
  - `log_det_J`: The log-determinant of the Jacobian. Tensor shape: `(B,)`.

=== Stage 2: Data Preparation
This stage samples and prepares batches from the offline dataset $cal(D)$.

- *Description:* A data loader samples mini-batches of transitions
  $(s_t, a_t, r_t, s_t+1, dots)$ from the replay buffer. For goal-conditioned
  tasks, a future state $g$ from the same trajectory is also sampled.
- *Input:* The full offline dataset $cal(D)$.
- *Output:* A mini-batch of tensors.
  - `states`: `(B, D_state)`
  - `actions`: `(B, D_action)`
  - `goals` (if applicable): `(B, D_state)`

=== Stage 3: Training Loop
This stage performs the main parameter updates. The specific loss function
depends on the task.

==== Case A: Imitation Learning (NF-BC / NF-GCBC)
- *Description:* This loop uses pure MLE.
  1. A batch of `(states, actions)` or `(states, goals, actions)` is sampled.
  2. The actions are passed as `x` and states/goals as the condition `y` to the
    NF model to get the latent variable `z` and `log_det_J`.
  3. The log-likelihood of the prior is computed: `log_p_z = log(p_0(z))`.
  4. The total log-likelihood is calculated: `log_prob = log_p_z + log_det_J`.
  5. The loss is the negative mean of `log_prob` across the batch.
  6. The model parameters $theta$ are updated via backpropagation.
- *Equation Reference:* Optimizes objectives from *Eq. (7)* or *Eq. (8)* using
  the NF density from *Eq. (3)*.
- *Input:* A data batch, NF model $f_theta$.
- *Output:* Updated model parameters $theta$.

==== Case B: Offline Reinforcement Learning (NF-RLBC)
- *Description:* This is an actor-critic loop.
  1. *Critic Update:*
    - A batch of `(states, actions, rewards, next_states)` is sampled.
    - The critic $Q_phi$ is updated by minimizing the Bellman error against a
      target critic $Q_overline(phi)$, as per *Eq. (9)*.
  2. *Actor Update:*
    - *VI Term:* Sample actions $a^pi_t$ from the policy for the batch states
      $s_t$. This is done by sampling $z ~ p_0(z)$ and computing
      $a^pi_t = f_theta^-1(z | s_t)$ using the inverse pass of the NF (*Eq. 4*).
      Calculate the policy's log probability $log pi_theta(a^pi_t|s_t)$. The VI
      loss term is $-Q_phi(s_t, a^pi_t)$.
    - *MLE Term:* Calculate the log probability $log pi_theta(a_t|s_t)$ for the
      actions $a_t$ from the dataset. The MLE loss term is
      $-log pi_theta(a_t|s_t)$.
    - The total actor loss is a weighted sum of the VI and MLE losses.
    - The policy parameters $theta$ are updated via backpropagation.
- *Equation Reference:* The critic update uses *Eq. (9)*. The actor update uses
  *Eq. (10)*.
- *Input:* A data batch, NF policy $pi_theta$, Q-network $Q_phi$.
- *Output:* Updated parameters $theta$ and $phi$.

=== Stage 4: Action Sampling (Evaluation)
This stage describes how to use the trained policy to interact with an
environment.

- *Description:* To generate an action for a given state $s_t$:
  1. Sample a random vector $z$ from the base distribution (e.g.,
    `z = torch.randn(1, D_action)`).
  2. Pass $z$ and the conditioning state $s_t$ to the inverse of the NF model,
    $f_theta^-1$.
  3. The output is the action to be executed.
- *Equation Reference:* This is a direct application of *Eq. (4)*.
- *Input:*
  - Current state `s_t`: `(1, D_state)`.
  - Trained NF policy $pi_theta$.
- *Output:*
  - Action `a_t`: `(1, D_action)`.

== Discussion

Based on the paper "Normalizing Flows are Capable Models for RL", here is a
detailed outline of the main questions investigated in the results and the
limitations discussed.

#question[
  Is the high computational cost of complex generative models necessary for
  imitation learning?
][
  This question challenges the assumption that highly complex and
  computationally expensive models, like diffusion policies, are required to
  achieve state-of-the-art performance in imitation learning (IL).
][
  - *Experiments:*
    - They proposed *NF-BC*, a simple behavior cloning (BC) algorithm
      implemented with their Normalizing Flow (NF) architecture.
    - This was evaluated on four multi-modal robotics tasks requiring
      fine-grained control: *PushT, UR3 BlockPush, Kitchen, and Multimodal Ant*.
    - The baselines were state-of-the-art generative models: *DiffPolicy-C*
      (convolutional diffusion), *DiffPolicy-T* (transformer diffusion),
      *VQ-BeT* (transformer with VQ-VAE), and *BeT* (transformer with k-means).

  - *Results:*
    - NF-BC *outperformed all baselines on 2 of the 4 tasks* (PushT and Kitchen)
      and performed competitively on the other two.
    - Crucially, NF-BC was the *simplest method*, using 2 to 3 times fewer
      hyperparameters than baselines like VQ-BeT and DiffPolicy.
    - It achieved these results with a significantly smaller model, using only a
      *fraction of the parameters* of the convolutional diffusion policy
      (DiffPolicy-C).

  - *Metrics Used:*
    - *Performance:* Measured by normalized rewards. For most tasks, this
      corresponds to the number of distinct expert behaviors the agent
      successfully imitates.
    - *Simplicity:* Measured by the inverse of the number of hyperparameters,
      with fewer being better.
    - *Efficiency:* The total number of model parameters

  - *Significance of Results:*
    - These results directly *challenge the notion that NFs lack expressivity*
      for complex IL tasks.
    - They demonstrate that the superior performance of modern generative models
      may not be worth the trade-off in computational cost and complexity, as a
      simpler NF can be competitive or even better. This is important for
      practitioners, as simpler models are easier to implement and require less
      hyperparameter tuning.

  - *Limitations:*
    - The authors note that the performance results for the baseline methods
      were taken from a prior publication that did not include error bars, so a
      direct statistical comparison is not possible.
]

#question[
  In conditional imitation learning, can an expressive policy alone suffice
  without a value function?
][
  This question investigates the common belief in offline Goal-Conditioned
  Reinforcement Learning (GCRL) that learning a value function is necessary for
  strong performance.
][
  - *Experiments:*
    - They proposed *NF-GCBC*, which applies their NF architecture to the
      Goal-Conditioned Behavior Cloning algorithm, a simple approach that does
      not learn a value function.
    - The evaluation was performed on *45 tasks* from the OGBench benchmark,
      designed to test a wide range of capabilities like long-horizon reasoning
      and learning from suboptimal data.
    - Baselines included both other BC-based methods (*GCBC* with a Gaussian
      policy, *FM-GCBC* with a flow-matching policy) and dedicated offline RL
      algorithms that learn a value function (*GCIQL, QRL, CRL*).

  - *Results:*
    - NF-GCBC *significantly outperformed the standard GCBC* (with a Gaussian
      policy) on 40 out of 45 tasks, highlighting the benefit of an expressive
      policy.
    - Despite being a simple BC-based algorithm, NF-GCBC *outperformed all
      baselines on aggregate*, including the more complex offline RL algorithms
      that learn a value function.
    - It performed *77% better* than FM-GCBC, the most similar baseline which
      also uses an expressive policy model (flow matching).

  - *Metrics Used:*
    - *Performance:* The primary metric was the average success rate of reaching
      a goal over 50 independent trials. Results were presented both as
      improvement over the standard GCBC baseline and as aggregated performance
      across all 45 tasks.

  - *Significance of Results:*
    - The results suggest that for many conditional IL tasks, a *sufficiently
      expressive policy model can be enough* to achieve high performance,
      potentially obviating the need for a more complex value-based offline RL
      algorithm.
    - The authors hypothesize that NFs perform better than similar models like
      flow-matching in this context because they *directly optimize the
      log-likelihood*, which may be more effective with simple architectures
      compared to the indirect probability estimation done by flow-matching
      models.

  - *Limitations:*
    - The authors acknowledge that while their method worked well, there are
      certain tasks (like `puzzle-3x3-play`) where methods that learn a value
      function are still necessary to succeed.
]

#question[
  Can NFs be integrated directly into offline RL algorithms without complex
  auxiliary techniques?
][
  This question examines whether NFs can be used as expressive policies in
  offline RL without the extra machinery (like distillation or resampling) that
  other expressive models typically require.
][
  - *Experiments:*
    - They proposed *NF-RLBC*, which integrates their NF policy directly into a
      minimal actor-critic algorithm (MaxEnt RL+BC). The flexibility of NFs
      allows them to be trained with both Variational Inference (the actor
      update) and Maximum Likelihood Estimation (the BC regularization)
      simultaneously.
    - The method was tested on *30 challenging offline RL tasks* requiring
      long-horizon reasoning and complex manipulation, such as
      `puzzle-3x3-play`, `scene-play`, and `humanoidmaze-medium-nav`.
    - Baselines included other offline RL algorithms that use expressive
      policies, such as *IDQL* (diffusion policy with importance resampling) and
      *FQL/IFQL* (flow-matching policy with distillation).

  - *Results:*
    - NF-RLBC *outperformed all baselines on 15 out of 30 tasks* and achieved
      top-2 results on another 10 tasks.
    - The performance was particularly strong on the tasks requiring the most
      long-horizon and sequential reasoning, such as `puzzle-3x3-play` and
      `scene-play`, where it performed *230% better than the next best baseline*
      on the puzzle task.

  - *Metrics Used:*
    - *Performance:* Success rate on goal-oriented tasks.

  - *Significance of Results:*
    - This shows that NF policies can be *directly integrated with existing
      offline RL algorithms* to achieve high performance, leading to much
      simpler overall methods.
    - The ability of NFs to use direct gradient-based optimization through the
      policy allows them to effectively search for good actions, a capability
      that other expressive policy classes must approximate with more complex,
      intermediate steps like distillation.

  - *Limitations:*
    - The paper does not ablate the specific components of the NF-RLBC
      algorithm, so the relative importance of the VI and MLE terms in the
      objective is not quantified.
]

#question[
  Can a single NF model efficiently support unsupervised exploration?
][
  This question explores whether a single NF can effectively model both the
  Q-function (for policy improvement) and the goal coverage density (for
  exploration) in Unsupervised Goal Sampling (UGS).
][
  - *Experiments:*
    - They proposed *NF-UGS*, where a single NF model is trained to estimate
      both the conditional goal distribution $p_theta(g|s, a)$ (which acts as
      the Q-function) and the marginal goal distribution $p_theta(g)$ (the
      coverage density). This was achieved by using a masking scheme on the
      conditioning variables.
    - Evaluation was on three difficult UGS maze environments: *ant-u-maze,
      ant-big-maze, and ant-hard-maze*, where success requires thorough
      exploration.
    - Baselines included *CRL-uniform* (samples goals uniformly), *CRL-minmax*
      (samples goals the agent finds hardest), and *CRL-oracle* (an oracle that
      uses test-time goals for training).

  - *Results:*
    - NF-UGS *consistently outperformed all unsupervised baselines* across the
      three environments.
    - Notably, NF-UGS also achieved a *higher asymptotic success rate than the
      CRL-oracle* on average, despite the oracle having access to privileged
      information (test goals).

  - *Metrics Used:*
    - *Performance:* Success rate over the course of training (environment
      steps).
    - *Aggregated Score:* The average of the final success rates across all
      three tasks.

  - *Significance of Results:*
    - This demonstrates that NFs are efficient at estimating the densities of
      non-stationary data in RL, which is crucial for exploration.
    - By using the estimated density to sample goals that maximize the entropy
      of the coverage distribution, NF-UGS avoids the performance plateaus seen
      in other heuristic-based exploration strategies.

  - *Limitations:*
    - The aim of this experiment was not to propose a new state-of-the-art UGS
      method, but rather to serve as a proof-of-concept for the density
      estimation capabilities of NFs in this setting.
]

=== Overall Limitations of the Work and Normalizing Flows

- *Restrictive Architectures:* The authors acknowledge that a primary limitation
  of NFs is that their architecture is constrained by the need for the
  transformation $f_theta$ to be invertible.
- *No Novel Architecture:* The paper's contribution is in showing the
  effectiveness of existing NF components in RL, not in designing new NF
  architectures. The design used is adapted from prior work.
