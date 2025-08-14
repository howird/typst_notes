= Decoupling Representation Learning from RL

== Overview

This paper proposes a method for *decoupling representation learning from
reinforcement learning (RL)* in visual domains. The central idea is to learn
visual features using a reward-free, unsupervised learning (UL) task, and then
train the control policy separately using these features.

=== Challenges and Approaches

- *Challenge 1: Inefficient and Task-Specific Feature Learning*
  - *Problem*: In standard deep RL, visual features are learned end-to-end using the
    same reward signal as the policy. This is inefficient under sparse rewards and
    produces features that are not easily transferable to new tasks.
  - *Hypothesis*: A powerful visual representation can be learned without any reward
    signal by focusing on the inherent temporal structure of observations in an
    environment.
  - *Proposed Solution*: Train the convolutional neural network (CNN) encoder using
    only a novel unsupervised task, *Augmented Temporal Contrast (ATC)*. The RL
    agent's policy is then trained independently on the latent representations
    produced by this encoder, whose weights can be frozen.
  - *Alternatives*: The standard approach is end-to-end joint training. Another
    common alternative is using UL tasks as an *auxiliary loss* to supplement the
    main RL loss, rather than replacing it entirely for the encoder.

- *Challenge 2: Generalizing Representations Across Tasks*
  - *Problem*: Features learned for a single RL task often do not generalize well to
    other tasks or environments.
  - *Hypothesis*: An encoder pre-trained on data from multiple environments
    simultaneously can learn a more general set of features.
  - *Proposed Solution*: Use ATC to pre-train a single, multi-task encoder on expert
    demonstrations from several environments. This frozen, pre-trained encoder is
    then used to successfully train separate RL agents on held-out environments.

- *Challenge 3: Policy Regularization with Frozen Encoders*
  - *Problem*: Data augmentation (like random image shifts) is crucial for
    regularizing not just the encoder but also the policy network. When using a
    frozen, pre-trained encoder, replaying full images just to perform this
    augmentation is computationally and memory intensive.
  - *Hypothesis*: The regularization effect of image augmentation can be
    approximated by applying a similar transformation directly to the compressed
    latent representations.
  - *Proposed Solution*: A new augmentation technique called *subpixel random
    shift*. It applies a random shift to the latent image by linearly interpolating
    between neighboring pixels, achieving the same performance benefits as
    image-based augmentation without the computational overhead.

=== Proposed Component: Augmented Temporal Contrast (ATC)

ATC is an unsupervised learning task designed to train a visual encoder for RL
environments.

- *Function*: It learns representations by associating an observation with a
  near-future observation from the same trajectory, distinguishing it from other
  observations (negatives) in a batch.
- *Inputs*:
  - An "anchor" observation, $o_t$.
  - A "positive" observation from a near-future timestep, $o_t+k$.
  - A set of "negative" observations, which are the positive samples corresponding
    to all other anchors in the training batch.
- *Key Architectural Elements*:
  - *Stochastic Data Augmentation*: A random shift is applied to input observations
    to enforce invariance.
  - *Convolutional Encoder ($f_theta$)*: The core component that is shared with the
    RL agent. It processes observations into latent representations.
  - *Momentum Encoder*: A slow-moving average of the primary encoder's weights, used
    to encode the positive examples for stabilized training.
  - *Predictor MLP ($h_psi$)*: A small multi-layer perceptron that predicts the
    future latent code from the anchor's code, acting as an implicit forward model.
  - *Contrastive Loss (InfoNCE)*: The objective function that pushes the
    representations of the anchor (via the predictor) and its positive sample closer
    together, while pushing them away from negative samples.
- *Output*: A trained convolutional encoder ($f_theta$) that produces robust,
  compressed representations of visual observations for a downstream RL policy.

=== Dependencies

To reproduce the paper's method, the following non-novel components are
required:

- *Environments*:
  - DeepMind Control Suite (DMControl)
  - Atari (from the Arcade Learning Environment)
  - DeepMind Lab (DMLab)
- *Base RL Algorithms*:
  - *RAD-SAC*: A variant of Soft Actor-Critic used for DMControl experiments.
  - *PPO*: Proximal Policy Optimization, used for Atari and DMLab experiments.
- *Core Unsupervised Learning Concepts*:
  - *Momentum Contrast (MoCo)*: The concept of using a momentum encoder for
    contrastive learning.
  - *InfoNCE Loss*: The noise-contrastive estimation loss function used.
- *Data*:
  - *Expert Demonstrations*: The offline pre-training benchmarks require datasets
    collected from partially-trained RL agents.

== Problem Formulation

The paper proposes an unsupervised learning (UL) task, *Augmented Temporal
Contrast (ATC)*, to learn visual representations from image-based observations
without relying on a reward signal. The core principle is to train a model to
associate an observation $o_t$ with another observation from a near-future time
step, $o_t+k$, under data augmentations. This task encourages the encoder to
learn the structural and transitional properties of the underlying Markov
Decision Process (MDP).

The architecture consists of four learned components:
1. A *convolutional encoder* ($f_theta$) that processes an augmented anchor
  observation $"AUG"(o_t)$ into a latent image $z_t$.
2. A *linear global compressor* ($g_phi$) which maps the latent image into a
  compact code vector, $c_t = g_phi (z_t)$.
3. A *residual predictor MLP* ($h_psi$) that acts as an implicit forward model,
  producing a predicted future code vector: $p_t = h_psi (c_t) + c_t$.
4. A *contrastive transformation matrix* ($W$).

To generate the target representation, a *momentum encoder* is used, which is a
slowly moving average of the primary encoder ($f_theta$) and compressor ($g_phi$)
networks. Its parameters, $overline(theta)$ and $overline(phi)$, are not updated
by backpropagation but by an exponential moving average of the online network's
weights:

$
  overline(theta) <-(1 - tau) overline(theta) + tau theta
  overline(phi.alt) <-(1 - tau) overline(phi.alt) + tau phi.alt quad "(1)"
$

The target code is then $overline(c)_t+k = g_overline(phi)(f_overline(theta)("AUG"(o_t+k)))$.

The model is trained using the *InfoNCE* loss. Logits are computed via a
bilinear product, $l = p_t W overline(c)_t+k$. Within a training batch, the
positive observation for one anchor serves as a negative example for all other
anchors. If we denote the predicted code for the $i$-th anchor as $p_i$ and the
target code for the $j$-th positive sample as $overline(c)_j+$, the logits are $l_i,j+ = p_i W overline(c)_j+$.
The loss function is then a cross-entropy loss over these logits:

$
  cal(L)_"ATC" = -sum_i log ( exp(l_(i,i+)) ) / ( sum_j exp(l_(i,j+)) ) quad "(2)"
$

== Pipeline

The following pipeline details the online training process where the encoder is
trained exclusively by ATC and is decoupled from the RL agent's gradient
updates, as described in Algorithm 1.

=== Initialization
- *Description*: Initialize all network parameters for the reinforcement learning
  agent and the unsupervised ATC model. Initialize the replay buffer for storing
  observations.
- *Inputs*: None.
- *Actions*:
  - Initialize weights for the ATC model: convolutional encoder ($theta$),
    compressor ($phi$), predictor ($psi$), and contrastive matrix ($W$).
  - Initialize weights for the RL agent's policy ($phi_pi$) and value networks.
  - Initialize the momentum encoder weights ($overline(theta), overline(phi)$) by
    copying the initial weights from the primary encoder and compressor.
  - Initialize an empty replay buffer $cal(S)$ for storing observations.
- *Outputs*:
  - Initialized models: $f_theta, g_phi, h_psi, W, pi_phi_pi$.
  - Initialized momentum models: $f_overline(theta), g_overline(phi)$.
  - Empty observation replay buffer $cal(S)$.

=== Environment Interaction & Data Collection
- *Description*: The agent interacts with the environment using its current policy
  to collect experience. Observations are stored for later use by both the RL and
  UL updates.
- *Inputs*:
  - Current environment observation $s$. Tensor shape depends on the environment,
    e.g., ($C, H, W$) like ($3, 84, 84$).
  - The policy network $pi_phi_pi$ and the shared encoder $f_theta$.
- *Actions*:
  - The agent passes the observation $s$ through the encoder $f_theta$ to get a
    latent representation.
  - The policy $pi_phi_pi$ selects an action $a$ based on the latent representation: $a ~ pi(dot | f_theta (s) \; phi_pi)$.
  - The action is executed in the environment, which returns a new observation $s'$ and
    reward $r$.
  - The new observation $s$ (or $s'$) is added to the replay buffer $cal(S)$.
- *Outputs*:
  - A replay buffer $cal(S)$ populated with environment observations.
  - A history of state-action-reward-next_state transitions for the RL update.

==== Stage 3: Policy Update (RL)
- *Description*: The RL agent's policy and value networks are updated using a
  batch of experience. The key aspect of decoupling is that gradients from this
  step *do not* update the encoder.
- *Inputs*:
  - A batch of transitions ($s, a, r, s'$) sampled from experience.
  - The policy $pi_phi_pi$, value networks, and the encoder $f_theta$.
- *Actions*:
  - The RL algorithm (e.g., PPO or RAD-SAC) computes its objective function.
  - Gradients are calculated and used to update the policy and value network
    parameters ($phi_pi$).
  - Crucially, gradients are stopped from flowing back into the encoder $f_theta$.
    The encoder is used in a "frozen" or "detached" state for this update.
- *Outputs*:
  - Updated policy and value network parameters $phi_pi$.

==== Stage 4: Encoder Update (ATC)
- *Description*: The encoder and other ATC components are updated using the
  unsupervised temporal contrastive loss. This is the sole source of learning for
  the encoder.
- *Inputs*:
  - The observation replay buffer $cal(S)$.
  - The ATC model ($f_theta, g_phi, h_psi, W$) and the momentum model ($f_overline(theta), g_overline(phi)$).
- *Actions*:
  1. *Sampling*: Sample a minibatch of anchor observations $o_t $ and their
    corresponding future positive observations $o_(t+k )$ from $cal(S)$.
  2. *Augmentation*: Apply a stochastic data augmentation (e.g., random shift) to all
    sampled observations.
  3. *Forward Pass (Anchor)*: Process the augmented anchor observations $"AUG"(o_t)$ through
    the primary network to get predicted future codes $p_t $.
    - $z_t = f_theta ("AUG"(o_t))$
    - $c_t = g_phi (z_t)$
    - $p_t = h_psi (c_t) + c_t$
  4. *Forward Pass (Positive/Target)*: Process the augmented positive observations $"AUG"(o_(t+k))$ through
    the *momentum* network to get target codes $overline(c)_(t+k )$.
    - $overline(z)_t+k = f_overline (theta)("AUG"(o_t+k))$
    - $overline(c)_t+k = g_overline (phi)(overline(z)_t+k)$
  5. *Loss Calculation*: Compute the InfoNCE loss $cal(L)_"ATC"$ using *Equation
    (2)*. This involves the bilinear product between all predicted codes $p_i $ and
    all target codes $overline(c)_(j+)$ in the batch.
  6. *Backpropagation*: Compute the gradient $nabla_theta_"ATC"cal(L)^"ATC"$ and
    update the parameters of the encoder, compressor, predictor, and contrastive
    matrix ($theta, phi, psi, W$).
  7. *Momentum Update*: Update the momentum network parameters ($overline(theta), overline(phi)$)
    using the exponential moving average rule from *Equation (1)*.
- *Outputs*:
  - Updated ATC model parameters: $theta, phi, psi, W$.
  - Updated momentum model parameters: $overline(theta), overline(phi)$.

This entire process, from Stage 2 to Stage 4, is repeated until the policy
converges.

== Discussion

Here is a detailed outline of the research questions, experiments, and results
from the paper.

=== Can a reward-free, unsupervised encoder fully replace an end-to-end trained encoder for online RL without losing performance?

- *Experiment Design*:
  - The authors conducted online RL experiments in three distinct domains: *DeepMind
    Control Suite (DMControl)*, *DeepMind Lab (DMLab)*, and *Atari*.
  - They compared two main setups:
    1. *End-to-End RL (Baseline)*: A standard agent where the convolutional encoder is
      trained jointly with the policy using only the RL loss.
    2. *Decoupled ATC*: An agent where the encoder is trained *only* with the
      unsupervised ATC loss. The policy is trained on the output of this encoder, but
      its gradients do not flow back to the encoder.
- *Metrics*:
  - The primary metric was *average return* or *score* over millions of environment
    or agent steps. Performance was visualized through learning curves showing the
    mean and standard deviation across multiple random seeds.
- *Results*:
  - In *DMControl* and *DMLab*, the decoupled ATC encoder achieved nearly equal or
    greater performance compared to the end-to-end baseline across all tested
    environments. In sparse-reward tasks like `Cartpole-Swingup-Sparse` and `Lasertag`,
    ATC provided a significant learning advantage.
  - In *Atari*, the results were mixed. Decoupled ATC worked as well as or better
    than the baseline in 5 of the 8 games tested. However, in games like `Breakout` and `Space Invaders`,
    its performance was lower.
- *Significance*:
  - This was the primary finding of the paper, demonstrating for the first time that
    completely decoupling representation learning from reinforcement learning is a
    viable and often superior strategy.
  - It shows that a reward-free signal based on temporal consistency can be
    sufficient to learn the critical visual features needed for complex control
    tasks, especially in settings where reward signals are sparse or delayed.
- *Limitations*:
  - The method is not universally superior. In a minority of Atari games, the reward
    signal appears necessary to learn the most relevant features, and the decoupled
    approach falls short. In these cases, using ATC as an auxiliary loss or just for
    weight initialization proved more effective than full decoupling.

=== How does ATC compare to other leading unsupervised learning methods when used for pre-training RL encoders?

- *Experiment Design*:
  - The authors proposed a new benchmark methodology for evaluating UL algorithms in
    an RL context.
  - *Step 1*: An encoder is pre-trained offline to convergence on a fixed dataset of
    expert demonstrations using a specific UL algorithm (e.g., ATC, VAE, CPC).
  - *Step 2*: The encoder's weights are frozen, and it is used to train a new RL
    agent from scratch. This isolates the quality of the learned representation
    itself.
  - Comparisons were made against: Augmented Contrast (AC) from CURL, a temporal VAE
    (VAE-T), Pixel Control, Contrastive Predictive Coding (CPC), and Spatio-Temporal
    Deep InfoMax (ST-DIM).
- *Metrics*:
  - Asymptotic performance (average return/score) of the RL agent trained with the
    frozen, pre-trained encoder.
- *Results*:
  - Across all tested domains (DMControl, DMLab, Atari), the ATC-trained encoder
    consistently matched or outperformed encoders trained with all other competing
    UL algorithms.
  - Notably, in `Lasertag`, ATC was significantly better than CPC and Pixel Control
    , and in several Atari games, it was the only UL method to match the performance
    of the end-to-end RL baseline.
- *Significance*:
  - These results establish ATC as a state-of-the-art unsupervised representation
    learning method specifically tailored for reinforcement learning.
  - The proposed evaluation protocol provides a standardized and effective way to
    benchmark the quality of representations for control tasks, separating feature
    learning from policy learning.
- *Limitations*:
  - The "expert demonstrations" used for pre-training were sourced from
    partially-trained agents, not necessarily optimal ones, which could affect the
    quality of the learned representations.

=== Can a single, multi-task encoder trained with ATC generalize to new and unseen environments?

- *Experiment Design*:
  - A single encoder was pre-trained using ATC on a dataset containing expert
    demonstrations from *multiple* environments at once.
  - This frozen, multi-task encoder was then evaluated by training separate RL
    agents on a set of *held-out* environments that were not seen during
    pre-training.
- *Metrics*:
  - Learning curves (return vs. steps) of RL agents on the held-out tasks, using the
    frozen multi-task encoder.
- *Results*:
  - *DMControl*: A single encoder trained on four environments generalized
    remarkably well, achieving strong performance on four completely new
    environments. Tasks with sparse rewards particularly benefited from the
    cross-domain pre-training.
  - *Atari*: Generalization proved more difficult. An encoder trained on eight games
    showed diminished performance compared to individually trained encoders. The
    authors noted this could be partially alleviated by increasing the network's
    capacity, suggesting a representational bottleneck.
- *Significance*:
  - This demonstrates that ATC can learn general, transferable features across a set
    of related tasks (like those in DMControl) without needing complex multi-task
    learning machinery. It points toward the feasibility of creating "foundation"
    encoders for robotic control.
- *Limitations*:
  - The experiments confirm that visual features do not always transfer well between
    highly dissimilar tasks, as seen in the Atari experiments. The effectiveness of
    multi-task pre-training is domain-dependent.

=== Which components of the ATC algorithm are most important for its performance?

- *Experiment Design*:
  - The authors conducted a series of ablation studies to isolate the impact of key
    components of their method.
  - *Random Shift*: They varied the probability of applying random shift
    augmentation and compared its effect across different environments. They also
    introduced and tested a novel *subpixel random shift* applied directly to latent
    images.
  - *Negative Sampling Strategy*: In `Breakout`, they tested using batches composed
    of trajectory segments versus individual transitions, which alters the "hardness"
    of negative examples.
  - *Encoder Analysis*: They visualized the spatial attention maps of different
    encoders to qualitatively assess what parts of an observation the models were
    focusing on.
- *Metrics*:
  - Learning curves for ablation experiments and qualitative comparison of attention
    heatmaps for encoder analysis.
- *Results*:
  - *Random Shift* is a critical component for performance in nearly all domains.
    The new *subpixel random shift* successfully restored performance when applied
    to latent images, removing the need to use full images during the RL update and
    thus saving computation.
  - The negative sampling strategy (using sequences) provided a significant benefit
    in `Breakout` but not in other tested games, suggesting its utility is
    task-specific.
  - The analysis showed that a high-performing ATC encoder learned to attend to
    control-relevant objects (the ball in `Breakout`), similar to an RL-trained
    encoder, while a poorly performing encoder focused on distracting features (the
    paddle).
- *Significance*:
  - These ablations justify the specific design choices in the ATC algorithm.
  - The introduction of subpixel random shift is a practical contribution that makes
    using pre-trained, decoupled encoders more efficient.
- *Limitations*:
  - The authors noted no strong correlation between ATC's accuracy on its own UL
    objective and the final downstream RL performance. This indicates that simply
    maximizing contrastive accuracy does not guarantee better features for control,
    leaving room for future research.
