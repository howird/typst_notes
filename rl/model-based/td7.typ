#import "../styles/things.typ": challenge, hypothesis, question

= For SALE: State-Action Representation Learning for DRL

== Overview

This paper introduces *TD7*, a deep reinforcement learning (RL) algorithm that
significantly improves upon the performance and stability of its predecessor,
TD3, for continuous control tasks. The core contributions are designed to
address two primary weaknesses in deep RL: sample inefficiency and policy
instability. This is achieved by introducing *SALE (State-Action Learned
Embeddings)*, a novel representation learning technique for low-level states,
and a refined use of *policy checkpoints*. The paper provides an extensive
empirical study to justify its design choices and demonstrates that TD7
outperforms state-of-the-art methods in both online and offline RL settings.

=== Challenges & Solutions

#challenge[
  Sample Inefficiency in Low-Level State Environments
][
  The difficulty of many RL tasks stems from the complexity of the environment's
  dynamics, not just the dimensionality of the observation space. Therefore,
  even for compact, low-level state vectors (e.g., joint angles), learning an
  intermediate representation that explicitly models the state-action
  interaction and dynamics can provide a stronger, more structured learning
  signal than the standard Bellman equation provides.

  #hypothesis[
    The difficulty of many RL tasks stems from the complexity of the
    environment's dynamics, not just the dimensionality of the observation
    space.
  ]

  SALE uses two encoders, $f$ and $g$, to learn a state embedding $z^s = f(s)$
  and a state-action embedding $z^( s a ) = g(z^s, a)$. These encoders are
  trained independently of the main RL agent by minimizing the prediction error
  between the current state-action embedding $z^( s a )$ and the embedding of
  the *next* state, $z^(s') = f(s')$. The learned embeddings are then
  concatenated with the original state and action as augmented input for the
  policy and value networks. The authors conduct a design study comparing their
  approach to other valid choices, including:
  - Using different learning targets for the encoders, such as the raw next
    state $s'$ or the next state-action embedding $z^( s' a' )$.
  - Training the encoders end-to-end with the value function instead of in a
    decoupled manner.
  - Using different normalization techniques like BatchNorm or LayerNorm instead
    of their proposed AvgL1Norm.
]

#challenge[
  Instability Caused by Representation Learning
][
  The authors observed that significantly expanding the action-dependent part of
  the value function's input (via the state-action embedding $z^( s a )$) leads
  to *extrapolation error*, even in the online setting. The value function
  becomes prone to overestimating Q-values for actions not well-represented in
  the replay buffer, causing performance dips.

  #hypothesis[
    The authors observed that significantly expanding the action-dependent part
    of the value function's input (via the state-action embedding $z^( s a )$)
    leads to *extrapolation error*, even in the online setting.
  ]

  To stabilize the learning target, the value from the target Q-network is
  clipped to lie within the minimum and maximum Q-values observed in the replay
  buffer so far. This prevents errant, large value estimates from destabilizing
  the training process.
]

#challenge[
  General Policy Instability in Deep RL
][
  RL policies can be unstable, with performance varying wildly during training.
  Borrowing from supervised learning, the best-performing policy encountered
  during training can be saved and used for evaluation, ensuring that the final
  reported performance is both high and stable, regardless of the current
  policy's quality.

  #hypothesis[
    RL policies can be unstable, with performance varying wildly during
    training.
  ]

  The agent's policy is held fixed for a number of *assessment episodes*. The
  performance is judged based on the *minimum reward* achieved across these
  episodes to penalize instability. If this minimum performance exceeds that of
  the stored checkpoint policy, the current policy is saved as the new
  checkpoint. At test time, the checkpoint policy is used for evaluation, not
  the most recently trained one. The number of assessment episodes is kept low
  during early training and increased later to balance exploration and
  stability.
]

=== High-Level Description of TD7

- *Component*: *TD7 (TD3 + 4 additions)* is a model-free, off-policy deep
  reinforcement learning algorithm for continuous control.
- *Key Additions to TD3*:
  1. *SALE*: For representation learning.
  2. *Policy Checkpoints*: For stable evaluation.
  3. *LAP (Loss-Adjusted Prioritization)*: A form of prioritized experience
    replay.
  4. *Behavior Cloning (BC) Term*: An auxiliary loss for the offline RL setting.
- *Inputs*: The algorithm takes standard MDP transition tuples $(s, a, r, s')$
  from a replay buffer.
- *Outputs*: A deterministic policy $pi(s)$ that maps states to actions to
  maximize cumulative reward. It also outputs learned state $(z^s)$ and
  state-action $(z^( s a ))$ embeddings as intermediate features.

=== Dependencies to Reproduce

- *Base Algorithm*: An implementation of *TD3 (Twin Delayed Deep Deterministic
  Policy Gradient)*.
- *Prioritized Replay*: An implementation of *LAP (Loss-Adjusted Prioritized)
  experience replay*.
- *Behavioral Cloning*: The regularization term from *TD3+BC* is required for
  the offline RL setting.
- *Environments/Datasets*:
  - *Online*: *OpenAI Gym* with the *MuJoCo* physics simulator , specifically
    the HalfCheetah, Hopper, Walker2d, Ant, and Humanoid v4 environments.
  - *Offline*: *D4RL (Datasets for Deep Data-Driven Reinforcement Learning)*
    benchmark , specifically the v2 MuJoCo datasets.

=== Perspectives Missing from the Abstract

- The abstract does not explicitly state that *extrapolation error*, a problem
  typically associated with *offline* RL, was found to be a significant issue in
  the *online* setting when using their state-action embeddings. This finding is
  a key motivation for their value clipping solution.
- It omits the specific and novel mechanism for *policy checkpoints*, namely
  using the *minimum performance* over several assessment episodes as the
  selection criteria to directly penalize unstable policies, which is a
  departure from using mean performance.
- The significance of the *decoupled training* of the SALE encoders is not fully
  conveyed. The empirical study shows that training the embeddings end-to-end
  with the value function performs significantly worse, highlighting that
  separating the dynamics prediction task from the value estimation task is
  critical.

=== Assumptions

- The paper assumes the agent operates in a *Markov Decision Process (MDP)*.
- It focuses on continuous control tasks where the action space can be
  normalized to a range of $[-1, 1]$.
- A core assumption is that for the tested environments, task difficulty is
  primarily defined by the *complexity of the underlying dynamical system*,
  making representation learning beneficial even with low-dimensional state
  vectors.
- The solutions are justified through extensive *empirical evaluation*, not
  theoretical guarantees. The authors note the lack of theoretical analysis as a
  limitation.

=== Recommended Prerequisite Papers

To fully appreciate the paper's contributions, a foundational understanding of
the following is recommended:

- *Fujimoto, S., van Hoof, H., & Meger, D. (2018).*Addressing Function
  Approximation Error in Actor-Critic Methods*.* This paper introduces *TD3*,
  which is the base algorithm that TD7 is built upon. Understanding TD3's
  mechanisms like clipped double Q-learning and delayed policy updates is
  essential.
- *Fujimoto, S., Meger, D., & Precup, D. (2019).*Off-Policy Deep Reinforcement
  Learning without Exploration*.* This paper formally introduces and analyzes
  *extrapolation error*, a concept central to motivating the value clipping
  component of TD7.

== Problem Formulation

Of course. Here is a detailed outline of the problem formulation for the project
described in the paper "For SALE: State-Action Representation Learning for Deep
Reinforcement Learning".

=== Foundational Reinforcement Learning Framework

The problem is framed as a standard *Markov Decision Process (MDP)*, which
provides the mathematical foundation for the reinforcement learning task.

- *MDP Definition*: An MDP is defined by the tuple
  $(cal(S), cal(A), R, p, gamma)$, where:
  - $cal(S)$ is the state space.
  - $cal(A)$ is the action space.
  - $R(s, a, s')$ is the reward function.
  - $p(s'|s, a)$ is the transition dynamics model.
  - $gamma in [0, 1)$ is the discount factor.

- *Objective*: The goal is to learn a policy, $pi: cal(S) -> cal(A)$, which is a
  mapping from states to actions that maximizes the expected discounted return:

$
  J(pi) = EE_(tau ~ pi) [sum_(t = 0)^infinity gamma^t r_t ]
$

where $tau = (s_0, a_0, s_1, a_1, ...)$ is a trajectory sampled by executing
policy $pi$.

- *State-Action Value Function (Q-function)*: The expected return after taking
  action $a$ in state $s$ and thereafter following policy $pi$ is given by the
  Q-function:

$
  Q^pi (s, a) = EE_(tau ~ pi) [sum_(t = 0)^infinity gamma^t r_t |s_0 = s, a_0 = a]
$

=== State-Action Representation Learning (SALE)

The core of the paper's contribution is learning embeddings that model
environment dynamics to provide a richer input for the RL agent.

- *Embedding Encoders*: SALE introduces a pair of encoder networks:
  - A state encoder, $f: cal(S) -> RR^( d_s )$.
  - A state-action encoder, $g: (RR^( d_s ), cal(A)) -> RR^(d_(s a))$.

- *Learned Embeddings*: These encoders produce two types of embeddings:
  - State Embedding: $z^s = f(s)$
  - State-Action Embedding: $z^(s a) = g(z^s, a)$

- *Dynamics Prediction Loss*: The encoders $(f, g)$ are trained jointly and in a
  decoupled manner from the RL agent. Their objective is to model the
  environment's transition dynamics in the latent space by minimizing the mean
  squared error (MSE) between the state-action embedding $z^(s a)$ and the
  embedding of the *next* state $s'$.

$
  cal(L)(f, g) = EE_((s, a, s') ~ cal(D)) [(g(f(s), a) - "stop_grad"(f(s')))^2 ]
$

This can be written more compactly as:

$
  cal(L)(f, g) = EE [(z^(s a) - "stop_grad"(z^(s')))^2 ]
$

The `stop_grad` operation prevents gradients from flowing through the target
$f(s')$, stabilizing the learning target.

=== Agent Formulation with SALE

The learned embeddings augment the inputs to the actor (policy) and critic
(value function) networks.

- *Augmented Network Inputs*: The policy and Q-function are conditioned on the
  original observations *and* the learned embeddings:
  - Value Function: $Q(s, a) -> Q(z^(s a), z^s, s, a)$
  - Policy: $pi(s) -> pi(z^s, s)$

- *Value Function Update (Critic)*: The critic is updated using a target value
  that incorporates the embeddings. To enhance stability, the embeddings used
  for the update are from *fixed* encoders $(f_t, g_t)$ from a previous
  iteration:

$
  y = r + gamma "clip"(Q_t (z'_(t - 1) z_(t - 1)^(s'), s', a'), Q_"min", Q_"max")
$

where $a' ~ pi_t(z^(s')_(t-1), s')$. The value clipping between
$[Q_"min", Q_"max"]$ (the observed range of Q-values in the replay buffer) is
the paper's solution to extrapolation error.

- *Policy Update (Actor)*: The actor is updated using the deterministic policy
  gradient, also using the fixed embeddings:

$
  nabla_phi.alt J(pi_phi.alt) approx EE_(s ~ cal(D)) [nabla_a Q(z_t^s, z_t^(s a), s, a)|_(a = pi(z_t^s, s)) nabla_phi.alt pi_phi.alt (z_t^s, s)]
$

For the offline setting, a behavior cloning (BC) term is added to the policy
objective to regularize it towards actions in the dataset:

$
  pi approx arg max_pi EE_((s, a) ~ cal(D)) [Q(s, pi(s)) - lambda|EE_(s ~ cal(D)) [Q(s, pi(s))]|_times (pi(s) - a)^2 ]
$

== Pipeline

=== Initialization Stage

This initial stage involves setting up all the required neural networks,
hyperparameters, and data structures before training begins.

- *Description*: Instantiate the actor, critic, and encoder networks, along with
  their respective target/fixed counterparts. The optimizer and the prioritized
  replay buffer are also initialized.
- *Inputs*:
  - Environment-specific dimensions: `state_dim`, `action_dim`.
  - Hyperparameters for the algorithm, such as learning rate, discount factor
    $gamma$, buffer capacity, and mini-batch size $N$.
- *Outputs (Initialized Objects)*:
  - *Actor-Critic Networks*:
    - Policy Networks: $pi_(t+1)$ (current) and $pi_t$ (target).
    - Value Networks: $(Q_(t+1,1), Q_(t+1,2))$ (current) and
      $(Q_(t,1), Q_(t,2))$ (target).
  - *SALE Encoder Networks*:
    - Current Encoders: $(f_(t+1), g_(t+1))$.
    - Fixed Encoders: $(f_t, g_t)$.
    - Target Fixed Encoders: $(f_(t-1), g_(t-1))$.
  - *Checkpoint Networks (Online RL)*:
    - Checkpoint Policy: $pi_c$.
    - Checkpoint Encoder: $f_c$.
  - *Optimizer*: A shared Adam optimizer for all networks.
  - *Replay Buffer*: A Loss-Adjusted Prioritized (LAP) replay buffer with a
    capacity of 1M transitions.

=== Data Collection Stage (Online RL)

The agent interacts with the environment to gather experience. This stage runs
for a set number of "assessment episodes" before a batch of training occurs.

- *Description*: The current policy $pi_(t+1)$ is used to select actions and
  step through the environment. The resulting transition tuples are collected.
  For the first 25k steps, actions are sampled uniformly to ensure initial
  exploration.
- *Inputs*:
  - Current environment state $s_k$ with shape $RR^("state_dim")$.
  - The current policy $pi_(t+1)$ and the fixed state encoder $f_t$.
- *Outputs*:
  - A transition tuple $(s_k, a_k, r_k, s_(k+1))$ to be stored in the replay
    buffer.
    - *Action $a_k$*: The action is generated by the policy plus exploration
      noise: $a_k = pi_(t+1)(f_t(s_k), s_k) + epsilon$, where
      $epsilon ~ cal(N)(0, 0.1^2)$. Shape is $RR^("action_dim")$.
    - *Reward $r_k$*: A scalar value. Shape is $RR$.
    - *Next State $s_(k+1)$*: The subsequent state from the environment. Shape
      is $RR^("state_dim")$.

=== Data Storage and Sampling Stage

Transitions are stored and then sampled in mini-batches for training.

- *Description*: The collected transition is added to the LAP replay buffer with
  maximum priority. For each training step, a mini-batch is sampled from this
  buffer according to priority weights controlled by hyperparameter $alpha$.
- *Inputs*:
  - A transition tuple $(s, a, r, s')$.
  - Mini-batch size $N = 256$.
- *Outputs (A Sampled Mini-batch)*:
  - States $s$: Tensor of shape $RR^(N times "state_dim")$.
  - Actions $a$: Tensor of shape $RR^(N times "action_dim")$.
  - Rewards $r$: Tensor of shape $RR^(N times 1)$.
  - Next States $s'$: Tensor of shape $RR^(N times "state_dim")$.

=== Representation Learning Stage (SALE Update)

The SALE encoders $(f_(t+1), g_(t+1))$ are trained to model the environment's
dynamics in latent space.

- *Description*: Using the sampled mini-batch, the current encoders are updated
  by minimizing the MSE between the state-action embedding and the embedding of
  the next state.
- *Inputs*:
  - The sampled mini-batch $(s, a, s')$.
  - The current encoder networks $(f_(t+1), g_(t+1))$.
- *Process & Outputs*:
  1. *Generate Embeddings*:
    - State Embedding: $z^s = "AvgL1Norm"(f_(t+1)(s))$. Shape:
      $RR^(N times 256)$.
    - State-Action Embedding: $z^(s a) = g_(t+1)(z^s, a)$. Shape:
      $RR^(N times 256)$.
  2. *Generate Target Embedding*:
    - Next State Embedding: $z^(s') = "stop_grad"("AvgL1Norm"(f_(t+1)(s')))$.
      Shape: $RR^(N times 256)$.
  3. *Calculate Loss & Update*:
    - Loss: $cal(L)_"SALE" = (z^(s a) - z^(s'))^2$.
    - A gradient descent step is performed on the parameters of $f_(t+1)$ and
      $g_(t+1)$, resulting in *updated encoder networks*.

=== Agent Training Stage (Critic & Actor Update)

The core reinforcement learning updates for the value functions and the policy
occur here, using the fixed embeddings to ensure a stable learning process.

- *Description*: The critic networks are updated using the Bellman equation, and
  the actor network is updated using the deterministic policy gradient. These
  updates are performed for a number of steps equal to the number of environment
  interactions since the last training batch.
- *Inputs*:
  - The sampled mini-batch $(s, a, r, s')$.
  - All current, target, and fixed networks.
- *Critic (Value) Update*:
  1. *Compute Target Value $y$*:
    - Get target fixed embeddings: $z^(s')_(t-1) = "AvgL1Norm"(f_(t-1)(s'))$.
    - Get target actions: $a' = pi_t(z^(s')_(t-1), s') + epsilon_"clip"$, where
      $epsilon_"clip" ~ "clip"(cal(N)(0, 0.2^2), -0.5, 0.5)$.
    - Get target fixed state-action embeddings:
      $z'^(s a)_(t-1) = g_(t-1)(z^(s')_(t-1), a')$.
    - Calculate the clipped double Q-learning target:
      $y = r + gamma "clip"(min_(i=1,2) Q_(t,i)(z'^(s a)_(t-1), z^(s')_(t-1), s', a'), Q_"min", Q_"max")$.
  2. *Compute Current Q-Value*:
    - Get fixed embeddings: $z^s_t = "AvgL1Norm"(f_t(s))$ and
      $z^(s a)_t = g_t(z^s_t, a)$.
    - Get current Q-estimates:
      $Q_("current", i) = Q_(t+1, i)(z^(s a)_t, z^s_t, s, a)$.
  3. *Calculate Loss & Update*:
    - Critic Loss: $cal(L)_"critic" = "Huber"(y - Q_("current", i))$ for each
      critic $i$.
    - Perform a gradient descent step, resulting in *updated critic networks*
      $(Q_(t+1,1), Q_(t+1,2))$.
- *Actor (Policy) Update (every 2 steps)*:
  1. *Compute Policy Action & Q-Value*:
    - Actions from current policy: $a_pi = pi_(t+1)(z^s_t, s)$.
    - Corresponding state-action embeddings: $z^(s a_pi)_t = g_t(z^s_t, a_pi)$.
    - Value of policy actions:
      $Q_pi = 0.5 sum_(i=1,2) Q_(t+1,i)(z^(s a_pi)_t, z^s_t, s, a_pi)$.
  2. *Calculate Loss & Update*:
    - Actor Loss: $cal(L)_"actor" = -Q_pi$.
    - Perform a gradient ascent step, resulting in an *updated policy network*
      $pi_(t+1)$.

=== Target and Fixed Network Update Stage

To maintain learning stability, the target and fixed networks are periodically
synchronized with the current networks using a hard update.

- *Description*: Every 250 training steps, the weights of the current networks
  are copied to their respective target/fixed counterparts.
- *Inputs*: The latest current networks $(Q_(t+1), pi_(t+1), f_(t+1), g_(t+1))$
  and fixed encoders $(f_t, g_t)$.
- *Outputs*:
  - Updated Target Networks: $Q_t <- Q_(t+1)$ and $pi_t <- pi_(t+1)$.
  - Updated Fixed Encoders: $(f_t, g_t) <- (f_(t+1), g_(t+1))$.
  - Updated Target Fixed Encoders: $(f_(t-1), g_(t-1)) <- (f_t, g_t)$ (from
    before the update).

=== Checkpointing and Evaluation Stage (Online RL)

After the assessment and training phases, the agent determines if the current
policy is a new "best" and, at evaluation intervals, measures performance.

- *Checkpointing Logic*:
  - *Description*: The minimum reward from the just-completed assessment
    episodes is compared against the stored performance of the checkpoint policy
    $pi_c$.
  - *Inputs*: A list of episode rewards from the assessment phase; the current
    checkpoint performance value.
  - *Outputs*: If the new minimum reward is higher, the checkpoint is updated:
    $pi_c <- pi_(t+1)$ and $f_c <- f_t$.
- *Evaluation Logic*:
  - *Description*: Every 5000 environment steps, performance is measured by
    averaging the undiscounted reward over 10 deterministic episodes using the
    checkpoint policy $pi_c$.
  - *Inputs*: The checkpoint policy $pi_c$ and checkpoint encoder $f_c$.
  - *Outputs*: A single scalar score representing the average performance at
    that point in training.

== Discussion

Of course. Here is a detailed outline of the main questions the paper
investigates, the experiments designed to answer them, the results, and their
significance.

#question[
  How does the proposed TD7 algorithm perform against other state-of-the-art
  methods?
][
  The authors benchmarked *TD7* against a suite of strong continuous control
  algorithms in two distinct settings. *Online Setting*: TD7 was compared to
  *TD3*, *SAC*, *TQC*, and *TD3+OFE* on five standard MuJoCo environments from
  the OpenAI Gym benchmark. Performance was tracked over 5 million environment
  time steps. *Offline Setting*: TD7 was compared to *CQL*, *TD3+BC*, *IQL*, and
  *X-QL* using the D4RL benchmark, which provides pre-collected datasets for
  offline learning.
][
  The primary metric was the *average undiscounted sum of rewards* over 10
  evaluation episodes. Performance was reported at intermediate (300k, 1M steps)
  and final (5M steps) stages, averaged over 10 seeds with a 95% confidence
  interval. In the offline setting, the metric was the *average D4RL normalized
  score* after 1 million training steps.
][
  TD7 *significantly outperforms all baselines* in both settings. In the online
  setting, TD7 demonstrated massively improved sample efficiency, achieving an
  average performance gain of *276.7% over TD3 at 300k time steps* and *50.7% at
  5M time steps*. This shows that TD7 learns much faster and achieves a higher
  final performance than existing methods. In the offline setting, the strong
  performance of TD7 over TD3+BC (which uses a similar offline approach)
  highlights that the *SALE representation is highly effective* for learning
  from static datasets.
]

#question[
  Which design choices are most critical for learning effective state-action
  representations?
][
  The authors conducted an extensive *design study* (Section 4.2) to dissect
  their SALE method. They systematically replaced core components of their
  proposed design with plausible alternatives from related literature and
  measured the impact on performance. The ablations were grouped into four
  categories: 1. *Learning Target*: What should the encoder predict?
  Alternatives included the raw next state $s'$, the output of a target encoder
  network $z'_{target}$, or adding a reward prediction loss. 2. *Network Input*:
  How should the learned embeddings be used? They tried removing the state
  embedding $z^s$, the state-action embedding $z^( s a )$, or the original
  $(s, a)$ from the value function's input. 3. *Normalization*: Is the proposed
  `AvgL1Norm` necessary? They compared it to `BatchNorm`, `LayerNorm`, no
  normalization, and using a cosine similarity loss instead. 4. *End-to-End vs.
  Decoupled*: Should the representation be learned jointly with the value
  function? They tested training the encoders end-to-end as an auxiliary loss to
  the value function.
][
  The metric was the *mean percent loss* in performance compared to the default
  TD7 configuration (without checkpoints) at 1M time steps, averaged across five
  MuJoCo environments and 10 seeds.
][
  *Decoupled training is crucial*: Learning the embeddings end-to-end with the
  value function performed significantly worse than the decoupled approach,
  where the encoders are trained with their own dynamics prediction loss. This
  suggests that mixing the value-learning signal with the
  representation-learning signal is harmful. *Augment, don't replace*:
  Performance was highest when the learned embeddings $(z^s, z^(s a))$ were
  concatenated with the original state and action $(s, a)$. Replacing the
  original input with just the embeddings led to poor results, indicating the
  embeddings capture complementary, but not complete, information. *Stabilize
  the learning signal*: Using the next state-action embedding $z^{s'a'}$ as a
  learning target performed very poorly because it relies on the non-stationary
  policy, destabilizing the representation.
]

#question[
  What causes instability when using SALE and how can it be fixed?
][
  The paper investigates the source of performance instability through an
  *extrapolation error study* (Section E). They plotted the performance and
  value estimates of individual seeds on the Ant environment without any value
  clipping. To isolate the cause, they ablated the inputs to the value function,
  specifically testing the effect of removing the state-action embedding
  $z^(s a)$ and reducing the dimensionality of the linear layer over the raw
  state-action input. Finally, they demonstrated the effect of their proposed
  solution: *clipping the target value* by the minimum and maximum Q-values seen
  so far in the replay buffer.
][
  Visual inspection of *per-seed learning curves* for both total reward and the
  critic's value estimate over time. Sharp drops in reward were correlated with
  spikes in the value estimate.
][
  The study found that using a high-dimensional state-action embedding
  ($z^(s a)$) makes the value function prone to *extrapolation error*, causing
  it to produce extreme overestimations for actions it hasn't seen before. This
  occurred even in the online setting and was the primary source of performance
  instability. The proposed solution of clipping the target value *effectively
  stabilizes the value estimates* without needing to alter the powerful
  high-dimensional representation. This finding is significant because it
  identifies and solves a key challenge in applying rich, high-dimensional
  representations to value functions in RL.
]

#question[
  What are the individual contributions of TD7's components and its
  computational cost?
][
  *Ablation Study*: A study was performed (Section G) removing each of the three
  main additions to TD3 one by one: *SALE*, *policy checkpoints*, and *LAP
  (prioritized replay)*. *Run Time Analysis*: The authors benchmarked the
  wall-clock time of TD7 and its online baselines to train for 1M environment
  steps on identical hardware.
][
  *Ablation*: Average normalized performance over all five MuJoCo tasks. *Run
  Time*: Total time in hours and minutes to complete training, and
  run-time-adjusted learning curves plotting reward against wall-clock time.
][
  The ablation study confirmed that *all three components contribute positively*
  to TD7's performance. While policy checkpoints offer a modest stability
  improvement, SALE and LAP are the primary drivers of the large performance
  gains. In terms of computational cost, TD7 is *more than twice as slow as its
  TD3 base* (1h 50m vs 47m). However, it is *significantly faster than other
  top-performing algorithms* like TQC (3h 50m) and TD3+OFE (3h 14m). This
  demonstrates that TD7 provides a favorable trade-off between performance and
  computational efficiency.
]

- *Limitations Acknowledged by the Authors*:
  - *Limited Scope*: The empirical study emphasizes depth on low-level state
    tasks rather than breadth, and does not explore other settings like
    image-based observations.
  - *Few Representation Baselines*: Due to a focus in the literature on vision,
    TD7 was only compared against one other representation learning method
    designed for low-level states.
  - *No Theoretical Results*: The paper is empirical, and the authors do not
    provide theoretical analysis or guarantees for why the proposed methods are
    effective.
