= Streaming DRL

== Overview

An overview of "Streaming Deep Reinforcement Learning Finally Works" by Elsayed
et al. (2024).

=== Challenges

- *Challenge: The "Stream Barrier"*
  - *Description*: Existing deep reinforcement learning (RL) algorithms, whether
    classic streaming methods or streaming adaptations of batch methods, are
    frequently unstable and fail to learn when processing experience as a continuous
    stream (one sample at a time without a replay buffer).
  - *Hypothesis*: The instability arises from several sources exacerbated by
    streaming updates: occasional large updates from non-i.i.d. samples,
    non-stationarity in activation distributions, and improper data scaling. The
    authors believed that by systematically addressing these stability issues,
    streaming deep RL could be made viable.
  - *Proposed Solution*: The paper introduces a set of common techniques bundled
    into a class of algorithms called *stream-x*. The core solutions include:
    1. *Overshooting-bounded Gradient Descent (ObGD)*: A novel optimizer that prevents
      learning instability from large updates by adjusting the step size. It
      approximates the "effective step size" to ensure updates do not drastically
      over-correct the error on a single sample, avoiding expensive backtracking line
      searches.
    2. *Activation Distribution Stabilization*: Using *LayerNorm* without learnable
      parameters before each activation function to maintain a standard normal
      distribution of pre-activations, which promotes favorable learning dynamics
      under non-stationarity.
    3. *Proper Data Scaling*: Using an online algorithm (Welford's algorithm) to
      compute the running mean and variance of observations and rewards, which are
      then used to normalize the inputs and scale the rewards to a consistent range.
  - *Alternative Solutions Discussed*: The paper shows that standard batch RL
    algorithms like PPO, SAC, and DQN, when adapted for streaming (batch size of 1),
    perform poorly and often diverge. It also notes that well-tuned Adam optimizers
    fail to stabilize classic streaming algorithms in these complex domains.

- *Challenge: Sample Inefficiency*
  - *Description*: Streaming learning is commonly believed to be inherently sample
    inefficient because each experience is used only once and then discarded, unlike
    batch methods that reuse samples from a replay buffer.
  - *Hypothesis*: Sample efficiency can be significantly improved in a streaming
    setting by enhancing credit assignment and reducing interference between
    updates.
  - *Proposed Solution*:
    1. *Eligibility Traces*: The stream-x algorithms incorporate eligibility traces ($lambda$)
      to propagate credit more effectively from rewards to past states and actions,
      mimicking the benefits of multi-step returns without needing to store past data.
    2. *Sparse Initialization*: A novel initialization scheme (`SparseInit`) is used
      where a large proportion of network weights are set to zero. This induces sparse
      representations, which reduces interference between updates for dissimilar
      inputs and has been shown to improve sample efficiency.
  - *Alternative Solutions Discussed*: Standard one-step methods are mentioned as
    having slower credit assignment. Batch RL with replay buffers is the primary
    alternative for achieving sample efficiency, but it is computationally expensive
    and not suitable for streaming.

=== Proposed Component: stream-x Algorithms

- *High-Level Description*: The paper proposes a class of streaming deep RL
  algorithms, called *stream-x*, that can learn effectively and stably from a
  single stream of experience without replay buffers, batch updates, or target
  networks. They are built by combining several key techniques: an *ObGD*
  optimizer, *LayerNorm*, online *data normalization/scaling*, *sparse
  initialization*, and *eligibility traces*.
- *Example Component*: `Stream AC(λ)` (Algorithm 7)
  - *Inputs* (per timestep):
    - Current state: $S$
    - Observed next state, reward, and terminal flag: $S^prime$, $R$, $T$
  - *Process*:
    1. The agent normalizes the state observation.
    2. It samples an action $A$ from its policy network.
    3. After executing the action, it receives $S^prime$, $R$, $T$ and
      normalizes/scales them.
    4. It computes the TD error $delta$ using its value network.
    5. It updates the eligibility traces for both the policy ($z_theta$) and value ($z_w$)
      networks.
    6. It uses the *ObGD* optimizer to update the policy and value network weights
      based on the TD error and the respective traces.
  - *Outputs* (updated at each timestep):
    - Updated policy network weights: $theta$
    - Updated value network weights: $w$

=== Dependencies

- *Environments*:
  - MuJoCo
  - DeepMind Control (DM Control) Suite
  - Atari 2600
  - MinAtar
- *Datasets*:
  - Electricity Transformer Temperature (ETTm2) dataset for the time-series
    prediction task.
- *Benchmark Models/Algorithms*:
  - PPO (Proximal Policy Optimization)
  - SAC (Soft Actor-Critic)
  - DQN (Deep Q-Network)
  - Classic streaming algorithms like Classic AC($lambda$) and Classic Q($lambda$).

=== Glaring Assumptions

- *Local Linearity*: The derivation of the computationally cheap overshooting
  bound for the ObGD optimizer assumes that the neural network function behaves
  linearly for small updates. The paper acknowledges this and states the
  assumption holds approximately when the updates are small, which is the goal of
  the optimizer itself.
- *Lipschitz Continuity of Gradient*: To simplify the overshooting bound for TD($lambda$),
  the analysis assumes that the gradient of the value function is Lipschitz
  continuous and that successive states are nearby, such that $abs(gamma nabla_w v(w;x')-nabla_w v(w;x))_i <= 1$.

=== Recommended Prerequisite Knowledge

- *Sutton, R. S., & Barto, A. G. (2018).*Reinforcement learning: An introduction:
  This paper heavily builds on fundamental RL concepts. The book is the standard
  reference for temporal difference (TD) learning, policy gradients, and
  eligibility traces, all of which are central to the work.
- *Sutton, R. S. (1988).*Learning to predict by the methods of temporal
  differences: The original paper introducing TD learning and eligibility traces ($lambda$),
  which is a core mechanism for improving sample efficiency in the proposed
  stream-x algorithms.
- *Welford, B. P. (1962).*Note on a method for calculating corrected sums of
  squares and products: Provides the single-pass algorithm used for the online
  normalization of observations and scaling of rewards, a key component for
  stability.

== Problem Formulation

Of course. Here is a detailed outline of the problem formulation and the
implementation pipeline for the project described in the paper.

== Problem Formulation

The paper frames the learning problem as an episodic *Markov Decision Process
(MDP)*, which is formally defined by the tuple $(cal(S),cal(A),cal(P),cal(R),gamma,d_0,cal(H))$.

- $cal(S)$: The set of all possible states.
- $cal(A)$: The set of all possible actions.
- $cal(P): cal(S) times cal(A) -> Delta(cal(S) times cal(R))$: The transition
  dynamics model, which gives a probability distribution over the next state and
  reward, given the current state and action.
- $cal(R)$: The set of possible reward signals.
- $gamma in [0, 1]$: The discount factor, which determines the present value of
  future rewards.
- $d_0$: The distribution of starting states for an episode.
- $cal(H)$: The set of terminal states.

The agent interacts with this environment based on a policy, $pi(A_t|S_t, theta)$,
which is a probability distribution over actions given the current state,
parameterized by weights $theta$. The goal of the agent falls into two
categories: prediction and control.

=== Agent Objectives

1. *Prediction*: The goal is to accurately estimate the value of being in a state
  or taking an action in a state. This is done by estimating one of two functions:
  - *State-Value Function ($v_pi$)*: The expected return starting from state $s$ and
    following policy $pi$.

$
  v_pi (s) = dot.op = EE_pi [G_t |S_t = s]
$

where the return $G_t$ is the sum of discounted future rewards: $G_t = dot= sum_( k=t+1 )^T gamma^k-t-1R_k$.
- *Action-Value Function ($q_pi$)*: The expected return from starting in state $s$,
  taking action $a$, and then following policy $pi$.

$
  q_pi (s, a) = dot.op = EE_pi [G_t |S_t = s, A_t = a]
$

2. *Control*: The goal is to find a policy that maximizes the expected return from
  the starting state distribution. This is represented by the objective function $J(theta)$:

$
  J(theta) = dot.op = EE_(S_0 ~ d_0) [v_(pi_theta)(S_0)]
$

=== Core Learning Equations

To achieve these objectives in a streaming setting, the paper relies on several
core equations:

- *Temporal Difference (TD) Error ($delta_t$)*: This is the fundamental signal for
  learning. Instead of waiting until the end of an episode to compute the return $G_t$,
  TD learning uses the immediate reward and the estimated value of the next state
  (bootstrapping) . The TD error is the difference between this one-step target
  and the current value estimate.

$
  delta_t = dot.op = R_(t + 1) + gamma hat(v)(S_(t + 1), w) - hat(v)(S_t, w) quad "(Equation 1)"
$

where $hat(v)(S, w)$ is the function approximator (e.g., a neural network) for
the value function with weights $w$.

- *Eligibility Traces ($z_t$)*: To improve credit assignment and sample
  efficiency, eligibility traces are used. These are short-term memory vectors
  that accumulate recent gradients, faded by $gamma lambda$.
  - For a value function, the trace update is:

$
  z_t = dot.op = gamma lambda z_(t - 1) + nabla_w hat(v)(S_t, w) quad "(Equation 2)"
$

- For a policy, the trace update is:

$
  z_t = gamma lambda z_( t-1 ) + nabla_theta log pi_theta(A_t|S_t, theta_t) quad "(Equation 3)"
$

The weight update then uses this trace, scaled by the TD error: $w_( t+1 ) = w_t + alpha delta_t z_t$.

- *Overshooting Bound Condition*: To stabilize learning, the optimizer bounds the
  update size. This is achieved by approximating the "effective step size" and
  ensuring it stays below a threshold. The key approximation used to avoid
  expensive computations is:

$
  xi approx alpha z^top (gamma nabla_w v(w; x') - nabla_w v(w; x)) <= kappa alpha overline(delta)norm(z)_1 quad "(Equation 4)"
$

where $kappa > 1$ is a scaling factor and $overline(delta) = max(abs(delta), 1)$.
This leads to the update size modulator $M <- alpha kappa overline(delta) norm(z_w)_1$ used
in the optimizer.

== Pipeline

The following pipeline details the implementation of the *Stream AC($lambda$)*
algorithm (Algorithm 7), which exemplifies the paper's contribution.

=== Initialization

This stage runs once at the beginning of training.

- *Inputs*:
  - Network architectures for the policy ($pi_theta$) and value ($hat(v)_w$)
    functions.
  - Hyperparameters: discount factor $gamma$, trace decay $lambda$, step sizes $alpha_pi, alpha_v$,
    scaling factors $kappa_pi, kappa_v$, sparsity level $s$.
- *Process*:
  1. Initialize policy and value network weights using *SparseInit* (Algorithm 1).
    This involves setting a fraction $s$ (e.g., 0.9) of weights at each layer to
    zero and the rest using LeCun initialization . Both networks are configured to
    use *LayerNorm* before each activation function.
  2. Initialize the eligibility trace vectors for the policy ($z_theta$) and value ($z_w$)
    networks to zero vectors.
  3. Initialize statistics for data scaling: observation mean $mu_S$ (vector),
    observation variance helper $p_S$ (vector), and reward scaling helper $p_r$ (scalar)
    are all initialized to zero. The global timestep counter $t$ is also initialized
    to one.
- *Outputs*:
  - Initialized policy weights $theta_0$ and value weights $w_0$.
  - Zero vectors $z_theta, 0$ and $z_w, 0$.
  - Initialized normalization and scaling statistics.

=== Environment Interaction

This stage runs at every timestep within an episode.

- *Inputs*:
  - Current normalized state $S_t$: Tensor of shape `(observation_dim,)`.
  - Current policy weights $theta_t$.
- *Process*:
  1. Sample an action $A_t$ from the policy network output distribution, $pi(dot | S_t, theta_t)$.
  2. Send action $A_t$ to the environment.
- *Outputs*:
  - Raw next state $S'_"raw"$, raw reward $R_"raw"$, and terminal flag $T$ from the
    environment.

=== Data Preprocessing & Scaling

This stage immediately follows the environment interaction at every timestep.

- *Inputs*:
  - $S'_"raw"$: Tensor of shape `(observation_dim,)`.
  - $R_"raw"$: Scalar.
  - Current normalization/scaling stats: $mu_S, p_S, p_r, t$.
- *Process*:
  1. *Normalize Observation*: Use *SampleMeanVar* (Algorithm 4) to update the running
    mean and variance of observations, and then normalize the raw next state to
    produce $S'_t+1$.

$
  S'_(t + 1) = (S'_"raw" - mu_S)/sqrt(sigma_S^2 + epsilon.alt)
$

2. *Scale Reward*: Use *ScaleReward* (Algorithm 5) to update the running statistics
  of the undiscounted return and scale the raw reward to produce $R_t+1$.

$
  R_(t + 1) = R_"raw" /sqrt(sigma_r^2 + epsilon.alt)
$

- *Outputs*:
  - Normalized next state $S'_t+1$: Tensor of shape `(observation_dim,)`.
  - Scaled reward $R_t+1$: Scalar.
  - Updated stats: $mu_S, sigma_S^2, p_S, sigma_r^2, p_r, t+1$.

=== TD Error & Eligibility Trace Update

This stage computes the core learning signals for the current timestep.

- *Inputs*:
  - $S_t, S'_t+1, R_t+1, A_t, T$.
  - Current network weights $w_t, theta_t$.
  - Previous eligibility traces $z_w, t-1, z_theta, t-1$.
  - Hyperparameters $gamma, lambda$.
- *Process*:
  1. Calculate the TD Error $delta_t$ using *Equation 1*. If $S'_t+1$ is a terminal
    state ($T="true"$), its value $hat(v)(S'_t+1, w_t)$ is set to 0.
  2. Update the value network eligibility trace $z_w,t$ using *Equation 2*. This
    requires a backward pass through the value network to get $nabla_w hat(v)(S_t, w_t)$.
  3. Update the policy network eligibility trace $z_theta,t$ using a modified version
    of *Equation 3* that includes entropy regularization for better exploration.
- *Outputs*:
  - TD Error $delta_t$: Scalar.
  - Updated value trace $z_w,t$: Tensor, shape `(num_value_params,)`.
  - Updated policy trace $z_theta,t$: Tensor, shape `(num_policy_params,)`.

=== Network Weight Update

This stage applies the learning signals to update the networks using the custom
optimizer.

- *Inputs*:
  - $delta_t, z_w,t, z_theta,t$.
  - Current weights $w_t, theta_t$.
  - Hyperparameters $alpha_v, alpha_pi, kappa_v, kappa_pi$.
- *Process*:
  1. Apply the *Overshooting-bounded Gradient Descent (ObGD)* optimizer (Algorithm 3)
    to update both networks. For each network: a. Calculate the update modulator $M$ based
    on *Equation 4*: $M <- alpha kappa max(abs(delta_t), 1)norm(z)_1$. b. Compute
    the bounded step size: $alpha' <- min(alpha/M, alpha)$. c. Perform the weight
    update: $w_t+1 <- w_t + alpha'_v delta_t z_w,t$ and $theta_t+1 <- theta_t + alpha'_pi delta_t z_theta,t$.
- *Outputs*:
  - Updated value weights $w_t+1$: Tensor, shape `(num_value_params,)`.
  - Updated policy weights $theta_t+1$: Tensor, shape `(num_policy_params,)`.

=== Loop or Reset

This final stage prepares for the next iteration.

- *Process*:
  1. Set the current state for the next iteration: $S_t <- S'_t+1$.
  2. If the episode has not terminated ($T="false"$), return to *Stage 2*.
  3. If the episode has terminated ($T="true"$), reset the eligibility traces $z_w$ and $z_theta$ to
    zero, get a new initial state from the environment, normalize it, and return to
    *Stage 2*.

== Discussion

Here is a detailed outline of the main questions investigated in the paper's
results and discussion sections.

=== Can Streaming Algorithms Overcome the "Stream Barrier"?

This question addresses the primary challenge identified in the paper: the
tendency of deep RL algorithms to become unstable or fail completely when
trained in a purely streaming fashion.

- *Experiments Designed*:
  - The core experiment demonstrates the "stream barrier" by comparing the proposed
    *`stream-x`* algorithms (`stream AC` and `stream Q`) against two other
    categories on three challenging benchmarks (MuJoCo, DM Control, and Atari) .
  - *Category 1 (Classic Streaming)*: Standard streaming algorithms like Classic AC($lambda$)
    and Q($lambda$) using a well-tuned Adam optimizer.
  - *Category 2 (Adapted Batch)*: State-of-the-art batch algorithms (PPO, SAC, DQN)
    converted to a streaming setting by setting their replay buffer and batch size
    to 1 (named PPO1, SAC1, and DQN1).

- *Metrics & Results*:
  - *Metric*: The *Final Average Episodic Return* after training for a fixed number
    of environment steps (20M for MuJoCo/DMC, 200M for Atari). If any of an
    algorithm's runs diverged, its performance was reported as zero to penalize
    instability.
  - *Results*: The experiments clearly showed the stream barrier. Both classic
    streaming methods and the streaming versions of batch methods performed poorly
    or failed entirely across the benchmarks (Figure 2) . In stark contrast, the `stream-x` algorithms
    successfully learned stable and effective policies, overcoming the barrier and
    performing competitively with (and sometimes better than) their batch
    counterparts.

- *Significance & Limitations*:
  - *Significance*: This is a critical result as it provides the first major
    evidence that streaming deep RL is not fundamentally flawed but requires a
    specific set of techniques to ensure stability. It challenges the prevailing
    view that batching and replay buffers are prerequisites for success in deep RL.
  - *Limitations*: This set of experiments primarily focuses on the final outcome
    and stability, not the learning dynamics or sample-by-sample learning process.

=== Are Streaming Methods Inherently Sample Inefficient?

This question challenges the common belief that because streaming agents use
each data sample only once, they must be less efficient than batch agents that
reuse samples.

- *Experiments Designed*:
  - The experiments compared the *learning curves* of `stream-x` algorithms against
    their batch and streaming counterparts over the entire training process.
  - *Continuous Control (MuJoCo)*: `stream AC(0.8)` was compared against PPO, SAC,
    PPO1, and SAC1 (Figure 3) .
  - *Discrete Control (MinAtar)*: `stream Q(0.8)` was compared against DQN, DQN1,
    and Classic Q(0.8) (Figure 4).

- *Metrics & Results*:
  - *Metric*: *Average Episodic Return* plotted against the number of *Time Steps*
    (or frames). The learning curve's slope and final height indicate sample
    efficiency and performance. A 90% confidence interval is shown as a shaded area.
  - *Results*: The results contradicted the assumption of inefficiency. `stream AC` was
    found to be *more* sample efficient than the batch algorithm PPO in several
    MuJoCo environments like Humanoid-v4 and Ant-v4. In MinAtar, `stream Q` was as
    sample efficient as the powerful batch algorithm DQN.

- *Significance & Limitations*:
  - *Significance*: These findings are highly significant as they demonstrate that
    with proper mechanisms like eligibility traces and sparse representations,
    streaming RL can match or even exceed the sample efficiency of prominent batch
    methods, removing a major perceived drawback of the streaming approach.
  - *Limitations*: The paper notes that the MinAtar environments may not be
    challenging enough to fully expose the stream barrier, which could make weaker
    streaming methods appear more viable than they would be in more complex tasks.

=== Which Architectural Components are Critical for Stability and Efficiency?

This question aims to understand the contribution of each novel technique
proposed in the `stream-x` framework.

- *Experiments Designed*:
  - A series of *ablation studies* were conducted on the `stream AC(0.8)` algorithm
    in four MuJoCo environments (Figure 7) .
  - Starting with the full algorithm, one component was removed at a time:
    1. `no ObGD`: The ObGD optimizer was replaced with a well-tuned Adam optimizer.
    2. `no obs/reward norm`: The online normalization of observations and scaling of
      rewards was removed.
    3. `no LayerNorm`: Layer normalization was removed from the network architecture.
    4. `no SparseInit`: The sparse initialization scheme was replaced with standard
      LeCun initialization.
  - A separate ablation studied the impact of *eligibility traces* by comparing `stream AC(0.8)` ($lambda=0.8$)
    against its one-step counterpart `stream AC(0)` ($lambda=0$) (Figure 8).

- *Metrics & Results*:
  - *Metric*: Learning curves (Average Episodic Return vs. Time Steps) were used to
    assess performance.
  - *Results*: The ablations revealed a clear hierarchy of importance.
    - *ObGD and Data Normalization*: Removing either of these was catastrophic,
      causing the agent to fail to learn entirely. This highlights their critical role
      in maintaining stability .
    - *LayerNorm*: Removing it caused a significant drop in performance, showing it is
      crucial for stable learning dynamics.
    - *SparseInit*: Removing it resulted in slower learning but did not cause failure,
      confirming its role is primarily in improving sample efficiency.
    - *Eligibility Traces*: Using traces (`λ=0.8`) led to substantially better
      performance than not using them (`λ=0`), confirming their effectiveness for
      credit assignment.

- *Significance & Limitations*:
  - *Significance*: These studies provide a precise breakdown of why the `stream-x` algorithms
    work. They isolate the components essential for *stability* (ObGD,
    normalizations) from those that boost *efficiency* (Sparsity, Traces), offering
    a clear recipe for building robust streaming agents.
  - *Limitations*: The ablation was conducted only on the `stream AC` algorithm in
    MuJoCo. While the components are shared, their exact impact might differ in
    other algorithms (like `stream Q`) or domains (like Atari).

=== What are the Limitations and Future Directions?

This question, addressed in the "Limitations and future works" section, reflects
on the boundaries of the current work and outlines a path forward for the field.

- *Discussion Points (No New Experiments)*:
  - *Off-Policy Learning*: The paper focuses on on-policy (`stream AC`, `stream SARSA`)
    and a simple off-policy (`stream Q`) algorithm. It acknowledges that it did not
    explore more complex off-policy methods that use importance sampling, which is a
    key area for future work.
  - *Model-Based RL*: The presented methods are all model-free. The paper suggests
    that developing streaming model-based methods, where an agent incrementally
    learns a model of the world, is a promising direction to further improve sample
    efficiency .
  - *Partial Observability*: The work handles partial observability with simple
    techniques (frame stacking, memory traces). A significant future step would be
    to integrate the `stream-x` framework with more sophisticated methods like
    real-time recurrent learning to handle a wider range of partially observable
    environments.

- *Significance*:
  - This section is important for positioning the paper within the broader research
    landscape. It honestly assesses the scope of its contributions and provides a
    valuable, high-level research agenda for reviving streaming deep RL, encouraging
    work on more advanced off-policy methods, model-based learning, and recurrent
    architectures.
