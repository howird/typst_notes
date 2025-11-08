#import "../styles/things.typ": challenge, hypothesis, question

= Simplifying TD Learning

This paper introduces *Parallelised Q-Network (PQN)*, a simplified, online, deep
Q-learning algorithm that is fast, stable, and memory-efficient, making it a
viable alternative to Proximal Policy Optimization (PPO) for modern vectorized
reinforcement learning.

=== Overview of Challenges and Solutions

#challenge[
  Instability of Temporal Difference (TD) Learning
][
  Temporal difference methods like Q-learning are known to be unstable when
  combined with nonlinear function approximators (e.g., deep neural networks)
  and off-policy data, a problem known as the deadly triad.

  #hypothesis[
    Regularization techniques can provably stabilize TD learning, removing the
    need for target networks and replay buffers.
  ]

  The paper provides a theoretical analysis demonstrating that incorporating
  *LayerNorm* and *$l^2$ regularization* into the Q-function approximator leads
  to a provably convergent TD algorithm. LayerNorm mitigates instability arising
  from the network's nonlinearity, while a small $l^2$ regularization term
  applied to the network's weights handles instability from the off-policy
  distributional shift.
]

#challenge[
  Inefficiency of Off-Policy Methods in Vectorized Environments
][
  Modern reinforcement learning increasingly uses vectorized environments that
  run in parallel on a single GPU to accelerate training.

  #hypothesis[
    Replacing the large, historical replay buffer with online, synchronous
    sampling across many parallel environments can enable efficient off-policy
    Q-learning on a single GPU.
  ]

  PQN eliminates the large replay buffer entirely. Instead, it collects
  experience by interacting with a large number of vectorized environments in
  parallel and performs synchronous updates on these fresh batches of data. This
  approach dramatically reduces memory requirements and allows for an end-to-end
  GPU learning pipeline.
]

=== Proposed Component: Parallelised Q-Network (PQN)

PQN is a simplified deep Q-learning algorithm that removes both the target
network and the large replay buffer. Stability is achieved by integrating
LayerNorm and optional $l^2$ regularization into the Q-network. The algorithm is
designed to leverage highly parallelized, or "vectorized," environments for data
collection and training. A version incorporating $lambda$-returns is also
presented to improve performance by integrating information over multiple time
steps.

- *Inputs*:
  - A batch of transition tuples $(s_t, a_t, r_t, s_t+1)$ collected
    synchronously from a large number, `I`, of parallel environments.
  - The current parameters, $phi$, of the regularized Q-network.
- *Outputs*:
  - The updated parameters, $phi$, of the Q-network, learned by minimizing the
    TD error on the collected batch.

=== Dependencies for Reproduction

- *Environments & Datasets*:
  - *Single-Agent*:
    - Baird's Counterexample
    - Arcade Learning Environment (ALE) (Atari-10 and Atari-57 suites)
    - Craftax
    - MinAtar
    - DeepSea
    - Classic Control (CartPole, Acrobot)
  - *Multi-Agent*:
    - SMAC / SMAX (Starcraft Multi-Agent Challenge)
    - Overcooked
    - Hanabi

=== Key Assumptions

- The theoretical analysis relies on standard assumptions in stochastic
  approximation theory:
  - The parameter space $Phi$ is a compact and convex set.
  - The TD-error vector $delta(phi, sigma.alt)$ is Lipschitz continuous.
  - The step-sizes $alpha_i$ satisfy the Robbins-Munro conditions
    ($sum alpha_i = infinity$, $sum alpha_i^2 < infinity$).
  - Data is sampled from a geometrically ergodic Markov chain.
- The proof for stabilization is derived for a Q-network architecture with
  specific properties, such as being wide and having $C^2$ continuous activation
  functions with bounded second-order derivatives.

=== Additional Perspectives

- *Speed vs. Sample Efficiency*: While the abstract claims PQN is competitive, a
  key finding is that it is significantly faster (up to 50x) than complex
  methods like Rainbow but can be less sample-efficient. The massive speed-up
  often compensates for the modest decrease in sample efficiency, achieving high
  performance in hours instead of days.
- *Exploration Limitation*: PQN uses a simple $epsilon$-greedy exploration
  strategy. Experiments show it underperforms on hard-exploration games in the
  Atari suite, identifying the need for more advanced exploration techniques as
  a clear area for future work.
- *Theoretical Contribution*: The primary contribution is not just the algorithm
  itself, but the rigorous theoretical analysis that for the first time proves
  that regularisation techniques (LayerNorm + $l^2$) can yield convergent TD
  algorithms without target networks or replay buffers, even with off-policy
  data.

=== Recommended Prerequisite Reading

- *Tsitsiklis, J. N., & Van Roy, B. (1997). An analysis of temporal-difference
  learning with function approximation.* This paper is foundational for
  understanding the instability problems "deadly triad" in TD learning that PQN
  is designed to solve.

== Problem Formulation

Here is a detailed outline of the problem formulation and implementation
pipeline for the Parallelised Q-Network (PQN) project.

=== Problem Formulation

The paper formulates the reinforcement learning problem within the framework of
a *Markov Decision Process (MDP)*.

- *MDP Definition*: An infinite-horizon discounted MDP is defined as a tuple
  $cal(M):= angle.l cal(S), cal(A),P_S,P_0,P_R,gamma angle.r$.
  - $cal(S)$: A bounded state space.
  - $cal(A)$: A bounded action space.
  - $P_S:cal(S) times cal(A)->cal(P)(cal(S))$: The state transition probability
    distribution.
  - $P_0 in cal(P)(cal(S))$: The initial state distribution.
  - $P_R:cal(S) times cal(A)->cal(P)([-r_"max",r_"max"])$: The bounded reward
    distribution.
  - $gamma in[0,1)$: The scalar discount factor.

- *Objective*: The agent's goal is to learn an optimal policy $pi^*$ that
  maximizes the expected discounted sum of future rewards, known as the return.

$
  J^pi : = EE_(tau_infinity ~ P_infinity^pi) [sum_(t = 0)^infinity gamma^t r_t ]
$

- *Value Functions*: The value of taking an action in a state is characterized
  by the action-value function, or Q-function.
  - The *Bellman equation* recursively defines the Q-function for a policy $pi$:

$
  Q^pi (x_t) = cal(B)^pi [Q^pi ](x_t)
$

where $x_t=(s_t, a_t)$ is the state-action pair, and $cal(B)^pi$ is the Bellman
operator:

$
  cal(B)^pi [Q](x_t) : = EE_(r_t, s_(t + 1)) [r_t + gamma Q(s_(t + 1), pi(s_(t + 1)))]
$

- The *optimal Q-function*, $Q^*$, satisfies the Bellman optimality equation:

$
  Q^* (x_t) = EE_(s_(t + 1) ~ P_S (x_t), r_t ~ P_R (x_t)) [r_t + gamma max_(a^prime) Q^* (s_(t + 1), a^prime)]
$

- *Temporal Difference (TD) Learning*: The paper uses TD learning to approximate
  the optimal Q-function with a parameterized function $Q_phi(x)$ (a neural
  network).
  - The general semi-gradient TD update for the parameters $phi$ is:

$
  phi.alt_(i + 1) = phi.alt_i + alpha_i (r + gamma Q_(phi.alt_i)(x^prime) - Q_(phi.alt_i)(x)) nabla_phi.alt Q_(phi.alt_i)(x) quad "(Eq. 1)"
$

- For *Q-learning*, this becomes an off-policy update targeting the maximum
  action-value of the next state:

$
  phi.alt_(i + 1) = phi.alt_i + alpha_i (r + gamma sup_(a^prime) Q_(phi.alt_i)(s^prime, a^prime) - Q_(phi.alt_i)(x)) nabla_phi.alt Q_(phi.alt_i)(x)
$

- *PQN-Specific Formulations*:
  - *Regularized Q-Network*: To ensure stability without a target network, the
    Q-network incorporates LayerNorm and optional $l^2$ regularization. The
    paper analyzes a general form for this network:

$
  Q_phi.alt^k (x) = w^top sigma_"Post" circle.small "LayerNorm"^k [sigma_"Pre" circle.small M x]
$

- *$lambda$-Returns*: To improve performance, PQN uses multi-step
  $lambda$-returns. The target for an update at timestep $t$, $R_t^lambda$, is
  computed recursively backwards from a trajectory of length $T$:

$
  R_t^lambda = r_t + gamma [lambda R_(t + 1)^lambda +(1 - lambda) max_(a^prime) Q_phi.alt (s_(t + 1), a^prime)]
$

The recursion is initialized at the end of the trajectory with
$R_i+T^lambda=max_a^(prime)Q_phi(s_i+T, a^prime)$.
- *Loss Function*: The network parameters $phi$ are updated by minimizing the
  mean squared error between the computed $lambda$-return targets and the
  predicted Q-values from the network:

$
  cal(L)(phi.alt) = EE_((x_t, R_t^lambda) ~ B) [(R_t^lambda - Q_phi.alt (x_t))^2 ]
$

== Pipeline

The PQN algorithm is implemented as a synchronous, end-to-end pipeline that is
fully compilable on a GPU.

=== Initialization
- *Description*: The pipeline begins by setting up the learning environment and
  initializing the neural network.
- *Inputs*:
  - Hyperparameters: Number of parallel environments `I`, rollout length `T`,
    learning rate $alpha$, discount $gamma$, lambda $lambda$, etc..
- *Processing*:
  1. Initialize `I` parallel environments and get their initial states $s_0^i$.
  2. Initialize the regularized Q-network $Q_phi$ with random parameters $phi$.
- *Outputs*:
  - *Initial States Tensor `S_0`*:
    - Shape: `[I, state_dim]`
    - Description: A batch containing the starting state for each of the `I`
      environments.
  - *Initial Network Parameters `φ`*.

=== Parallel Experience Collection
- *Description*: The agent interacts with all environments in parallel for a
  fixed number of steps (`T`) to collect a batch of fresh experience. This stage
  replaces the use of a replay buffer.
- *Inputs*:
  - *Current States Tensor `S_t`*: Shape `[I, state_dim]`
  - *Q-Network Parameters `φ`*
  - *Exploration Rate `ε`*
- *Processing*:
  1. This stage runs for `T` consecutive steps.
  2. For each step, an action $a_t^i$ is chosen for each environment `i` using
    an $epsilon$-greedy policy based on the Q-network's output:
    $Q_phi(s_t^i, dot)$.
  3. All `I` actions are passed to the vectorized environment, which executes
    them in parallel.
  4. The environment returns the next states $s_t+1^i$ and rewards $r_t^i$ for
    all `I` environments.
  5. The transitions $(s_t^i, a_t^i, r_t^i, s_t+1^i)$ are stored in a temporary
    buffer.
- *Outputs*:
  - *Experience Buffer*:
    - Shape: A collection of tensors, e.g., states `[T, I, state_dim]`, actions
      `[T, I, 1]`, rewards `[T, I, 1]`.
    - Description: Contains the `T * I` transitions collected during the
      rollout.

=== $lambda$-Return Target Calculation
- *Description*: Using the collected buffer of experience, target Q-values are
  calculated for each transition using the $lambda$-returns formulation.
- *Inputs*:
  - *Experience Buffer* from Stage 2.
  - *Q-Network Parameters `φ`*
  - *Hyperparameters `γ` and `λ`*
- *Processing*:
  1. The calculation proceeds *backwards in time*, from the last step `T-1` to
    the first step `0` of the collected trajectories.
  2. The target for the last step, $R_T-1^lambda, i$, is bootstrapped from the
    Q-network's value of the final state $s_T^i$.
  3. For all preceding steps $t < T-1$, the target $R_t^lambda, i$ is computed
    recursively using the equation for *$lambda$-returns*:

$
  R_t^lambda,i = r_t^i+gamma [lambda R_t+1^lambda,i+(1-lambda)max_a^(prime)Q_phi(s_t+1^i, a^prime)]
$

- *Outputs*:
  - *Target Tensor `Y`*:
    - Shape: `[T, I, 1]`
    - Description: A tensor containing the calculated target value for every
      state-action pair in the experience buffer.

=== Network Update
- *Description*: The Q-network parameters are updated via gradient descent to
  minimize the error between its predictions and the calculated targets. The
  paper notes that multiple updates (epochs) can be performed on the same batch
  of data to improve sample efficiency.
- *Inputs*:
  - *Experience Buffer* from Stage 2.
  - *Target Tensor `Y`* from Stage 3.
  - *Q-Network Parameters `φ`*.
- *Processing*:
  1. The algorithm can loop for a specified number of `epochs`.
  2. Within each epoch, the `T * I` collected experiences are divided into
    minibatches.
  3. For each minibatch `B`: a. Compute the predicted Q-values for the
    state-action pairs $(s_t^i, a_t^i)$ in the minibatch:
    $Q_"pred" = Q_phi(s_t^i, a_t^i)$. b. Fetch the corresponding pre-calculated
    targets $y_t^i$ from the Target Tensor `Y`. c. Calculate the *Mean Squared
    Error Loss*, as defined in the problem formulation. d. Compute the gradient
    of the loss with respect to the network parameters: $nabla_phi cal(L)(phi)$.
    e. Update the parameters $phi$ using an optimizer like Adam or RAdam.
- *Outputs*:
  - *Updated Network Parameters `φ'`*. These parameters are then fed back into
    Stage 2 for the next round of experience collection.

== Discussion

Here is a detailed outline of the main questions the paper aims to answer, the
experiments designed to address them, and the corresponding results and
limitations.

#question[
  Does the proposed regularization scheme stabilize a provably divergent
  problem?
][
  The authors tested their algorithm on *Baird's Counterexample*. This is a
  small, canonical MDP specifically constructed to make off-policy TD methods
  with linear function approximation diverge. They ran an ablation study
  comparing four versions of their agent: 1. With LayerNorm + $l^2$
  regularization. 2. With LayerNorm only.
  3. With $l^2$ regularization only. 4. With no normalization or regularization.
][
  *Metric*: The `Loss (log)` over 5000 training `Episodes` was used to measure
  stability. Divergence is indicated by an exploding loss. *Results*: The
  variants with no normalization (both with and without $l^2$ reg) diverged
  catastrophically, with the loss reaching ~$10^16$. In contrast, the versions
  using *LayerNorm* remained stable, with a low, controlled loss. The
  combination of *LayerNorm + $l^2$* provided the most stable result.
][
  This result provides strong empirical evidence for their theoretical analysis.
  It demonstrates that LayerNorm is the primary contributor to mitigating
  instability, with a small amount of $l^2$ regularization being necessary to
  handle the remaining off-policy instability as predicted by their theory.
]

#question[
  How does PQN compare against established baselines in complex single- and
  multi-agent tasks?
][
  *Single-Agent (Atari)*: PQN was benchmarked against PPO on the Atari-10 suite
  (a smaller, representative subset of games) and against DQN and Rainbow on the
  full Atari-57 suite. *Single-Agent (Craftax)*: PQN was compared to PPO in the
  open-ended Craftax environment, which requires solving multiple sub-tasks and
  has a large, sparse observation space, making it a challenge for
  memory-intensive methods. *Multi-Agent (MARL)*: PQN was combined with Value
  Decomposition Networks (VDN) and evaluated in Smax (a vectorized version of
  SMAC), Overcooked, and Hanabi against strong MARL baselines like MAPPO and
  QMix.
][
  *Metrics*: The primary metrics were task-specific scores, such as
  `Atari-10 Score`, `Win Rate IQM` (Smax), and `Returns` (Craftax, Hanabi),
  plotted against environment frames or timesteps to measure sample efficiency.
  `Computational Time` was also a key metric. *Results*: In Atari, PQN
  outperformed PPO in sample efficiency and was up to *50x faster* than Rainbow
  while achieving a similar median score. In Craftax, PQN was more
  sample-efficient and achieved a higher final score than PPO when both used an
  RNN. In all MARL tasks, PQN-VDN was highly competitive, outperforming QMix and
  MAPPO in Smax and being significantly more sample-efficient than MAPPO in
  Hanabi.
][
  These results establish PQN as a powerful, fast, and simple algorithm that can
  compete with or even surpass more complex, state-of-the-art methods across a
  wide variety of domains. Its performance in Craftax and Smax is particularly
  notable, as it provides a viable off-policy Q-learning baseline for modern
  pure-GPU environments where one was previously lacking.
]

== Which algorithmic components are most crucial for PQN's success?

This question is addressed through a series of ablation studies designed to
isolate the impact of each key component of the PQN algorithm.

- *Experiments Designed*:
  1. *Regularization*: Compared performance with LayerNorm, BatchNorm, and no
    normalization in the Atari-10 suite. A second experiment in Craftax
    specifically tested BatchNorm as an *input* normalizer for large, sparse
    observations.
  2. *$lambda$-Returns*: Evaluated different values for $lambda$ in the Atari-10
    suite to determine the importance of multi-step returns.
  3. *Replay Buffer*: Compared the standard buffer-free PQN to a version that
    uses a large (1M-experience) replay buffer, both running in a pure-GPU
    setting in Craftax.
  4. *Parallelism*: Varied the number of parallel environments from 1 to 512 in
    MinAtar to measure its impact on sample efficiency and training speed.

- *Results and Metrics*:
  - *Metrics*: Aggregate scores like `Atari-10 Score` and `Max Normalised IQM`
    (MinAtar) were used.
  - *Results*:
    - *Regularization* is critical. LayerNorm significantly improves performance
      in Atari, whereas BatchNorm can be detrimental. However, for environments
      with large, sparse inputs like Craftax, using BatchNorm just on the input
      layer is highly effective.
    - Using *$lambda$-returns* is an important design choice, with $lambda=0.65$
      substantially outperforming 1-step TD learning ($lambda=0$).
    - Removing the *replay buffer* is key to speed. The version with a buffer
      took ~6x longer to train for the same final performance due to memory
      access overhead.
    - Increasing the number of *parallel environments* significantly boosts both
      performance and wall-clock training speed.
  - *Significance*: These ablations justify PQN's design. They confirm that
    network normalization is essential for stability, multi-step returns are
    crucial for performance, and the combination of eliminating the replay
    buffer while increasing parallelism is what enables its remarkable speed on
    modern hardware.
  - *Limitations*: The optimal hyperparameter values found (e.g., $lambda=0.65$)
    are specific to the tested environments and may require tuning for different
    tasks.
