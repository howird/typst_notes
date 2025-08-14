= Self Prediction as an Auxiliary Task

== Overview

This paper investigates auxiliary learning tasks in reinforcement learning (RL)
to understand when and why certain methods are effective. It specifically
focuses on resolving a theory-practice gap where empirical results often favor
latent self-prediction, while some theoretical work suggests observation
reconstruction should be superior .

=== Challenges

- *Challenge 1: Reconciling the Theory-Practice Gap*
  - *Problem*: Theoretical work has suggested that observation reconstruction should
    provide better features than latent self-prediction, yet empirical studies often
    show the opposite .
  - *Hypothesis*: The effectiveness of a feature learning method depends on its use
    case: whether it is used as a stand-alone pre-training objective or as an
    auxiliary task combined with a primary RL objective like Temporal Difference
    (TD) learning .
  - *Approach*: The paper theoretically analyzes the learning dynamics of both
    latent self-prediction and observation reconstruction under two distinct
    scenarios:
    1. *Stand-alone Setup*: Where only the feature learning loss updates the
      representation .
    2. *Auxiliary Task Setup*: Where the representation is updated by both the feature
      learning loss and the TD learning loss .
  - *Analysis*:
    - In the *stand-alone* case, observation reconstruction is often superior because
      it learns features corresponding to the top *singular vectors* of the transition
      matrix ($P^pi$), which is generally optimal when the reward function is unknown
      .
    - In the *auxiliary task* case, latent self-prediction is preferable. It learns
      features corresponding to the *eigenvectors* of $P^pi$ . This property allows
      for the existence of a stationary point for the combined loss ($L"lat+td"$) that
      can still perfectly represent the value function, assuming the reward lies in an
      invariant subspace . In contrast, the joint reconstruction and TD loss ($L"rec+td"$)
      can have conflicting objectives, as the optimal features for reconstruction (top
      singular vectors) may not align with the eigenvectors needed to represent the
      value function, leading to guaranteed errors .

- *Challenge 2: Formalizing the Impact of Environmental Factors*
  - *Problem*: It is unclear how environmental structures like irrelevant "distractions"
    or different "observation functions" (how states are represented) affect the
    performance of auxiliary tasks .
  - *Hypothesis*: The structure of the observation space and transition dynamics
    significantly impacts the relative performance of different auxiliary losses .
    Specifically, latent self-prediction should be more robust to arbitrary changes
    in the observation function .
  - *Approach*: The paper introduces analytically tractable formalisms for these two
    concepts:
    1. *Observation Functions*: Modeled as an invertible linear transformation ($cal(O)$)
      applied to the base state representation, which accounts for correlations
      between state observations .
    2. *Distractions*: Modeled using factored MDPs, where the total state is a
      Kronecker product of a reward-relevant process and a reward-free distracting
      process ($P_M times.circle P_N$ where $R_N=0$) .
  - *Analysis*:
    - Introducing an observation matrix $cal(O)$ simply results in a linear basis
      change for the learned features in latent self-prediction and TD learning,
      preserving their stability properties .
    - However, for observation reconstruction, the learned features become the
      singular vectors of a transformed matrix ($cal(O)^( -1 ) P^pi cal(O)$), which
      may no longer be optimal for representing the value function .

=== Proposed Component

The paper does not propose a new algorithm. Instead, it introduces an
*analytical framework* to study and understand the learning dynamics of existing
auxiliary tasks in RL under more realistic conditions.

- *High-Level Description*: The framework uses the mathematics of two-layer linear
  networks and dynamical systems theory to analyze the gradient flow of different
  loss functions (latent self-prediction, observation reconstruction, TD learning)
  . It formalizes environmental properties like distractions and observation
  functions to study their impact on the learned representations.
- *Inputs to the Framework*:
  - A policy-induced transition matrix ($P^pi$) and reward vector ($r^pi$) of an MDP
    .
  - An optional invertible observation matrix ($cal(O)$) to model correlated state
    observations .
  - An optional factored MDP structure ($M times.circle N$) to model distractions .
- *Outputs of the Framework*:
  - Theoretical predictions about the stationary points of the learning dynamics
    (i.e., what features are learned) .
  - Qualitative insights into the comparative performance and robustness of
    different auxiliary loss functions when used alone or with TD learning .

=== Dependencies for Reproduction

- *Environments/Datasets*:
  - *MinAtar Suite*: A collection of five Atari-inspired environments (Asterix-v1,
    Breakout-v1, Freeway-v1, Seaquest-v1, SpaceInvaders-v1) .
  - *DeepMind Control (DMC) Suite*: A set of continuous control tasks . The
    experiments use 15 different DMC environments .
- *Baseline Algorithms*:
  - *Double DQN*: Used for experiments in the MinAtar suite .
  - *TD3*: Used for experiments in the DMC suite .

=== Key Assumptions

- *Linear Model Analysis*: The core theoretical results are derived under the
  assumption of two- to three-layer linear networks, which simplifies the complex
  non-linearities of deep neural networks .
- *Fixed Policy Evaluation*: The theoretical analysis is conducted for the policy
  evaluation setting, where the agent's policy is held fixed. It does not model
  the dynamics of policy improvement .
- *Two-Timescale Assumption*: The analysis assumes that certain components of the
  model (e.g., the latent dynamics model $F$ or the value head $hat(V)$) are
  optimized "infinitely faster" than the feature encoder $Phi$ .
- *State Representation and Distribution*: The base case for the theory assumes a
  uniform distribution over states and that state representations are orthogonal
  (i.e., one-hot vectors), such that $EE[x x^top]=I$ .

== Problem Formulation

The paper's theoretical analysis is framed within a discounted Markov Decision
Process (MDP) using a linear function approximation scheme.

*1. Markov Decision Process (MDP)*
An MDP is defined by the tuple $(cal(X), cal(A), cal(P), r, gamma)$, where:
- $cal(X)$ is the state space, with $|cal(X)|=n$ for finite spaces.
- $cal(A)$ is the action space.
- $cal(P)$ is the transition kernel. For a fixed policy $pi$, this is represented
  by a stochastic matrix $P^pi in RR^n times n$.
- $r$ is the reward function, represented by a vector $r^pi in RR^n$ for a fixed
  policy.
- $gamma in [0, 1)$ is the discount factor.

The value function for a policy $pi$ is the expected discounted return, which
can be written in matrix form as:

$
  V^pi=(I-gamma P^pi)^-1r^pi
$

*2. Linear Function Approximation*
The analysis uses a two-layer linear network to model feature learning. The key
components are:
- *Encoder ($Phi$)*: A matrix $Phi in RR^( n times k )$ that maps a one-hot state
  vector $x in RR^n$ to a $k$-dimensional embedding, where $k < n$. The feature
  representation is $phi(x) = x^top Phi$.
- *Latent Model ($F$)*: A matrix $F in RR^( k times k )$ that models the
  transition dynamics in the latent space.
- *Decoder ($Psi$)*: A matrix $Psi in RR^( k times n )$ that maps a latent
  embedding back to the state space, used for reconstruction tasks.
- *Value Weights ($hat(V)$)*: A vector $hat(V) in RR^k$ for the linear value
  function head.

*3. Loss Functions*
The paper analyzes the learning dynamics of three fundamental loss functions,
assuming a uniform and fixed state sampling distribution $cal(D)$.

- *(Eq. 1) Observation Reconstruction Loss ($L_"rec"$)*: Aims to reconstruct the
  next state observation from the current state's latent embedding.

$
  L_"rec" (Phi, F, Psi) = EE_(x ~ cal(D)) [norm(x^top Phi F Psi - x^top P^pi)_2^2 ]
$

- *(Eq. 2) Latent Self-Prediction Loss ($L_"lat"$)*: Aims to predict the *latent
  representation* of the next state. The target features are computed with a
  stop-gradient ($"sg"$) operation, meaning their gradients are not propagated
  back to the encoder.

$
  L_"lat" (Phi, F) = EE_(x ~ cal(D)) [norm(x^top Phi F - [x^top P^pi Phi]_"sg")_2^2 ]
$

- *(Eq. 3) TD Learning Loss ($cal(L)_"td"$)*: The standard Temporal Difference
  loss for learning the value function. The TD target is treated as a fixed label
  via the stop-gradient operation.

$
  cal(L)_"td" (Phi, hat(V)) = EE_(x ~ cal(D)) [norm(x^top Phi hat(V) - [x^top (r^pi + gamma P^pi Phi hat(V))]_"sg")_2^2 ]
$

*4. Environmental Factors*
The framework is extended to formalize two key environmental structures.

- *Observation Funcaions*: Modeled as an invertible matrix $cal(O) in RR^( n times n )$ that
  reparameterizes the state observations. The input to the encoder becomes $x^top cal(O)$ instead
  of $x^top$.
- *Distractions*: Modeled using a factored MDP, where the full transition matrix
  is a Kronecker product of a reward-relevant process ($P_M$) and a distracting,
  reward-free process ($P_N$). The combined transition matrix is $P_M times.circle P_N$.

== Pipeline

This pipeline outlines the empirical evaluation process used in the paper, which
tests different loss configurations on top of a Double DQN (for MinAtar) or TD3
(for DMC) agent.

*Stage 1: Initialization*
- *Description*: Sets up the environment, neural networks, and replay buffer.
- *Inputs*:
  - Environment ID (e.g., `MinAtar/Seaquest-v1`).
  - Hyperparameters from Table 1, such as learning rates, batch size (512), and
    exploration epsilon ($epsilon=0.05$ for MinAtar).
- *Process*:
  1. Instantiate the selected RL environment (MinAtar or DMC).
  2. Initialize the neural networks with random weights according to the architecture
    in Table 2 (MinAtar) or Table 3 (DMC). This includes the Encoder ($Phi$), Q-Head
    ($hat(V)$), and their corresponding target networks.
  3. If an auxiliary task is used, initialize the Latent Model ($F$) and, for
    reconstruction, the Decoder ($Psi$).
  4. Initialize an empty replay buffer.
- *Outputs*:
  - Initialized online and target neural networks.
  - An empty replay buffer.

*Stage 2: Data Collection & Storage*
- *Description*: The agent interacts with the environment to collect experience.
- *Inputs*:
  - Current environment state $s_t$.
  - Online Q-Network (or Actor-Critic networks for TD3).
  - Exploration parameter $epsilon$.
- *Process*:
  1. For the current state $s_t$, select an action $a_t$ using an $epsilon$-greedy
    policy (for DQN) or the actor's policy plus exploration noise (for TD3).
  2. Execute the action in the environment to receive the next state $s_t+1$, reward $r_t$,
    and a done signal $d_t$.
  3. Store the transition tuple $(s_t, a_t, r_t, s_t+1, d_t)$ in the replay buffer.
    The paper does this with a random policy for the first 5000 steps.
- *Outputs*:
  - A replay buffer populated with transition tuples.

*Stage 3: Network Training*
- *Description*: A batch of data is sampled from the replay buffer to update the
  network weights based on the combined loss function.
- *Inputs*:
  - A batch of transitions from the replay buffer.
  - *States ($s$)*: Tensor of shape `(Batch Size, Channels, Height, Width)`. For
    MinAtar, this is `(512, 1, 10, 10)`.
  - *Next States ($s'$)*: Tensor of shape `(Batch Size, Channels, Height, Width)`.
  - *Actions, Rewards, Dones*: Tensors of shape `(Batch Size, 1)`.
- *Process*:
  1. *Feature Encoding*: Pass the state batch $s$ and next state batch $s'$ through
    the *Encoder* network ($Phi$) to get latent representations.
    - *Outputs*: Latent vectors $z = Phi(s)$ and $z' = Phi(s')$ of shape `(Batch Size, k)`,
      where $k=100$ for MinAtar.
  2. *TD Loss Calculation*: Compute the TD loss using *Eq. 3*. The TD target is
    calculated using the target networks to implement the stop-gradient ($[...]_"sg"$).
    For Double DQN, actions are selected with the online network but evaluated with
    the target network.
  3. *Auxiliary Loss Calculation (if applicable)*:
    - *For Latent Self-Prediction*:
      1. Predict the next latent state: $z'_"pred" = F(z)$.
      2. Get the target latent state using the target encoder: $z'_"target" = Phi_"target"(s')$.
      3. Compute the loss using *Eq. 2*: $L_"lat" = norm(z'_"pred" - z'_"target")^2$.
    - *For Observation Reconstruction*:
      1. Predict the latent dynamics: $z'_"latent" = F(z)$.
      2. Reconstruct the next observation using the decoder: $s'_"rec" = Psi(z'_"latent")$.
      3. Compute the loss using *Eq. 1*: $L_"rec" = norm(s'_"rec" - s')^2$.
  4. *Total Loss & Backpropagation*:
    - Combine the losses: $L_"total" = cal(L)_"td" + L_"aux"$.
    - Compute gradients of $L_"total"$ with respect to the online network parameters.
    - *Crucial Logic*: The paper distinguishes between two training setups:
      - *Auxiliary Task Setup*: Gradients from both $cal(L)_"td"$ and $L_"aux"$ are
        backpropagated to update the Encoder ($Phi$).
      - *Stand-alone Setup*: The gradient from $cal(L)_"td"$ is blocked from reaching
        the Encoder ($Phi$) by detaching the latent vector $z$ before it is passed to
        the Q-Head. Only the auxiliary loss updates the encoder.
  5. Update the weights of the online networks using an optimizer (e.g., Adam).
- *Outputs*: Updated weights for the online networks.

*Stage 4: Target Network Update*
- *Description*: The target networks are updated to slowly track the online
  networks, stabilizing training.
- *Inputs*: Online network weights, target network weights.
- *Process*:
  - *For MinAtar (Hard Update)*: Every 1000 environment steps, copy the online
    network weights directly to the target networks: $theta_"target" <- theta_"online"$.
  - *For DMC (Soft Update)*: After each training step, perform a soft update:
    $theta_"target" <- tau theta_"online" + (1-tau)theta_"target"$, where $tau=0.995$.
- *Outputs*: Updated weights for the target networks.

== Discussion

Here is a detailed outline of the main questions investigated in the paper's
empirical results, including the experimental design, outcomes, and limitations
for each.

=== 1. Stand-alone vs. Auxiliary Task Performance

- *Question*: How does the performance of latent self-prediction ($L_"lat"$) and
  observation reconstruction ($L_"rec"$) compare when used as a *stand-alone*
  feature learning objective versus an *auxiliary* task combined with a TD loss?
  This question aims to empirically test *Insight 1* (superiority of
  reconstruction alone) and *Insight 3* (superiority of latent prediction as an
  auxiliary task).

- *Experiment Design*:
  - *Environments*: The MinAtar suite of five games.
  - *Metrics*: The primary metric is the *mean episode reward* over 40 million
    environment steps, averaged across 30 random seeds.
  - *Ablations*:
    1. *Auxiliary Task Setup (Figure 2)*: Standard Double DQN agents were trained with
      three different loss functions:
      - $L_"TD"$ only (the baseline DQN).
      - $L_"lat" + L_"TD"$ (latent self-prediction as an auxiliary task).
      - $L_"rec" + L_"TD"$ (observation reconstruction as an auxiliary task).
      In this setup, gradients from both the TD loss and the auxiliary loss update the
      encoder network.
    2. *Stand-alone Setup (Figure 3)*: The encoder's features were used for the TD
      loss, but its weights were *only* updated by the feature-learning loss. This was
      achieved by detaching the encoder's gradients from the TD loss path. The
      configurations were:
      - $L_"lat" + L_"TD"$ (detach): Encoder updated by $L_"lat"$ only.
      - $L_"rec" + L_"TD"$ (detach): Encoder updated by $L_"rec"$ only.
      - $L_"TD"$ (detach): A baseline where the encoder is randomly initialized and
        never updated (random features).

- *Results and Significance*:
  - In the *auxiliary setup*, adding an auxiliary loss generally improved
    performance over the DQN baseline. Latent self-prediction ($L_"lat"$) was the
    stronger auxiliary task, outperforming observation reconstruction in three of
    the five environments (Asterix, Seaquest, SpaceInvaders). This empirically
    supports *Insight 3*.
  - In the *stand-alone setup*, the results were inverted. Observation
    reconstruction ($L_"rec"$) learned significantly better features and achieved
    much higher rewards than latent self-prediction. Latent self-prediction often
    failed to learn useful features, performing similarly to the random-feature
    baseline. This empirically supports *Insight 1*.
  - *Significance*: These results resolve the "theory-practice gap" mentioned in the
    introduction. They demonstrate that observation reconstruction is a better
    objective for general-purpose, decision-agnostic feature learning, while latent
    self-prediction is more effective as a helper or regularizer for a specific
    decision-aware task like value learning.

- *Limitations*:
  - The results are not universal. For example, in the Seaquest environment,
    stand-alone observation prediction performed exceptionally well, even
    outperforming its use as an auxiliary task. The authors suggest this might be
    due to the sparse reward structure of that specific MDP, highlighting that no
    single method is superior in all contexts.

=== 2. Robustness to Observation Space Distortion

- *Question*: How does arbitrarily distorting the observation space affect the
  performance of different auxiliary tasks? This experiment was designed to test
  *Insight 2*, which hypothesizes that latent self-prediction should be more
  robust to such changes than observation reconstruction.

- *Experiment Design*:
  - *Environments & Metrics*: Same as the previous experiment (MinAtar suite, mean
    episode reward).
  - *Ablation (Figure 4)*: The experiment was run in the *auxiliary task setup*. A
    fixed, random, invertible binary matrix was multiplied with the flattened
    observation vector at each step before being fed to the agent. This simulates a
    change in the observation function without changing the underlying state
    dynamics.

- *Results and Significance*:
  - All algorithms were strongly and negatively impacted by the random observation
    distortion, showing that the theoretical invariance of self-prediction did not
    fully translate to this complex, non-linear setting.
  - However, in two environments (Freeway and Seaquest), the agent with the latent
    self-prediction auxiliary loss was able to recover more of its original
    performance compared to the observation reconstruction agent or the vanilla DQN
    baseline.
  - *Significance*: The results provide partial and weak validation for *Insight 2*.
    While not fully invariant, latent self-prediction showed slightly higher
    resilience in some cases. The experiment also surprisingly revealed that the
    standard DQN baseline relied heavily on the implicit correlations in the
    original observation space, as it suffered the most from the distortion.

- *Limitations*:
  - The authors state that their theoretical claim of invariance likely relies too
    heavily on the simplified linear gradient-flow model, which doesn't hold for
    complex neural networks.
  - The use of a convolutional neural network as the encoder likely interacts with
    the linear distortion in a way not captured by the theory. The original MinAtar
    observations are also far more complex than the one-hot vectors assumed in the
    analysis.

=== 3. Robustness to Distracting State Dynamics

- *Question*: How does the performance of auxiliary tasks change when the
  observations are corrupted by distracting information, and does the *structure*
  of that distraction matter?

- *Experiment Design*:
  - *Environments & Metrics*: Same as the previous experiments. All tests were in
    the *auxiliary task setup*.
  - *Ablations*: Two types of distractions were concatenated to the original game
    observations along the channel dimension:
    1. *Random Noise (Figure 5)*: Distracting channels were filled with unstructured
      noise sampled independently from a Bernoulli distribution. The hypothesis was
      that this predictable noise would be easier for the models to ignore.
    2. *Structured Noise (Figure 6)*: Distracting channels consisted of observations
      from a different, concurrently running MinAtar environment (Freeway-v1) with
      random actions. This noise has its own complex but irrelevant dynamics, making
      it a more challenging distraction.

- *Results and Significance*:
  - With *random noise*, there was a small performance advantage for using the
    latent self-prediction loss in some environments.
  - *Structured noise* posed a much greater challenge to all algorithms, completely
    preventing learning in several environments. In this difficult setting, no
    single algorithm had a clear advantage over the others.
  - *Significance*: These results validate the paper's claim that the *nature* of
    the distraction is critical. Performance is not just about the presence of
    noise, but about its structure and how it might interfere with the spectral
    properties of the underlying environment's transition kernel.

- *Limitations*:
  - The analysis of how these specific distraction models quantitatively affect the
    eigenvalues or singular values of the transition matrix is beyond the paper's
    scope, making the predictions qualitative.
  - The paper criticizes that previous empirical work on distractions often fails to
    specify the nature of the distraction model, and these results show why such
    details are crucial for interpreting outcomes.
