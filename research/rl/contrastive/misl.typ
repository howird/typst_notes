#import "../styles/things.typ": challenge, hypothesis, question

= Mutual Information Skill Learning

== Overview

This paper analyzes a state-of-the-art (SOTA) unsupervised skill learning
algorithm, METRA, and proposes a new, simpler algorithm, Contrastive Successor
Features (CSF), that achieves comparable performance by re-interpreting METRA's
success within the traditional framework of Mutual Information Skill Learning
(MISL).

=== Challenges and Approaches

#challenge[
  Questioning the Viability of Mutual Information for Skill Learning
][
  Recent work, particularly METRA, suggested that moving away from Mutual
  Information (MI) maximization in favor of a Wasserstein dependency measure was
  necessary for high performance in skill discovery, casting doubt on the entire
  MISL framework.

  #hypothesis[
    The paper hypothesizes that METRA's strong performance is not due to
    abandoning MI, but rather due to specific, effective implementations that
    can be explained within the MISL framework.
  ]

  The authors re-interpret METRA's two main components:
  1. *Representation Learning:* They show that METRA's representation objective,
    which uses dual gradient descent to enforce a Lipschitz constraint, is
    approximately equivalent to maximizing a contrastive lower bound on the
    mutual information $I^beta (S,S';Z)$. This connection is established via a
    second-order Taylor approximation.
  2. *Policy Learning:* They show METRA's policy objective is equivalent to
    maximizing a lower bound on MI *plus* an additional term. They identify this
    extra term as being related to an information bottleneck, which discourages
    the policy from compressing unnecessary information about the state
    transitions, thereby promoting exploration.

  *Alternative Solution:* The alternative is METRA's original formulation, which
  optimizes the Wasserstein dependency measure under a temporal distance metric.
]

#challenge[
  Simplifying the SOTA Algorithm
][
  METRA's implementation is complex, relying on a dual gradient descent
  procedure with extra hyperparameters to learn its state representations.

  #hypothesis[
    A simpler algorithm that directly optimizes the core components identified
    in the analysis of METRA (a contrastive MI bound and an information
    bottleneck-style reward) can match SOTA performance with fewer "moving
    parts".
  ]

  The paper proposes Contrastive Successor Features (CSF), a new MISL algorithm.
  1. *Representation Learning:* CSF directly optimizes the contrastive MI lower
    bound (Eq. 8 in the paper) to learn state representations, eliminating the
    need for METRA's dual gradient descent.
  2. *Policy Learning:* CSF uses the same information bottleneck-inspired
    intrinsic reward as METRA, $r(s,s',z) = (phi_k(s') - phi_k(s))^top z$.
    However, it leverages the linearity of this reward structure by using
    *successor features* to learn a vector-valued critic, which is more direct
    than using a generic off-the-shelf RL algorithm.

  *Alternative Solution:* For policy learning, one could use standard
  actor-critic algorithms like SAC or TD3 to optimize the intrinsic reward,
  which is a common practice.
]

=== Proposed Component: Contrastive Successor Features (CSF)

At a high level, *CSF* is an unsupervised reinforcement learning algorithm for
skill discovery. It learns a set of diverse behaviors without an external reward
function by maximizing the mutual information between latent "skill" vectors and
the state transitions they produce.

- *Inputs:*
  - A stream of observations (state vectors or images) from a reward-free Markov
    Decision Process (MDP).
  - A latent skill space $cal(Z)$, from which a skill vector $z$ is sampled for
    each episode (e.g., from a uniform distribution over the d-dimensional unit
    hypersphere, $p(z) = "UNIF"(SS^d-1)$).
- *Outputs:*
  - A learned state representation encoder $phi: cal(S) mapsto RR^d$.
  - A skill-conditioned policy $pi: cal(S) times cal(Z) mapsto Delta(cal(A))$
    capable of executing a diverse range of behaviors corresponding to different
    skills $z$.
  - A successor feature network
    $psi: cal(S) times cal(A) times cal(Z) mapsto RR^d$, which serves as a
    d-dimensional critic for the policy.

=== Dependencies for Reproducibility

- *Environments:*
  - *Gym:* Ant, HalfCheetah.
  - *DeepMind Control (DMC) Suite:* Quadruped, Humanoid.
  - *LEXA:* Kitchen, Robobin.
- *Downstream Tasks (for evaluation):*
  - Hierarchical control tasks from Park et al. (2024): AntMultiGoal,
    HumanoidGoal, QuadrupedGoal, HalfCheetahGoal, HalfCheetahHurdle.
- *Baseline Algorithms (for comparison):*
  - METRA, CIC, DIAYN, DADS, VISR.

=== Assumptions

- *Parametrization is Key:* The success of the method is critically dependent on
  the specific inner-product parameterization of the critic function,
  $f(s, s', z) = (phi(s') - phi(s))^top z$. Ablation studies show that
  alternative parameterizations (e.g., a monolithic MLP or Gaussian/Laplacian
  kernels) lead to catastrophic performance failure.
- *Taylor Approximation Validity:* The core theoretical argument connecting
  METRA to contrastive learning hinges on a second-order Taylor approximation
  (Proposition 2). While the paper provides empirical evidence for this in a
  low-dimensional setting ($d=2$) , it conjectures that the relationship holds
  in higher dimensions without a formal proof.
- *Expressive Variational Family:* The underlying MISL framework assumes that
  the chosen family of variational distributions is expressive enough to
  represent the true posterior distribution of skills given states,
  $p^pi(z|s, s')$, for any given policy $pi$ (Assumption 1).

== Problem Formulation

Here is a detailed outline of the problem formulation and implementation
pipeline for the Contrastive Successor Features (CSF) algorithm.

=== Problem Formulation

The project aims to perform unsupervised skill discovery in a reward-free Markov
Decision Process (MDP). The core idea is to learn a diverse set of behaviors, or
"skills," by maximizing the *mutual information* ($I$) between a latent skill
variable $Z$ and the state transitions $(S, S')$ that result from executing a
policy conditioned on that skill.

1. *Primary Objective: Mutual Information Maximization*
  The fundamental goal is to find a skill-conditioned policy $pi(a|s, z)$ that
  maximizes the mutual information $I^pi (S,S';Z)$. This objective encourages
  the policy to produce state transitions that are highly predictable from the
  given skill $z$, and conversely, skills that are easily distinguishable from
  the resulting transitions. The objective can be expressed as maximizing the
  expected log-posterior of the skill:

$
  max_pi I^pi (S, S'; Z) = max_pi EE_(z ~ p(z), s, s' ~ p^pi (dot | z)) [log p^pi (z|s, s')] quad(E q . 1)
$

where $p^pi(dot|z)$ denotes the distribution over transitions generated by
policy $pi$ conditioned on skill $z$.

2. *Variational Inference Framework*
  Since the true posterior $p^pi(z|s, s')$ is intractable, the problem is
  approached using variational inference. A tractable variational distribution
  $q(z|s,s')$ is introduced to approximate the true posterior. This leads to an
  iterative, two-step optimization process:
  - *E-Step:* Update the variational distribution $q$ to best approximate the
    posterior generated by past policies (the behavior policy $beta$).

$
  q_(k + 1) <- arg max_q EE_(p^beta (s, s', z)) [log q(z|s, s')] quad(E q . 2)
$

- *M-Step:* Update the policy $pi$ to maximize the intrinsic reward defined by
  the frozen variational distribution $q_k$.

$
  pi_(k + 1) <- arg max_pi EE_(p^pi (s, s', z)) [log q_k (z|s, s')] quad(E q . 3)
$

3. *CSF-Specific Representation Learning (The "E-Step")*
  CSF defines the variational distribution $q(z|s,s')$ using a specific
  energy-based parameterization involving a state encoder
  $phi: cal(S) mapsto RR^d$:

$
  q(z|s, s') eq.delta (p(z) e^((phi.alt(s') - phi.alt(s))^top z))/(EE_(p(z')) [e^((phi.alt(s') - phi.alt(s))^top z')]) quad(E q . 4)
$

Substituting (Eq. 4) into the E-step objective (Eq. 2) yields CSF's *contrastive
representation learning objective* for the encoder $phi$:

$
  cal(J)(phi.alt) = underbrace(
    EE_(p^beta (s comma s' comma z)) [(phi.alt(s') - phi.alt(s))^top z], "Positive Term",
  ) - xi underbrace(
    EE_(p^beta (s comma s')) [log EE_(p(z')) [e^((phi.alt(s') - phi.alt(s))^top z')]], "Negative Term",
  ) quad(E q . 5)
$

Here, the positive term pulls the representation of a transition
$phi(s')-phi(s)$ closer to its corresponding skill $z$, while the negative term
pushes it away from other skills. $xi$ is a fixed hyperparameter, set to 5 in
the paper.

4. *CSF-Specific Policy Learning (The "M-Step")*
  For policy learning, CSF defines an intrinsic reward by taking only the
  positive term from the representation objective, a key insight derived from
  analyzing METRA's connection to the information bottleneck.
  - *Intrinsic Reward:*

$
  r(s, s', z) eq.delta(phi.alt(s') - phi.alt(s))^top z quad(E q . 6)
$

- *Policy Objective:* The policy learning objective becomes maximizing the
  expected discounted sum of this linear reward.

$
  max_pi EE_pi [sum_(t = 0)^H gamma^t r(s_t, s_(t + 1), z)]
$

- *Successor Features:* Because the reward is linear in $z$, CSF uses *successor
  features (SF)* for efficient policy learning. It learns a vector-valued critic
  $psi_omega(s, a, z) in RR^d$ that estimates the discounted sum of future
  representation differences. The critic is trained with a TD-loss, and the
  actor is updated to maximize the inner product of the SF critic's output and
  the skill vector:

$
  max_pi EE_(s, z) ~ p^beta, a ~ pi(a|s, z) [psi(s, a, z)^top z] quad(E q . 7)
$

== Pipeline

This pipeline follows Algorithm 1 in the paper and describes one full iteration
of data collection and training.

*Stage 1: Initialization*
- *Description:* Initialize all required neural networks and the replay buffer.
- *Inputs:* None.
- *Action:*
  1. Create a state encoder network $phi_theta: cal(S) mapsto RR^d$.
  2. Create a skill-conditioned policy network (actor)
    $pi_eta: cal(S) times cal(Z) mapsto Delta(cal(A))$.
  3. Create a successor feature network (critic)
    $psi_omega: cal(S) times cal(A) times cal(Z) mapsto RR^d$.
  4. Create a target critic network $psi_overline(omega)$ with weights identical
    to $psi_omega$.
  5. Initialize an empty replay buffer $cal(D)$.
- *Outputs:* Initialized networks
  $phi_theta, pi_eta, psi_omega, psi_overline(omega)$ and buffer $cal(D)$.

*Stage 2: Data Collection*
- *Description:* The agent interacts with the environment using the current
  policy to collect experience.
- *Inputs:* Current policy $pi_eta$, environment.
- *Action:* For a set number of trajectories:
  1. Sample a skill vector $z ~ "UNIF"(SS^d-1)$. Tensor shape: `[d]`.
  2. For each step in the episode, observe state $s_t$, sample action
    $a_t ~ pi_eta(a|s_t, z)$, and observe the next state $s_t+1$.
  3. Store the full transition tuple $(s_t, a_t, s_t+1, z)$ in the replay buffer
    $cal(D)$.
- *Outputs:* Replay buffer $cal(D)$ populated with new experiences.

*Stage 3: Network Training*
- *Description:* Perform one round of gradient updates for the encoder, critic,
  and actor networks using data sampled from the replay buffer.
- *Inputs:* Replay buffer $cal(D)$, all networks
  ($phi_theta, pi_eta, psi_omega, psi_overline(omega)$).
- *Action:*
  1. Sample a mini-batch of `B` transitions $(s_i, a_i, s'_i, z_i)_i=1^B$ from
    $cal(D)$.
    - `s_i`, `s'_i`: State batch. Tensor shape: `[B, state_dim]`.
    - `a_i`: Action batch. Tensor shape: `[B, action_dim]`.
    - `z_i`: Skill batch. Tensor shape: `[B, d]`.
  2. *Update Encoder $phi_theta$:*
    - Compute the contrastive loss from *(Eq. 5)*.
    - Calculate representation differences:
      $Delta phi_i = phi_theta(s'_i) - phi_theta(s_i)$. Tensor shape: `[B, d]`.
    - The positive term is the average of the batch-wise dot product
      $(Delta phi dot z)$.
    - The negative term involves computing the log-sum-exp of dot products
      between each $Delta phi_i$ and all skills $z_j$ in the batch (in-batch
      negatives).
    - Combine terms to get the final loss and perform one gradient step on
      $theta$.
  3. *Update Critic $psi_omega$:*
    - Compute the TD-error for the successor features.
    - Get next actions from the policy: $a'_i ~ pi_eta(a'|s'_i, z_i)$.
    - Compute the TD target using the target critic:
      $hat(psi)_i = (phi_theta(s'_i) - phi_theta(s_i)) + gamma psi_overline(omega)(s'_i, a'_i, z_i)$.
      Tensor shape: `[B, d]`.
    - Get current SF predictions: $psi_"current"_i = psi_omega(s_i, a_i, z_i)$.
      Tensor shape: `[B, d]`.
    - The loss is the mean squared error:
      $cal(L)(omega) = EE[(psi_"current"_i - hat(psi)_i)^2]$.
    - Perform one gradient step on $omega$.
  4. *Update Actor $pi_eta$:*
    - Update the policy using the objective from *(Eq. 7)*.
    - Sample actions from the current policy: $a_i ~ pi_eta(a|s_i, z_i)$.
    - Get the SF values from the updated critic:
      $psi_i = psi_omega(s_i, a_i, z_i)$. Tensor shape: `[B, d]`.
    - The loss is the negative of the expected objective:
      $cal(L)(eta) = -EE[(psi_i)^top z_i]$.
    - Perform one gradient step on $eta$.
  5. *Update Target Network:*
    - Softly update the target critic weights using Polyak averaging:
      $overline(omega) <- tau omega + (1-tau)overline(omega)$.
- *Outputs:* Updated network parameters $theta, omega, eta, overline(omega)$.
  The process then repeats from Stage 2.

== Discussion

Here is a detailed outline of the main questions the paper aimed to answer, the
experiments designed to address them, and the corresponding results and
limitations.

#question[
  What objective do METRA's representations *actually* optimize?
][
  The paper's first theoretical claim is that METRA, despite its motivation of
  enforcing a strict per-transition distance constraint
  ($norm(phi(s')-phi(s))_2^2 <= 1$), actually optimizes an *expected* distance
  constraint ($EE[norm(phi(s')-phi(s))_2^2] = 1$) in practice. The first
  experiment sought to verify this empirically.
][
  The authors trained the METRA algorithm on the state-based Ant environment,
  using a 2-dimensional representation space ($d=2$) to simplify analysis. After
  training for 20 million environment steps, they sampled 10,000 transitions
  from the replay buffer. They then computed the squared L2 norm of the
  representation difference, $norm(phi(s')-phi(s))_2^2$, for each sampled
  transition and plotted the results as a histogram.
][
  *Metric:* The primary metric was the empirical average of the squared L2
  norms. A qualitative inspection of the histogram was also used. *Result:* The
  empirical average was *0.9884*, which is nearly identical to the predicted
  value of 1.0. The histogram showed that while the average was 1, the
  individual values varied widely, with many transitions having a norm greater
  than 1. *Significance:* This result strongly supports the paper's theoretical
  re-interpretation (Proposition 1). It clarifies that METRA enforces its
  constraint "on average" rather than strictly, which is a crucial step in
  linking its objective to a contrastive loss. *Limitations:* This verification
  was performed only on a single, relatively simple "didactic" environment (Ant)
  with a low-dimensional representation space ($d=2$). It's not explicitly
  proven that this finding generalizes to more complex, image-based tasks with
  higher-dimensional representations.
]

#question[
  Do METRA's representations resemble those from contrastive learning?
][
  If METRA's objective is equivalent to a contrastive loss, as the paper claims,
  then its learned representations should exhibit the geometric properties
  characteristic of contrastive learning: *alignment* and *uniformity*. This
  experiment aimed to visualize these properties.
][
  The same trained METRA agent from the previous experiment was used (Ant,
  $d=2$). Two key statistics were visualized using histograms:
  1. *Gaussianity/Alignment:* The conditional difference, $phi(s')-phi(s)-z$,
    was plotted to see if it follows an isotropic Gaussian distribution around
    the origin.
  2. *Uniformity:* The angles of the normalized marginal difference,
    $(phi(s')-phi(s))/norm(phi(s')-phi(s))_2$, were plotted to see if they were
    uniformly distributed on the unit circle.
][
  *Metrics:* The evaluation was a qualitative visual inspection of the resulting
  2D and 1D histograms. *Result:* The visualizations confirmed the hypotheses.
  The conditional differences formed an isotropic Gaussian-like blob (Fig. 2b),
  and the angles of the normalized differences were uniformly distributed (Fig.
  2c). *Significance:* This provides strong empirical evidence for the paper's
  core thesis that METRA is implicitly performing contrastive learning. It
  visually demonstrates that the representations learned via METRA's complex
  objective have the same geometric structure as those from simpler,
  well-understood contrastive losses. *Limitations:* This analysis shares the
  same limitations as the first experiment: it's a qualitative analysis
  performed on a single, low-dimensional, state-based environment.
]

#question[
  What are the key ingredients for a high-performing MISL algorithm?
][
  This question was tackled through a series of three ablation studies on the
  Ant environment.
][
  *Ablation 1: Is the Wasserstein objective necessary?*
  - *Design:* They created "METRA-C," a version of METRA where the complex
    representation objective was replaced with the direct contrastive loss from
    the paper's theory (Eq. 8), and compared its performance to the original
    METRA.
  - *Result:* Using state coverage as the metric, METRA-C performed identically
    to the original METRA.
  - *Significance:* This showed that the Wasserstein dependency measure is not
    the essential component of METRA's success; a simpler contrastive loss
    suffices. This is a major pillar of their argument for simplifying the SOTA
    algorithm.

  *Ablation 2: Is the Information Bottleneck (IB) reward structure important?*
  - *Design:* They tested a version of CSF that used the *full* mutual
    information lower bound as the intrinsic reward, which includes an
    "anti-exploration" term that standard CSF omits.
  - *Result:* The CSF variant without the IB-style reward completely failed to
    explore, while the standard CSF performed well.
  - *Significance:* This is a crucial finding, indicating that simply maximizing
    mutual information is not enough. The specific reward structure, which they
    connect to an information bottleneck principle, is vital for encouraging
    exploration.

  *Ablation 3: Is the critic's parameterization important?*
  - *Design:* They tested several alternative parameterizations for the critic
    function in CSF, such as using a monolithic MLP, a Gaussian kernel, or a
    Laplacian kernel, instead of the standard inner product form
    $(phi(s')-phi(s))^top z$.
  - *Result:* All alternative parameterizations were "catastrophic for
    performance" and failed to learn meaningful skills.
  - *Significance:* This demonstrates that the specific architectural choice of
    taking the *difference* of state representations and using an *inner
    product* with the skill is a key, non-trivial design decision for the
    algorithm's success.
]

#question[
  Can a simpler algorithm (CSF) match the SOTA?
][
  The ultimate question is whether the new, simplified algorithm, CSF, which was
  derived from the preceding analysis, can achieve performance on par with the
  more complex SOTA algorithm, METRA, and other baselines.
][
  CSF was benchmarked against METRA and four other baselines (DIAYN, DADS, CIC,
  VISR). The evaluation was conducted across six diverse environments, including
  state-based (Ant) and image-based (Humanoid, Quadruped) tasks, and navigation
  and manipulation domains. Performance was measured across three different
  evaluation settings:
  1. *Exploration:* State coverage in the unsupervised phase.
  2. *Zero-shot Goal Reaching:* Ability to reach goals without fine-tuning.
  3. *Hierarchical Control:* Performance on downstream tasks when used as a
    low-level controller.
][
  *Metrics:* State coverage, staying time fraction, and downstream task return
  were used for the three settings, respectively. *Result:* CSF performed
  roughly on par with METRA across the full suite of tasks, sometimes slightly
  better and sometimes slightly worse, but consistently competitive. It
  generally outperformed all other baselines. *Significance:* This is the
  paper's main conclusion. It validates their entire analytical journey by
  showing that the simplified algorithm, built on their insights, successfully
  matches the performance of the more complex state-of-the-art method. This
  provides strong evidence that MI maximization remains a viable and effective
  framework for skill discovery. *Limitations:* The authors acknowledge that the
  experiments were on standard but contained benchmarks. It remains an open
  question how CSF would perform on more complex, partially observable
  environments or how it could be scaled to pre-train on large offline datasets.
]
