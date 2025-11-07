#import "../styles/things.typ": challenge, hypothesis, question

= Student-Informed Teacher Training

== Overview

This paper, "Student-Informed Teacher Training," introduces a framework to
address a key issue in privileged imitation learning. Here is an overview of its
core concepts.

=== Challenges

#challenge[
  Teacher-Student Asymmetry
][
  In privileged imitation learning, a "teacher" policy is trained with full
  environmental information (e.g., exact obstacle locations), while a "student"
  policy must learn from limited, high-dimensional observations (e.g., camera
  images). This information gap causes the teacher to learn behaviors that the
  student is incapable of imitating due to its partial observability.

  #hypothesis[
    The performance gap between the teacher and student is upper-bounded by the
    expected action difference (specifically, the KL-Divergence) between them.
    The authors hypothesize that by making the teacher aware of this gap during
    its own training, it will learn behaviors that are inherently more imitable
    by the student.
  ]

  The paper proposes modifying the teacher's learning objective to directly
  account for the student's limitations. This is achieved in two ways:
  1. *Reward Penalty*: A penalty term, based on the KL-Divergence between the
    teacher's and student's predicted actions, is subtracted from the teacher's
    task reward. This discourages the teacher from visiting states where its
    actions are hard for the student to replicate.
  2. *Policy Gradient Alignment*: An additional gradient term, also derived from
    the teacher-student KL-Divergence, is added to the teacher's policy update.
    This directly supervises the teacher's network weights, pushing its
    representations to align with the student's.

  Standard imitation learning methods like *Behavior Cloning (BC)* and *DAgger*
  often fail in this asymmetric setting because they assume the student can
  perceive everything necessary to replicate the expert. Other approaches
  augment the *student's* learning by combining imitation loss with a separate
  reinforcement learning objective (e.g., *HLRL*, *COSIL*). This paper's method
  differs by regularizing the *teacher's* policy instead.
]

#challenge[
  Computational Cost of Joint Training
][
  Calculating the student's action at every single step of the teacher's
  training would require rendering expensive, high-dimensional observations
  (like images) continuously, negating the efficiency benefits of using a
  privileged teacher.

  The framework introduces a lightweight *proxy student network* ($hat(F)_S$).
  - This proxy network takes the *teacher's* low-dimensional, privileged
    observations as input and is trained to approximate the action distribution
    of the actual student.
  - This approximation allows the reward penalty and alignment gradient to be
    calculated at every teacher step without high simulation costs.
  - The proxy network is kept synchronized with the real student network during
    a separate *Alignment Phase*, which uses a smaller, sparsely collected
    dataset of paired teacher and student observations.
]

=== Proposed Component: Student-Informed Teacher Training

- *Description*: A joint learning framework that trains a teacher, student, and
  proxy student through three alternating phases: *Roll-Out*, *Policy Update*,
  and *Alignment*. The core innovation is a teacher policy that is optimized not
  only for the task reward but also for its "imitability" by a student with
  limited observations. Key architectural components include a *proxy student
  network* to efficiently approximate student actions and a *shared action
  decoder* to encourage a common feature space between the teacher and student.
- *Inputs*:
  - *Teacher Policy ($pi_T$)*: Privileged, low-dimensional state information
    (e.g., object coordinates, relative distances, agent velocity).
  - *Student Policy ($pi_S$)*: Limited, high-dimensional observations (e.g., RGB
    images) and non-privileged state information (e.g., robot joint positions).
- *Outputs*:
  - A trained *teacher policy* that has learned to perform the task using
    behaviors that are observable and imitable by the student (e.g., avoiding
    self-occlusion, orienting a camera towards obstacles).
  - A trained *student policy* with significantly higher task success rates
    compared to traditional imitation learning baselines.

=== Dependencies

- *Environments & Simulators*:
  - *Color Maze*: A custom 2D grid-world built with the Gym framework.
  - *Flightmare*: A simulator for agile quadrotor flight, used for the obstacle
    avoidance task.
  - *Omniverse Isaac Gym*: A robotics simulator, used for a vision-based Franka
    robot arm drawer-opening task.
- *Pre-trained Models*:
  - *DINOv2*: A frozen vision encoder used by the student policy to process
    image inputs in the quadrotor and manipulation tasks.
- *Core Algorithms*:
  - *PPO (Proximal Policy Optimization)*: The reinforcement learning algorithm
    used as the backbone for teacher training.

=== Additional Notes

- *Missing Perspectives from Abstract*: The abstract introduces the high-level
  concepts of a reward penalty and an alignment step, but omits the crucial
  implementation details that make the framework practical. Specifically, it
  does not mention the *proxy student network* used to avoid expensive
  simulations or the *shared action decoder* used to promote feature alignment.
- *Underlying Assumptions*:
  - The framework requires that the teacher and student policies can be trained
    jointly and online, which may not apply to scenarios involving a fixed,
    pre-trained expert or offline datasets from human demonstrations.
  - The method assumes that the proxy student can effectively learn to mimic the
    student's policy from teacher observations, which relies on a sufficient
    correlation between the two observation spaces.
- *Recommended Prerequisites*: To fully appreciate the paper's contributions, a
  foundational understanding of the following is recommended:
  - *DAgger*: Ross, S., Gordon, G., & Bagnell, D. (2011). *A reduction of
    imitation learning and structured prediction to no-regret online learning*.
    This paper introduces a foundational interactive imitation learning
    algorithm that is a key point of comparison.
  - *PPO*: Schulman, J., et al. (2017). *Proximal policy optimization
    algorithms*. The proposed method is built directly on top of this popular
    reinforcement learning algorithm.

== Problem Formulation

The paper formulates the problem by extending the standard imitation learning
(IL) framework to make the teacher policy aware of the student's limitations.

1. *Standard Reinforcement Learning (RL) Foundation*
  - The problem is set within a *Markov Decision Process (MDP)*, defined as a
    tuple $cal(M)=(cal(S),cal(A),P,R,gamma,mu_0)$.
    - $cal(S)$: State space
    - $cal(A)$: Action space
    - $P$: State transition probability function, $P(s_t+1|s_t, a_t)$
    - $R$: Reward function, $r(s_t, a_t)$
    - $gamma$: Discount factor
    - $mu_0$: Initial state distribution
  - The goal in standard RL is to find an optimal policy $pi^*$ that maximizes
    the expected discounted return $J(pi)$.

$
  J(pi) = EE_(s ~ mu_0) [sum_(t = 0)^infinity gamma^t r(s_t, a_t) | s_0 ~ mu_0, a_t ~ pi(dot | s_t), s_(t + 1) ~ P(dot.op|s_t, a_t)]
$

2. *Imitation Learning (IL) Performance Bound*
  - The paper's method is motivated by a theoretical upper bound on the
    performance difference between a teacher policy ($pi_T$) and a student
    policy ($pi_S$).
  - The performance gap is bounded by the expected action difference between the
    two policies, which is measured under the teacher's state distribution
    $d_pi_T$.

$
  J(pi_T) - J(pi_S) <= (2 sqrt(2) r_(m a x))/((1 - gamma)^2) sqrt(epsilon.alt) quad(1)
$

where $epsilon$ is the upper bound on the expected KL-Divergence:

$
  EE_(s ~ d_(pi_T)) [D_(K L)(pi_T (dot | s), pi_S (dot | s))] <= epsilon.alt quad(2)
$

- Traditional IL focuses on updating the student policy $pi_S$ to minimize this
  KL-Divergence.

3. *Proposed Objective Function*
  - The core idea of the paper is to shift the perspective and find a teacher
    policy $pi_T$ that optimizes the task reward *while also* considering the
    alignment with the student.
  - They reformulate the teacher's objective function, $tilde(J)(pi_T)$, by
    directly incorporating the expected KL-Divergence from Eq. 2 as a penalty
    term.

$
  tilde(J)(pi_T) = EE_(s ~ d_(pi_T), a ~ pi_T (dot | s)) [r(s, a) - D_(K L)(pi_T (dot | s), pi_S (dot | s))] quad(3)
$

- Taking the gradient of this new objective with respect to the teacher's
  parameters $theta$ yields two distinct terms:

$
  nabla_theta tilde(J)(pi_T) = underbrace(
    integral nabla_theta p_theta (tau)(R(tau) - D_theta (tau)) d tau, "Policy Gradient",
  ) - underbrace(
    integral p_theta (tau) nabla_theta D_theta (tau) d tau, "KL-Div Gradient",
  ) quad(8)
$

- *Policy Gradient*: This is the standard policy gradient term, but the reward
  is now modified to include the KL-Divergence as a penalty. This encourages the
  teacher to explore states where student-teacher alignment is high.
- *KL-Div Gradient*: This second term represents a direct supervised signal on
  the teacher's weights, pushing the teacher's action distribution to match the
  student's.

4. *Practical Simplification for Continuous Actions*
  - For continuous action spaces using PPO, where policies are often modeled as
    multivariate Gaussians, the KL-Divergence term simplifies.
  - Assuming the teacher and student share the same covariance matrix
    $Sigma_T = Sigma_S$ (achieved via a shared action decoder), the
    KL-Divergence reduces to the mean difference between the action
    distributions.

$
  D_(K L)(pi_T (dot | s_t), pi_S (dot | s_t)) = 1/2 ["const" +(mu_T (s_t) - mu_S (s_t))^T Sigma_T^(-1)(mu_T (s_t) - mu_S (s_t))] quad(10)
$

- Optimizing this loss (specifically, the KL-Div Gradient term in Eq. 8) aligns
  the teacher network $F_T$ with the proxy student network $hat(F)_S$.

== Pipeline

The proposed framework is implemented as an alternating training loop with three
distinct phases, as shown in Figure 1 of the paper.

=== Roll-Out Phase
This phase is a modified version of the standard on-policy data collection step.

- *Inputs*:
  - Teacher Policy $pi_T$: Comprised of teacher encoder $F_T$ and shared decoder
    $A$.
  - Proxy Student Policy $hat(pi)_S$: Comprised of proxy student encoder
    $hat(F)_S$ and shared decoder $A$.
  - Task Environment $cal(E)$.
  - KL-Divergence weight $lambda_1$.
- *Process*:
  1. For each timestep $t$ in a trajectory, the teacher policy receives a
    privileged state observation $o_T$ from the environment.
  2. The teacher policy $pi_T$ and the proxy student policy $hat(pi)_S$ both
    process $o_T$ to compute their respective action distributions.
  3. A *reward penalty* is calculated using the KL-Divergence between the two
    distributions. This directly implements the penalty in the "Policy Gradient"
    term of *Eq. 8*. The modified reward is:
    $r_t' <- r_t - lambda_1 D_( K L )(pi_T(s_t) || hat(pi)_S(s_t))$.
  4. The teacher samples an action $a_t ~ pi_T(o_T)$ and executes it in the
    environment to get the next state $s_t+1$ and the original task reward
    $r_t$.
  5. The experience tuple $(s_t, a_t, r'_t, s_( t+1 ))$ is stored in an
    experience buffer $cal(B)_"exp"$.
  6. For a small subset of the visited states, a corresponding high-dimensional
    student observation $o_S$ is rendered. The paired observation $(o_T, o_S)$
    is stored in a separate alignment buffer $cal(B)_"align"$.
- *Outputs*:
  - *Experience Buffer ($cal(B)_"exp"$)*: A buffer of transition data for
    training the teacher. Shape: `[num_steps, feature_dim]`.
  - *Alignment Buffer ($cal(B)_"align"$)*: A smaller buffer of paired
    teacher-student observations. Shape:
    `[num_align_samples, dim(o_T) + dim(o_S)]`.

=== Policy Update Phase
In this phase, the teacher policy is updated using the data collected during the
roll-out.

- *Inputs*:
  - Experience Buffer $cal(B)_"exp"$.
  - Teacher Policy $pi_T$.
  - Proxy Student Policy $hat(pi)_S$ (frozen).
  - KL-Divergence weight $lambda_2$.
- *Process*:
  1. A mini-batch of experiences is sampled from $cal(B)_"exp"$.
  2. A combined loss function is computed. This loss consists of the standard
    PPO policy loss (calculated using the modified rewards $r'_t$) and the
    *KL-Div Gradient* term from *Eq. 8*. The combined loss is:
    $cal(L) = cal(L)_"policy" + lambda_2 D_( K L )(pi_T || hat(pi)_S)$. For
    continuous actions, the second term is based on *Eq. 10*.
  3. A single backward pass is performed through the combined loss to update the
    network parameters.
- *Outputs*:
  - *Updated Teacher Policy $pi_T$*: The weights of the teacher encoder $F_T$
    and the shared action decoder $A$ are updated.

=== Alignment Phase
This is the only phase that requires paired teacher and student data. It
synchronizes the student and proxy student networks.

- *Inputs*:
  - Alignment Buffer $cal(B)_"align"$.
  - Student Encoder $F_S$.
  - Proxy Student Encoder $hat(F)_S$.
  - Teacher Encoder $F_T$ (frozen).
  - Shared Action Decoder $A$ (frozen).
- *Process*:
  1. A mini-batch of paired observations $(o_T, o_S)$ is sampled from
    $cal(B)_"align"$.
  2. *Student-to-Teacher Alignment*:
    - $o_T$ and $o_S$ are passed through their respective encoders ($F_T$ and
      $F_S$) to produce feature vectors $h_T$ and $h_S$.
    - An L1 loss is computed between the features:
      $cal(L)_S <-> T = norm(h_T - h_S)_1$. Gradients are backpropagated to
      update *only the student encoder $F_S$*.
  3. *Proxy-to-Student Alignment*:
    - $o_T$ is passed through the proxy student encoder $hat(F)_S$ to get
      features $hat(h)_S$.
    - An L1 loss is computed between the proxy and student features:
      $cal(L)_hat(S) <-> S = norm(h_S - hat(h)_S)_1$.
    - Gradients are backpropagated to update *only the proxy student encoder
      $hat(F)_S$*.
- *Outputs*:
  - *Updated Student Encoder $F_S$*: Aligned to mimic the feature
    representations of the frozen teacher.
  - *Updated Proxy Student Encoder $hat(F)_S$*: Aligned to mimic the feature
    representations of the student.

== Discussion

This outline details the main research questions the paper aimed to answer,
supported by experiments, results, and their significance.

#question[
  Can the framework force a teacher to find a student-imitable path when its own
  optimal path is invisible to the student?
][
  The paper uses a 2D *Color Maze* task. In this grid world, the teacher can see
  a short, optimal path through a "maze," but to the student, this path is
  indistinguishable from deadly "lava" cells. The only safe path for the student
  is a longer route around the maze's exterior. They compare the behavior of a
  teacher and student trained with their full alignment framework against a
  teacher trained without alignment and standard imitation learning baselines
  (Behavior Cloning and DAgger).
][
  The primary metric was the *agent's trajectory*, visualized as an occupancy
  grid, to qualitatively assess whether the agent reached the goal and which
  path it took. The standard teacher, blind to the student's limitations, learns
  the short path through the maze, which the student cannot follow. In contrast,
  the teacher trained with the proposed alignment framework learns to completely
  avoid the maze and instead takes the longer, sub-optimal (for itself) path
  around the exterior. The student is then able to successfully imitate this
  new, visible path.
][
  This experiment provides a clear and definitive proof-of-concept. It shows
  that the framework can successfully alter a teacher's strategy away from its
  own optimal solution to one that is explicitly learnable by a student with
  partial observability. This is a simple, tabular environment designed
  specifically to illustrate the core problem. While effective as a
  demonstration, it does not on its own prove the method's utility in more
  complex, high-dimensional, real-world scenarios.
]

#question[
  Can the framework improve performance in a complex, vision-based task by
  teaching "perception awareness"?
][
  A *vision-based obstacle avoidance task with a quadrotor* in the Flightmare
  simulator. The teacher has perfect knowledge of all obstacle positions, while
  the student must navigate using only a forward-facing RGB camera with a
  limited field of view. Their method (with and without alignment) was
  benchmarked against five baselines: BC, DAgger, HLRL, DWBC, and COSIL.
][
  The primary quantitative metric, measuring the percentage of successful
  obstacle avoidance runs. To quantify *how* the behavior changed, they measured
  the *Velocity Angle* (angle between where the drone is looking and where it's
  going) and the *Number of Obstacles in View*. Lower angles and a higher number
  of visible obstacles are better. The proposed method with alignment achieved
  the highest success rate of *46%*, a significant improvement over all
  baselines, including their own method without alignment (38%) and DAgger (8%).
  The aligned policy learned to actively orient the camera in the direction of
  flight, resulting in a much smaller Velocity Angle (32.2°) and more obstacles
  kept in view (3.51) compared to all other methods.
][
  This demonstrates that the framework scales to complex, dynamic control tasks.
  More importantly, it shows the emergence of "perception-aware" behavior
  without being explicitly programmed. The teacher learns that to be imitated,
  it must actively point the camera to provide the student with necessary
  information. While the success rate is the highest, at 46% it is still far
  from perfect, indicating that the task remains extremely challenging and the
  solution is not a complete one.
]

#question[
  Can the framework learn to avoid self-occlusion in a vision-based manipulation
  task?
][
  A *vision-based drawer opening task* with a Franka robot arm in Omniverse
  Isaac Gym. The camera is positioned such that the robot's arm can easily block
  its own view of the drawer handle. The teacher gets the precise 3D position of
  the handle, while the student only sees the camera image. The same set of
  baselines from the quadrotor experiment were used.
][
  The primary metrics were *Success Rate* and visual inspection of student-view
  images to check for occlusion. The method with alignment achieved a *88%
  success rate* (and up to 95% in ablations), vastly outperforming baselines
  like DAgger (34%) and COSIL (56%). Visual evidence confirms that the aligned
  teacher learns emergent strategies to keep the handle visible, such as
  approaching from above or adjusting its arm links, which the non-aligned
  teacher fails to do.
][
  This result shows the framework's applicability to complex robotic
  manipulation, a key domain. It effectively solves the problem of
  self-occlusion by modifying the teacher's behavior. Interestingly, the aligned
  teacher's own performance *also* improved, suggesting the alignment encourages
  more robust, efficient motions. The authors note that the task's difficulty
  was slightly constrained by a small randomization range for the cabinet's
  position, which allowed even non-aligned policies to achieve some success
  through memorization.
]

#question[
  Which components of the proposed framework are most critical for its success?
][
  The authors conducted an ablation study on the vision-based manipulation task,
  which was the most complex environment. They systematically removed or
  modified three key components of their framework:
  1. The *KL-Divergence penalty* in the reward function.
  2. The *KL-Divergence gradient* in the policy update step.
  3. The *shared action decoder* between the teacher and student.
][
  The impact of each ablation was measured by the change in the final *Success
  Rate*. Removing the *KL-Div Gradient* and the *shared action decoder* caused
  the most severe performance drops, with success rates falling to 47% and 62%,
  respectively. Removing the *reward penalty* had a smaller but still notable
  negative impact, with the success rate dropping to 74%.
][
  This study validates the authors' design choices. It reveals that directly
  aligning the policies via the *KL-Div Gradient* and encouraging a common
  feature space via the *shared action decoder* are the most critical components
  for the framework's success. The ablation was only performed on one of the
  three tested environments. While it was the most complex one, the relative
  importance of the components could potentially vary in different task
  settings.
]
