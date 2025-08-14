= PhysDiff

== Overview

=== Challenges & Solutions
The paper addresses two primary challenges in generative human motion modeling:

- *Challenge 1: Physical Implausibility*
  - Existing motion diffusion models often generate motions with physical artifacts
    like floating, ground penetration, or foot-sliding because they do not account
    for the laws of physics.
  - *Approach*: The proposed solution integrates a *physics-based motion projection
    module* directly into the iterative denoising process of the diffusion model.
    This module uses a pre-trained motion imitation policy within a physics
    simulator to correct the generated motion at specific steps, ensuring it adheres
    to physical constraints.
  - *Hypothesis*: The authors hypothesized that iteratively applying these
    physics-based corrections *during* the diffusion process, rather than just once
    at the end, would be more effective. This iterative process is thought to keep
    the generated motion close to the manifold of physically-plausible motions while
    still staying true to the learned data distribution.
  - *Alternative Solution*: A simpler alternative is to apply the physics-based
    projection as a post-processing step on the final generated motion. However, the
    paper demonstrates that if the initial kinematic motion is too physically
    flawed, a single correction step is insufficient and can even produce
    unnatural-looking results.

- *Challenge 2: Computational Cost of Physics Simulation*
  - The physics-based projection module is computationally expensive, making it
    infeasible to apply at every step of the diffusion process.
  - *Approach*: The paper investigates different *scheduling strategies* to
    determine when and how often to apply the physics projection. The most effective
    strategy was found to be applying a small number of projection steps (e.g.,
    four) consecutively toward the end of the diffusion process.
  - *Hypothesis*: The authors hypothesized that applying physics corrections during
    the early stages of diffusion would be detrimental. In these early steps, the
    noisy input contains little information, and the denoiser tends to output a
    motion close to the dataset's average pose, which is often static and physically
    implausible. Forcing this average pose to be physically correct would push it
    far from the actual data distribution, hindering the diffusion process.

=== Proposed Component: PhysDiff

- *Description*: PhysDiff is a physics-guided motion diffusion model. It is
  designed as a *plug-and-play module* that can be incorporated into the sampling
  (inference) process of various pre-trained kinematic motion diffusion models
  without requiring them to be retrained. The core of the model is a
  *physics-based motion projection* module, denoted $cal(P)_pi$, which enforces
  physical constraints during generation.
- *Inputs*:
  - A conditional input `c`, which can be a text description (e.g., "A person runs
    forward quickly") or a discrete action label (e.g., "Lift Dumbbell").
  - An initial tensor of Gaussian noise, $x_T^1:H$.
- *Outputs*:
  - A physically-plausible human motion sequence, $x^1:H$, represented as a series
    of joint positions or rotations.

=== Dependencies
To reproduce the method, the following non-novel components are required:

- *Datasets*:
  - *HumanML3D*: Used for the text-to-motion generation task. It is a text-annotated
    subset of the AMASS and HumanAct12 datasets.
  - *HumanAct12*: Used for the action-to-motion generation task.
  - *UESTC*: Also used for the action-to-motion generation task.
  - *AMASS*: A large-scale motion capture database used to train the motion
    imitation policy for the physics projection module.
- *Pre-trained Models*:
  - PhysDiff uses existing diffusion models as its core denoiser. The paper
    specifically leverages:
    - *MDM* (Motion Diffusion Model)
    - *MotionDiffuse*
- *Physics & Body Models*:
  - *IsaacGym*: A GPU-based physics simulator used for running the motion imitation
    environments.
  - *SMPL*: A skinned multi-person linear model used to represent the human
    character's body shape and pose.

=== Additional Insights

- *Performance Trade-off*: A key finding, not highlighted in the abstract, is the
  trade-off between physical plausibility and motion quality (as measured by FID).
  While more physics projection steps consistently reduce physical errors (like
  penetration and floating), they can degrade motion quality if too many are used,
  particularly in the early stages of diffusion.
- *Computational Overhead*: The paper is transparent about the inference slowdown.
  Due to the physics simulation, PhysDiff is approximately *1.7x to 2.5x slower*
  than the baseline MDM model, though this gap narrows with larger batch sizes.
- *Assumption of Plausibility*: The method assumes that any motion sequence
  produced by the motion imitation policy within the physics simulator is "physically-plausible".
  This is true in that it obeys the simulator's physics engine, but it is not a
  guarantee that the motion is natural or human-like; that is governed by the
  quality of the learned imitation policy.

=== Recommended Prerequisites

For a deeper understanding of the core concepts leveraged in this paper, the
following areas are recommended:
- *Denoising Diffusion Models*: Foundational knowledge of how denoising diffusion
  probabilistic models (DDPM) and denoising diffusion implicit models (DDIM) work
  is essential.
- *Physics-Based Motion Imitation*: The physics projection module is based on
  learning character control policies with deep reinforcement learning. The paper "DeepMimic"
  is a seminal work in this area and is cited as a key influence.

Here is a detailed outline of the problem formulation and implementation
pipeline for the PhysDiff project.

== Problem Formulation

The primary goal is to generate a physically-plausible human motion, $x^1:H$,
conditioned on an input `c` (like text or an action label). A motion is defined
as a sequence of poses, $x^1:H = x^h_h=1^H$, where each pose $x^h in RR^J times D$ represents
the D-dimensional features of J body joints. The method is built upon the
framework of denoising diffusion models.

=== Denoising Diffusion Model Framework

The generation process is modeled as the reverse of a diffusion process. It
starts with pure Gaussian noise $x_T$ and iteratively denoises it over a series
of timesteps to produce a clean motion sample $x_0$. This process is governed by
solving a stochastic differential equation (SDE):

$
  d x = -(beta_t + dot(sigma)_t) sigma_t nabla_x log p_t (x) d t + sqrt(2 beta_t) sigma_t d omega_t quad(1)
$

The model is trained to learn the score function $nabla_x_t log p_t(x_t)$, which
is directly related to the minimum mean squared error (MMSE) estimator, $tilde(x)$,
of the clean motion given the noisy motion $x_t$:

$
  tilde(x) : = EE [x|x_t ] = x_t + sigma_t^2 nabla_(x_t) log p_t (x_t) quad(2)
$

A denoiser network, $D$, is trained to approximate this MMSE estimator by
minimizing a denoising autoencoder objective:

$
  EE_(x ~ p_0 (x), t ~ p(t), epsilon.alt ~ p(epsilon.alt)) [lambda(t) norm(x - D(x + sigma_t epsilon.alt, t, c))_2^2 ] quad(3)
$

=== Physics-Guided Diffusion Sampling

The key innovation is to inject physical constraints into the sampling process.
Instead of using the direct output of the denoiser, $tilde(x)$, to guide the
next diffusion step, PhysDiff first projects it into a physically-plausible
space using a module $cal(P)_pi$. The projected motion, $hat(x)^1:H = cal(P)_pi(tilde(x)^1:H)$,
is then used in a modified DDIM-like update step. The mean of the next sample, $mu_s$,
is calculated as:

$
  mu_s : = hat(x)^(1 : H) + sqrt(sigma_s^2 - v_s)/sigma_t (x_t^(1 : H) - hat(x)^(1 : H)) quad(4)
$

=== Physics-Based Motion Projection via Imitation

The projection module, $cal(P)_pi$, is realized through a motion imitation
policy trained with reinforcement learning (RL). The policy controls a character
in a physics simulator to mimic a reference motion. The objective is to maximize
a reward function, $r^h$, which measures how closely the simulated motion
matches the reference motion. The reward is a weighted sum of four components:

$
  r^h = w_p r_p^h + w_v r_v^h + w_j r_j^h + w_q r_q^h quad(5)
$

The sub-rewards encourage matching different aspects of the motion:
- *Pose Reward ($r_p^h$)*: Matches local joint rotations.
- *Velocity Reward ($r_v^h$)*: Matches joint velocities.
- *Joint Position Reward ($r_j^h$)*: Matches 3D world joint positions.
- *Global Rotation Reward ($r_q^h$)*: Matches global joint rotations.

== Pipeline

The PhysDiff inference pipeline is an iterative process that refines a noisy
motion into a clean, physically-plausible one. The process uses 50 diffusion
timesteps by default.

=== Initialization

- *Description*: The process begins by creating an initial tensor of random noise
  that has the same dimensions as the final desired motion.
- *Inputs*:
  - A condition `c` (e.g., text prompt).
  - Motion dimensions: `H` (sequence length), `J` (number of joints), `D` (feature
    dimension per joint). For the HumanML3D dataset, this is a single vector of
    dimension 263.
- *Output*:
  - A noisy motion tensor, $x_T^1:H$.
  - *Shape*: `(H, 263)` for HumanML3D, or generally `(H, J, D)`.

=== Denoising Step (Core Diffusion)

- *Description*: In each iteration of the diffusion loop (from timestep $t=T$ down
  to 1), a pre-trained denoiser network predicts the clean version of the motion
  based on the current noisy input. This stage uses classifier-free guidance for
  conditioning.
- *Inputs*:
  - Current noisy motion, $x_t^1:H$. *Shape*: `(H, 263)`.
  - Current timestep, `t`.
  - Condition, `c`.
- *Output*:
  - Denoised motion estimate, $tilde(x)^1:H$. *Shape*: `(H, 263)`.
- *Equation Usage*: This step approximates the MMSE estimator $tilde(x)$ by using
  the trained denoiser network $D$ from *Equation (3)*.

=== Physics-Based Projection (Conditional)
- *Description*: This is the core contribution of PhysDiff. Based on a predefined
  schedule, the denoised motion is passed through a physics-based projection
  module to enforce physical laws. The paper finds that applying this for four
  consecutive steps near the end of the process works well. If not scheduled for
  the current timestep, this stage is skipped.
- *Inputs*:
  - Denoised motion, $tilde(x)^1:H$. *Shape*: `(H, 263)`.
- *Process*:
  1. The motion data (joint positions) is converted to SMPL joint angles via inverse
    kinematics.
  2. An RL agent in the *IsaacGym* physics simulator controls a character to mimic
    this motion. The agent's policy is driven by the reward function defined in
    *Equation (5)* to ensure the resulting motion is both physically valid and close
    to the denoised estimate.
  3. The resulting simulated motion is captured.
- *Output*:
  - Physically-projected motion, $hat(x)^1:H$. *Shape*: `(H, 263)`.

=== Diffusion Update and Sampling
- *Description*: The pipeline uses the (potentially physics-projected) motion to
  compute the parameters for the next, slightly less noisy motion sample. It then
  draws this sample from a Gaussian distribution.
- *Inputs*:
  - Physically-projected motion, $hat(x)^1:H$. *Shape*: `(H, 263)`.
  - Noisy motion from the start of the step, $x_t^1:H$. *Shape*: `(H, 263)`.
  - Timesteps `t` and `s` (where $s < t$).
- *Process*:
  1. Calculate the mean $mu_s$ for the next sample's distribution using the PhysDiff
    update rule from *Equation (4)*. This blends the physically-projected motion
    with the previous noisy motion.
  2. Sample the next motion, $x_s^1:H$, from the distribution $cal(N)(mu_s, v_s I)$.
    The variance is determined by a hyperparameter $eta$, which is set to 0.
- *Output*:
  - The next noisy motion sample, $x_s^1:H$. *Shape*: `(H, 263)`. This becomes the
    input for Stage 2 in the next iteration.

*Stage 5: Final Output*
- *Description*: After the final timestep ($t=0$), the loop terminates, and the
  resulting tensor represents the final generated motion.
- *Input*:
  - The last denoised sample, $x_0^1:H$.
- *Output*:
  - A clean, physically-plausible human motion sequence. *Shape*: `(H, 263)`.

== Discussion

=== Performance against State-of-the-Art
Can PhysDiff achieve state-of-the-art (SOTA) motion quality and physical
plausibility on standard benchmarks?

- *Experiments & Ablations*
  - The model was tested on two primary tasks: *text-to-motion generation* on the
    HumanML3D dataset and *action-to-motion generation* on the HumanAct12 and UESTC
    datasets.
  - To demonstrate its plug-and-play nature, two versions of PhysDiff were created
    by integrating the physics projection module with two different SOTA denoisers:
    *MDM* and *MotionDiffuse (MD)*.
  - These versions were compared against a suite of existing SOTA methods, including
    the original MDM, MotionDiffuse, T2M, and ACTOR.

- *Metrics Used*
  - *Motion Quality & Relevance*:
    - *Fréchet Inception Distance (FID)*: Measures the statistical similarity between
      the distributions of generated and real motions. Lower scores are better.
    - *R-Precision*: Assesses how well the generated motion matches the input text
      description.
    - *Accuracy*: For action-to-motion, this is the accuracy of an action classifier
      on the generated motions.
  - *Physical Plausibility* (all measured in mm, lower is better):
    - *Penetrate*: Measures the depth of ground penetration by body parts.
    - *Float*: Measures the distance of the character from the ground when it should
      be in contact.
    - *Skate*: Measures unnatural foot sliding while a foot is in contact with the
      ground.
    - *Phys-Err*: The sum of the three physics metrics, serving as an overall physical
      error score.

- *Results & Significance*
  - On the HumanML3D text-to-motion task, *PhysDiff w/ MDM* not only achieved a new
    SOTA FID score (0.433) but also reduced the total physical error (`Phys-Err`) by
    over *86%* compared to the baseline MDM.
  - On the action-to-motion tasks, PhysDiff improved the `Phys-Err` by *78%* on
    HumanAct12 and a massive *94%* on UESTC, all while maintaining competitive FID
    and accuracy scores.
  - *Significance*: These results demonstrate that PhysDiff drastically improves
    physical realism without sacrificing, and in some cases even improving, the
    generative quality and diversity of the motion. This validates its effectiveness
    as a general-purpose enhancement for different diffusion models and tasks.

=== Optimal Scheduling for Physics Projections
How do the number and placement of the computationally expensive physics
projection steps affect performance?

- *Experiments & Ablations*
  - *Number of Projections*: The number of physics projection steps was varied from
    0 (no physics) to 50 (physics at every step). These steps were applied
    consecutively at the end of the diffusion process.
  - *Placement of Projections*: For a fixed budget of four projection steps,
    different scheduling strategies were tested:
    1. `Uniform`: Spreading the four steps evenly across the 50 diffusion steps.
    2. `Start/End`: Placing some steps at the beginning and some at the end.
    3. `End`: Placing all four steps at the end of the process, either consecutively (`Space 1`)
      or with gaps in between (`Space 2`, `Space 3`).

- *Metrics Used*: FID, R-Precision, and Phys-Err.

- *Results & Significance*
  - A clear *trade-off* was observed: while more projection steps always improved
    physical plausibility (lower `Phys-Err`), the motion quality metrics (FID,
    R-Precision) first improved and then deteriorated after a certain point. The
    authors found *four steps* provided the best balance.
  - The placement experiments showed that scheduling the projection steps
    *consecutively towards the end* of the diffusion process yielded the best
    results in both motion quality and physical plausibility.
  - *Significance*: This analysis provides crucial practical guidance for using the
    model efficiently. It confirms the hypothesis that applying physics corrections
    during the very noisy, early stages of diffusion is harmful because it can push
    the motion away from the learned data distribution.

=== Comparison with Post-Processing
Does iteratively applying physics *during* diffusion offer a true advantage over
simply using it as a final clean-up step?

- *Experiments & Ablations*
  - A direct comparison was made between PhysDiff and a post-processing baseline.
  - In the baseline, a standard motion diffusion model (MDM) generated a final
    motion, and then the physics projection module was applied one or more times to
    this finished motion.

- *Metrics Used*: FID, R-Precision, and Phys-Err, visualized in Figure 6 of the
  paper.

- *Results & Significance*
  - Simply applying the physics projection as a post-processing step *failed to
    improve the motion*; in fact, it often deteriorated both the physical
    plausibility and the motion quality.
  - *Significance*: This is a critical finding that validates the core design of
    PhysDiff. It proves that there is a *synergy* between the diffusion and physics
    components. The iterative process allows the model to recover from physically
    awkward states by alternating between physics correction and moving closer to
    the data distribution. A final kinematic motion may be too physically
    implausible to be corrected effectively in one go.

=== Limitations and Future Work
What are the primary limitations of the proposed PhysDiff model?

- *Analysis*
  - The main limitation discussed is the computational cost, evaluated by comparing
    inference times.
  - The "Conclusion and Future Work" section explicitly states this limitation and
    suggests directions to mitigate it.

- *Results & Significance (Limitations)*
  - *Inference Speed*: The primary drawback is the increased inference time due to
    the use of a physics simulator. PhysDiff is *1.7x to 2.5x slower* than the
    baseline MDM model when generating a single motion. This performance gap narrows
    with larger batch sizes due to the parallelization capabilities of the physics
    simulator, but it remains a significant overhead.
  - *Future Work*: The authors propose that this limitation could be addressed in
    the future by either using a faster physics simulator or improving the
    efficiency of the projection module to require fewer steps.
