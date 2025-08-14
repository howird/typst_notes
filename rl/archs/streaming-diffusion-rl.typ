= Streaming Fast Diffusion Policy

== Overview

This paper introduces the *Streaming Diffusion Policy (SDP)*, a method to
accelerate action generation in diffusion-based robotic policies, making them
more suitable for real-time control tasks.

== Challenges & Solutions

- *Challenge: Slow inference speed of diffusion models for robotics.*
  - Standard diffusion policies are computationally intensive, requiring many
    iterative denoising steps to generate a clean action trajectory from pure noise.
    This latency limits their application in environments that demand fast, reactive
    control.
  - *Approach*: Instead of generating a fully clean action trajectory at each
    observation, SDP generates a *partially denoised* trajectory where only the
    immediate action is noise-free, and subsequent actions have increasing levels of
    noise. This is managed through a persistent, rolling *action buffer*.
  - *Hypothesis*: The core insight is that the robot only needs the immediate action
    for execution. By reusing the partially denoised future actions from the
    previous step's buffer, a new action can be generated with far fewer denoising
    iterations (e.g., $N/h$ steps instead of $N$).
  - *Alternatives*: The primary alternative discussed is *distillation* (e.g.,
    Consistency Policy). However, distillation can be computationally expensive to
    train, is often unstable, and can degrade the performance and diversity of the
    policy. It also removes useful properties of the original diffusion model, such
    as composability.

== SDP: High-Level Description

- *Component*: A visuomotor policy that modifies the sampling and training of a
  standard diffusion model to enable rapid, recursive action generation.
- *Inputs*:
  - A sequence of observations $O_t$ (e.g., images and robot proprioception).
  - A persistent action buffer $B$ containing a partially denoised action trajectory
    from the prior timestep.
  - A vector of per-action noise levels $k$ corresponding to the buffer.
- *Outputs*:
  - A noise-free chunk of one or more actions $A_t$ to be immediately executed in
    the environment.
  - An updated action buffer for the next timestep, created by removing the executed
    action chunk and appending a new, purely noisy chunk at the end.

== Dependencies

- *Environments & Datasets*:
  - *Push-T Task*: A simulated and real-world benchmark where a robot pushes a
    T-shaped block into a target zone, introduced by Chi et al. (2023) .
  - *Robomimic*: A suite of simulated robotic manipulation tasks from offline human
    demonstrations. The paper uses the `Lift`, `Can`, `Square`, and `Transport` tasks.
- *Foundational Models & Schedulers*:
  - *Diffusion Policy*: The architectural foundation for SDP. The U-Net architecture
    from Diffusion Policy is adapted, and it serves as the primary performance
    baseline.
  - *Consistency Policy*: Used as a key baseline for comparing inference speed and
    performance.
  - *DDPM / DDIM*: The underlying diffusion model frameworks. DDPM is used for
    training and some experiments , while the faster DDIM scheduler is used for
    real-world evaluation.

== Additional Analysis

- *Key Perspectives Beyond the Abstract*:
  - The mechanism for achieving variable noise is a persistent *action buffer* that
    is recursively updated. This buffer is "rolled over" by executing the first
    (clean) action chunk, discarding it, and appending a new chunk of pure Gaussian
    noise to the end of the buffer.
  - Performance is highly dependent on the training scheme. The paper finds that a
    *chunk-wise noise corruption* scheme, which mimics the noise levels used during
    sampling, is critical. The best results were achieved by combining this with
    other schemes (specifically, 80% chunk-wise and 20% constant noise during
    training).

- *Glaring Assumptions*:
  - The policy assumes that a good plan for the next state can be efficiently
    derived from a noisy version of the current plan. This holds for the continuous
    control tasks tested but may be less effective in environments with sharp
    discontinuities where replanning from scratch is necessary. The authors
    acknowledge this as a trade-off: longer buffers increase speed but may reduce
    reactivity.
  - The policy assumes it can recover from a simple, uninformative buffer
    initialization (e.g., all zeros). Experiments show this *"Zero" primer* works as
    well as more complex initializations while being faster and more general.

- *Recommended Prerequisite Reading*:
  - Chi, C., Feng, S., Du, Y., et al. (2023). *"Diffusion policy: Visuomotor policy
    learning via action diffusion."*
    - *Reasoning*: SDP is presented as a direct improvement upon Diffusion Policy.
      Understanding the original model's architecture, training, and sampling is
      essential to fully appreciate the novel contributions of SDP, particularly the
      modified sampling loop and noise scheduling.

== Problem Formulation

The core objective is to learn a visuomotor policy $pi(a_t|O_t)$ that maps a
sequence of past observations $O_t$ to a current action $a_t$. This is framed as
a conditional generative modeling problem, where the policy learns the
distribution of future action sequences conditioned on observations.

*1. Core Variables:*
- *Observation*: $O_t$ is a sequence of the last $T_o$ observations at timestep $t$.
  In the paper, this includes images and proprioceptive sensor data.
- *Action*: $a_t in RR^D_a$ is the action vector of dimension $D_a$ to be executed
  at timestep $t$.
- *Action Sequence (Buffer)*: $A_t = [a_t, a_t+1, ..., a_t+T_a-1]$ is a sequence
  of $T_a$ future actions. This is the central object that the diffusion model
  learns to generate.

*2. Generative Modeling Goal:*
The policy aims to model the conditional distribution $p(A_t|O_t)$. Standard
diffusion policies achieve this by learning to reverse a process that gradually
adds Gaussian noise to a clean action sequence $A_t^0$ until it becomes pure
noise $A_t^N$.

*3. Denoising Process:*
The reversal is done via a learned denoising function $epsilon_theta (A_t^k;O_t, k)$,
which predicts the noise added to a corrupted action sequence $A_t^k$ at noise
level $k$.

- *Standard Diffusion Policy Update:* In a typical diffusion policy, a single
  noise level $k$ is applied to the entire action sequence. The update rule for
  one denoising step is:

$
  A_t^(k - 1) <- alpha_k (A_t^k - gamma_k epsilon.alt_theta (A_t^k ; O_t, k) + cal(N)(0, sigma_k^2)) quad(1)
$

where $alpha_k$, $gamma_k$, and $sigma_k$ are coefficients derived from the
noise schedule.

- *Streaming Diffusion Policy (SDP) Update:* The key innovation in SDP is to
  assign a *separate noise level* $k_i$ to each action $A_t,i$ in the sequence.
  The noise level vector is $k = [k_0, k_1, ..., k_T_a-1]$. The denoising update
  becomes:

$
  A_(t, i)^(k_i - 1) <- alpha_(k_i)(A_(t, i)^(k_i) - gamma_(k_i) epsilon.alt_theta (A_t^k ; O_t, k)_i + cal(N)(0, sigma_(k_i)^2)) quad(2)
$

Note that the network $epsilon_theta$ still processes the entire noisy sequence $A_t^k$ and
the full noise vector $k$ to predict the noise for each component.

*4. Training Objective:*
The model $epsilon_theta$ is trained to minimize the mean squared error between
the predicted noise and the actual noise that was added. For SDP, this loss is
computed over sequences with varying per-action noise levels.

- *SDP Denoising Loss:*

$
  cal(L) = EE_(A_t^0 ~ "data", epsilon.alt ~ cal(N)(0, I), k ~ "dist") [norm(epsilon.alt^k - epsilon.alt_theta (A_t^0 + epsilon.alt^k ; O_t, k))^2 ] quad(8)
$

where $k in ZZ^T_a$ is a vector of noise levels sampled from a specific training
distribution (e.g., chunk-wise increasing, independent, etc.), and $epsilon^k$ is
the corresponding noise added to the clean sequence $A_t^0$.

== Pipeline

The implementation is divided into two phases: training the denoising network
and executing the policy for inference.

=== Training

The goal of this phase is to train the noise prediction network $epsilon_theta$ to
handle variable, per-action noise levels.

- *Stage 1: Data Sampling*
  - *Input*: A dataset of expert demonstrations, consisting of pairs of observation
    sequences and corresponding future action sequences.
  - *Process*: Sample a mini-batch of clean action sequences $A_t^0$ and their
    corresponding conditioning observations $O_t$.
  - *Output*: A batch of clean action sequences.
    - *Shape*: `[batch_size, T_a, D_a]`
  - *Output*: A batch of conditioning observations.
    - *Shape*: Varies based on modality (e.g., `[batch_size, T_o, H, W, C]` for
      images).

- *Stage 2: Trajectory Noise Corruption*
  - *Input*: The batch of clean action sequences $A_t^0$.
  - *Process*:
    1. For each sample in the batch, sample a vector of integer noise levels $k = [k_0, ..., k_T_a-1]$ according
      to a predefined training scheme (e.g., 80% chance of chunk-wise increasing
      noise, 20% chance of constant noise).
    2. Sample a corresponding Gaussian noise tensor $epsilon^k$ where the variance of
      the noise for each action $a_t+i$ is determined by its noise level $k_i$.
    3. Create the noisy action sequence $A_t^k = A_t^0 + epsilon^k$.
  - *Output*: The noisy action sequences $A_t^k$, the ground truth noise $epsilon^k$,
    and the noise level vectors $k$.
    - *Shape (all)*: `[batch_size, T_a, D_a]`, `[batch_size, T_a, D_a]`, `[batch_size, T_a]`.

- *Stage 3: Model Update*
  - *Input*: The noisy action sequences $A_t^k$, ground truth noise $epsilon^k$,
    noise level vectors $k$, and conditioning observations $O_t$.
  - *Process*:
    1. Feed the inputs into the denoising network $epsilon_theta$ to get the predicted
      noise: $hat(epsilon)^k = epsilon_theta (A_t^k;O_t, k)$.
    2. Calculate the loss using *Equation (8)*: $cal(L) = "MSE"(epsilon^k, hat(epsilon)^k)$.
    3. Perform backpropagation and update the weights of $epsilon_theta$.
  - *Output*: A trained denoising network $epsilon_theta$.

=== Inference / Execution

This is the recursive pipeline for generating actions in real-time. The action
sequence is divided into $h$ chunks of size $T_b$, where $T_a = h times T_b$.
The total number of diffusion steps is $N$.

- *Stage 1: Initialization (at t=0)*
  - *Input*: The initial observation from the environment, $O_0$.
  - *Process*:
    1. Initialize the action buffer $B$. The paper finds the *Zero* primer is most
      effective: $B$ is a tensor of zeros.
    2. Initialize the per-action noise level vector $k$. This is set to a chunk-wise
      increasing schedule, e.g., the first chunk has noise level $N/h$, the second has $2N/h$,
      ..., up to $N$ for the last chunk.
    3. Add noise to the zero-initialized buffer $B$ according to the noise levels in $k$.
  - *Output*: The initial noisy action buffer $B$ and the noise level vector $k$.
    - *Shape*: `[T_a, D_a]` and `[T_a]`.

- *Stage 2: Denoising Loop (for each timestep t)*
  - *Input*: The current action buffer $B$, noise level vector $k$, and the current
    observation $O_t$.
  - *Process*: This stage aims to make the *first chunk* of the buffer clean.
    1. Iterate for $N/h$ denoising steps. In each step:
    2. Use *Equation (2)* to update the entire buffer $B$ by calling the trained model $epsilon_theta (B;O_t, k)$.
    3. Decrement every element in the noise level vector $k$ by 1.
  - *Output*: An updated buffer $B'$ where the first chunk is (nearly) noise-free,
    and an updated noise vector $k'$ where the levels for the first chunk are 0.
    - *Shape*: `[T_a, D_a]` and `[T_a]`.

- *Stage 3: Execute and Refresh Buffer*
  - *Input*: The denoised buffer $B'$ and noise vector $k'$.
  - *Process*:
    1. *Execute*: Extract the first, clean chunk of actions from the buffer: $A_t = B'[:T_b]$.
      Send this chunk to the robot controller for open-loop execution.
    2. *Roll Buffer*: Remove the executed chunk from the buffer: $B_"new" = B'[T_b:]$.
    3. *Append Noise*: Create a new chunk of pure Gaussian noise $z ~ cal(N)(0, I)$ of
      size `[T_b, D_a]`. Append this to the end of the buffer: $B <- "concat"(B_"new", z)$.
    4. *Roll Noise Levels*: Update the noise level vector similarly. Remove the first $T_b$ elements
      (which were 0) and append $T_b$ new elements with the max noise level, $N$.
  - *Output*: The action chunk $A_t$ for execution, and the refreshed buffer $B$ and
    noise vector $k$ to be used as input for the next denoising loop at timestep $t+T_b$.
    - *Shape*: `[T_b, D_a]`, `[T_a, D_a]`, `[T_a]`.

== Discussion

=== What is the optimal configuration for SDP?

This question explores the best choices for two key design parameters: the
*action primer* used to initialize the buffer at the start of an episode and the
*noise corruption scheme* used during training.

- *Experiments & Ablations*
  - The authors conducted an ablation study on the state-based *Push-T task* due to
    its lightweight nature, which allows for rapid testing.
  - *Action Primers*: They compared three methods for initializing the action
    buffer:
    1. *Denoise*: Fully denoise an action trajectory from the initial observation.
    2. *Constant*: Fill the buffer by repeating a known initial action.
    3. *Zero*: Initialize the buffer as a zero-tensor.
  - *Noise Schemes*: They evaluated different strategies for applying noise to
    action trajectories during training, including Constant, Linearly increasing,
    Independent, and Chunk-wise increasing variance, as well as several
    combinations.

- *Results & Metrics*
  - *Metrics Used*: `Avg. score` (task success) and `Avg. pred. time` (inference
    latency).
  - *Action Primer Results*: All three primers achieved an identical average score
    of *0.90*. However, the "Denoise" method was significantly slower (0.58s
    prediction time) compared to "Zero" and "Constant" (0.30s) because it requires
    an initial full denoising process.
  - *Noise Scheme Results*: The training scheme had a high impact on performance.
    Simple schemes like Constant and Linear performed poorly in isolation. The best
    performance (*0.90 score*) was achieved with a combination of *80% Chunk-Wise
    and 20% Constant* noise corruption during training.

- *Significance & Conclusions*
  - The *Zero primer* was chosen as the default method because it is fast,
    effective, and more general than the Constant primer, which assumes the first
    action is known.
  - It is crucial that the noise corruption scheme used in training *matches the
    chunk-wise nature of the sampling process* at inference time. This finding
    highlights the importance of aligning the training distribution with the
    test-time distribution for achieving optimal performance.

- *Limitations*
  - These design choices were optimized on a single, state-based task. The authors
    acknowledge that further exploration might reveal different configurations that
    perform better on more complex, vision-based tasks.

=== How much faster is SDP than standard diffusion policies?

This question validates the central claim of the paper: that the streaming
approach significantly reduces prediction time.

- *Experiments & Ablations*
  - The experiment compared the prediction time of *SDP* against a standard
    *Diffusion Policy* on the Push-T task.
  - They systematically increased the *prediction horizon* ($T_a$), which for SDP
    corresponds to increasing the number of chunks ($h$) in the buffer, and measured
    the impact on prediction time and performance.

- *Results & Metrics*
  - *Metrics Used*: `Prediction time [s]` and `Average score`.
  - *Prediction Time Results*: SDP's prediction time *decreased monotonically* as
    the horizon grew, dropping from ~1.0s to ~0.1s. In contrast, the standard
    Diffusion Policy's prediction time remained *constant* at a high level (~1.15s).
  - *Performance Results*: Throughout the experiment, SDP's average score remained
    stable and comparable to the baseline Diffusion Policy, indicating that the
    speedup did not come at the cost of performance.

- *Significance & Conclusions*
  - The results confirm that SDP's sampling speedup is inversely proportional to the
    number of chunks ($h$) in the buffer, as the number of required denoising steps
    is reduced to $N/h$.
  - This demonstrates that the streaming approach is an effective method for
    dramatically accelerating diffusion-based policies while preserving their high
    performance.

- *Limitations*
  - The paper explicitly notes a *trade-off between speed and reactivity*. While a
    longer buffer (more chunks) increases sampling speed, an excessively long buffer
    could negatively impact performance on tasks that require quick reactions to
    changes in the environment, as the policy might be acting on stale information.

=== How does SDP compare to other SOTA methods?

This question benchmarks SDP against relevant baselines to show its
competitiveness on challenging simulated and real-world tasks.

- *Experiments & Ablations*
  - SDP was benchmarked against *Diffusion Policy* (a high-performance baseline) and
    *Consistency Policy* (a fast, distillation-based baseline).
  - *Simulated Tasks*: The comparison was run on five image-based tasks: Push-T and
    the Robomimic tasks Lift, Can, Square, and Transport.
  - *Real-World Task*: All three methods were trained on 134 human demonstrations of
    a real-world Push-T task and evaluated on a physical robot.

- *Results & Metrics*
  - *Metrics Used*: Max/average performance scores for simulation; `Coverage success` and `Sampling time` for
    the real-world task.
  - *Simulation Results*: SDP achieved performance *comparable to Diffusion Policy*
    across all tasks. Both methods were generally more performant than Consistency
    Policy, especially on complex, long-horizon tasks like Transport.
  - *Real-World Results*: SDP achieved the *highest success rate (0.85)*,
    significantly outperforming Diffusion Policy (0.50) and Consistency Policy
    (0.40). Its sampling time (0.07s) was faster than Diffusion Policy (0.13s) but
    slower than the highly distilled Consistency Policy (0.01s).

- *Significance & Conclusions*
  - SDP provides an excellent balance of high performance and fast inference, making
    it a powerful and practical choice for robotic control.
  - The authors observed that in the real world, SDP was less prone to getting stuck
    in small, oscillatory movements—a common failure mode for the other policies.
    They hypothesize this is because SDP's *persistent action buffer* helps maintain
    momentum across timesteps, making it more robust than policies that constantly
    re-plan from scratch.

- *Limitations*
  - During real-world rollouts, the robot's movement was *halted while the policy
    sampled* the next action. Although the sampling time is short, this synchronous
    deployment is not fully continuous.
  - While SDP proved more robust against repetitive oscillatory motions, the paper
    notes that *completely alleviating this issue* remains an interesting avenue for
    future research.
