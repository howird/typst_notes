#import "../styles/things.typ": challenge, hypothesis, question

= DeepPhase

This paper, "DeepPhase: Periodic Autoencoders for Learning Motion Phase
Manifolds," introduces a novel unsupervised neural network architecture to
improve character motion synthesis by learning a structured representation of
motion.

== Overview

=== Challenges

#challenge[
  Poor Spatio-Temporal Alignment & Interpolation
][
  Motion capture data is often sparse, particularly for transitions
  between different movements (e.g., walking to running). When models
  interpolate poses in the raw motion space, the resulting animation often
  appears blurry, smoothed-out, or erratic because spatial similarity of poses
  does not guarantee temporal similarity. This can cause characters to
  stagnate or diverge into unrealistic states.
  #hypothesis[

    The authors hypothesize that all character motion possesses
    local periodicity in both space and time. Even non-cyclic movements (like
    punching or sitting) and transitions can be considered temporal slices of
    periodic motions. Inspired by the central pattern generators in
    neuroscience, they believe motion can be modeled as a composition of
    multiple local periodic signals.
  ]

  To exploit this periodicity, the paper proposes the
  *Periodic Autoencoder*. This network transforms unstructured motion data
  into a multi-dimensional "phase manifold." In this manifold, the distance
  between points becomes a more effective measure of spatio-temporal
  similarity, enabling better alignment and interpolation.

  *Alternative Approaches Mentioned*:
  - *Classic Methods*: Techniques like Motion Graphs or interpolation using
    radial basis functions often require significant manual data alignment and
    preprocessing.
  - *Manual Phase Labeling*: Previous methods like Phase-Functioned Neural
    Networks (PFNN) and Local Motion Phases (LMP) define phase based on
    hand-crafted rules, typically foot-ground contacts. This approach is not
    easily applicable to movements without clear contacts (e.g., dancing,
    stylized arm movements) and requires careful parameter tuning.

]

- *Challenge: Synthesizing High-Quality, Diverse Motion*
  - *Problem*: Neural controllers often struggle to produce vivid, sharp, and
    complex movements. Without proper temporal alignment, they tend to blend
    dissimilar motions at the wrong times, leading to stiff or blurry results.
    This is especially true for motions where different body parts move
    asynchronously with different periodicities (e.g., waving while walking).
  - *Hypothesis*: By feeding the learned phase features into a motion synthesis
    network, the model can better align the data over time. The consistent,
    circular structure of the phase manifold forces the motion to progress
    forward in time, preventing the character from getting stuck and enabling
    smoother, more realistic transitions.
  - *Proposed Approach*: The learned phase manifold features are used as input
    for two downstream tasks:
    1. A neural motion controller (a Weight-Blended Mixture-of-Experts
      framework) that autoregressively predicts the next phase and motion state.
    2. A distance metric for Motion Matching, where the low-dimensional phase
      vector replaces high-dimensional pose features for finding the next best
      motion clip.

=== Periodic Autoencoder Architecture

The core contribution is the *Periodic Autoencoder*, a specialized temporal
convolutional autoencoder.

- *Model Type*: An autoencoder with an inductive bias in the latent space that
  enforces a periodic structure.
- *Input*: A window of 3D joint velocity trajectories, represented as a matrix
  $X in RR^D times N$, where $D$ is the number of degrees of freedom and $N$ is
  the number of frames.
- *Process*:
  1. An *encoder* (convolutional layers) maps the input motion $X$ to a latent
    embedding $L in RR^M times N$, where $M$ is the number of desired phase
    channels.
  2. A *differentiable Fast Fourier Transform (FFT)* layer analyzes each latent
    channel in $L$ to compute periodic parameters: *Amplitude ($A$)*, *Frequency
    ($F$)*, and *Offset ($B$)*.
  3. A separate fully-connected layer computes the *Phase Shift ($S$)*.
  4. These parameters are used to reconstruct a parameterized latent space
    $hat(L)$ using the sinusoidal function:
    $hat(L) = A dot sin(2pi dot (F dot T - S)) + B$.
  5. A *decoder* (deconvolutional layers) reconstructs the original motion
    curves from $hat(L)$.
- *Primary Output*: The trained model is used to extract a feature vector for
  any given motion frame, forming the *Phase Manifold ($P$)*. A point on this
  manifold is a vector in $RR^2M$, computed for each of the $M$ channels as:

$
  P_2i-1^(t) = A_i^(t) dot sin(2pi dot S_i^(t))
$

$
  P_2i^(t) = A_i^(t) dot cos(2pi dot S_i^(t))
$

This manifold is then used as the feature space for downstream animation tasks.

=== Dependencies

The method relies on the following non-novel components:

- *Motion Capture Datasets*:
  - *Quadruped Locomotion*: Data from Zhang et al. 2018.
  - *Stylized Locomotion*: Data from Mason et al. 2022, containing 20 different
    locomotion styles.
  - *Dance Motion*: The AIST++ dataset from Li et al. 2021, which pairs dance
    choreography with music.
  - *Biped and Football*: Internally sourced datasets for general bipedal
    locomotion and football dribbling maneuvers.
- *Foundational Architectures*:
  - *Neural Motion Controller*: The application of the phase manifold for motion
    synthesis is demonstrated using a *Weight-Blended Mixture-of-Experts*
    framework, based on the models from Starke et al. 2020 and Zhang et al.
    2018.

=== Prerequisite Reading

To fully appreciate the paper's contribution, understanding the evolution of
phase-based motion synthesis is beneficial.

- *Holden, D., Komura, T., & Saito, J. (2017).*Phase-functioned neural networks
  for character control*.* This paper introduced the core concept of using a
  single, manually-defined phase variable to align locomotion data, which is the
  foundational idea that *DeepPhase* generalizes to multiple, learned phases
  without manual rules.
- *Starke, S., Zhao, Y., Komura, T., & Zaman, K. (2020).*Local motion phases for
  learning multi-contact character movements*.* This is the most direct
  predecessor, which extended the phase concept to multiple limbs (local phases)
  but still relied on contact events to define them. The current paper directly
  addresses this limitation.

== Problem Formulation

The central goal of the paper is to learn a low-dimensional, periodic
representation of character motion, called a *phase manifold*, from unstructured
motion capture data. This is framed as an unsupervised learning problem where a
neural network must learn to encode and then decode motion sequences, with a
strong inductive bias that forces the latent representation to be periodic.

The model, a *Periodic Autoencoder*, takes a window of motion data as input and
aims to reconstruct it. The key is that the latent space is explicitly
parameterized as a series of sinusoidal functions.

1. *Input Motion*: A window of motion is represented as a matrix
  $X in RR^( D times N )$, where $D$ is the number of degrees of freedom (e.g.,
  $3 times "number of joints"$) and $N$ is the number of time frames in the
  window.

2. *Encoder*: An encoder network $g$ maps the input motion $X$ to a latent
  embedding $L in RR^( M times N )$, where $M$ is the number of desired latent phase
  channels.

$
  L = g(X) quad(1)
$

3. *Latent Space Parameterization*: Each of the $M$ channels in the latent space
  $L$ is assumed to be a periodic signal. This signal is defined by four
  parameters: Amplitude ($A_i$), Frequency ($F_i$), Offset ($B_i$), and Phase
  Shift ($S_i$) for each channel $i in 1, ..., M$. These parameters are
  extracted as follows:
  - A differentiable Fast Fourier Transform (FFT) is applied to each latent
    channel $L_i$ to get the Fourier coefficients $c_i$. The power spectrum
    $p_i$ is then calculated:

$
  p_(i, j) = 2/N abs(c_(i, j))^2 quad(2)
$

- The *Amplitude ($A$)*, *Frequency ($F$)*, and *Offset ($B$)* are derived from
  the power spectrum:

$
  A_i = sqrt(sum_(j = 1)^K p_(i comma j)), quad F_i = (sum_(j = 1)^K (f_j dot.op p_(i, j)))/(sum_(j = 1)^K p_(i, j)), quad B_i = c_(i, 0)/N quad(3)
$

where $K = floor(N/2)$ and $f$ is the vector of frequency bands.
- The *Phase Shift ($S$)* is learned separately by a fully-connected (FC) layer
  that predicts a 2D vector $(s_x, s_y)$ for each channel, from which the phase
  is computed:

$
  S_i = "atan2"(s_y, s_x) quad(4)
$

4. *Latent Space Reconstruction*: The parameterized latent space, $hat(L)$, is
  reconstructed using the extracted parameters and the time vector $T$ for the
  window.

$
  hat(L) = f(T; A, F, B, S) = A dot.op sin(2 pi dot.op(F dot.op T - S)) + B quad(5)
$

5. *Decoder and Loss Function*: A decoder network $h$ takes the parameterized
  latent space $hat(L)$ and reconstructs the output motion curves $Y$.

$
  Y = h(hat(L)) quad(6)
$

The entire network is trained by minimizing the Mean Squared Error (MSE) between
the original input $X$ and the reconstructed output $Y$.

$
  cal(L) = "MSE"(X, Y) quad(7)
$

6. *Phase Manifold Feature*: After training, the final feature representation
  for a given frame at time $t$ is a point on the *phase manifold*
  $P^(t) in RR^2M$. This feature is constructed from the learned amplitude
  $A^(t)$ and phase shift $S^(t)$ for that frame, effectively projecting the
  phase onto a 2D circle scaled by the amplitude for each channel.

$
  P_(2 i - 1)^(t) = A_i^(t) dot.op sin(2 pi dot.op S_i^(t))
$

$
  P_(2 i)^(t) = A_i^(t) dot.op cos(2 pi dot.op S_i^(t)) quad(8)
$

This $P^(t)$ vector is the final, low-dimensional feature used for downstream
tasks.

== Pipeline

This pipeline details the end-to-end process of training the Periodic
Autoencoder and using it for inference to extract phase manifold features.

=== Data Preprocessing

- *Description*: Raw motion capture data is converted into a suitable format for
  the network. This involves calculating joint velocities, normalizing them
  relative to the character's root, and slicing the data into fixed-size
  overlapping windows.
- *Input*: A long sequence of motion capture data (joint positions/rotations).
- *Process*:
  1. Calculate 3D joint velocities for all joints.
  2. Transform velocities into the root's local coordinate system.
  3. Slide a window of length $N$ (e.g., 121 frames) across the entire dataset
    to create samples.
  4. For each window, subtract the window-wise mean to center the curves.
- *Output*: A batch of motion windows $X$.
- *Tensor Shape*: $X in RR^B times D times N$, where $B$ is batch size, $D$ is
  DoF (e.g., $3 times 24$ joints = 72), and $N$ is window length (e.g., 121).

=== Encoder Forward Pass
- *Description*: The batch of motion windows is passed through the convolutional
  encoder to create a latent representation.
- *Input*: Motion window batch $X$.
- *Process*: Pass $X$ through the encoder network $g$, which consists of 1D
  convolutions, batch normalization, and `tanh` activations.
- *Output*: Latent embedding batch $L$.
- *Tensor Shape*: Input: $X in RR^B times D times N$. Output:
  $L in RR^B times M times N$, where $M$ is the number of phase channels (e.g.,
  10).

=== Latent Space Parameterization
- *Description*: The latent embedding $L$ is deconstructed into its constituent
  periodic parameters ($A, F, B, S$). This is the core of the architecture's
  inductive bias.
- *Input*: Latent embedding batch $L$.
- *Process*:
  1. *FFT Path*: Apply a differentiable FFT layer to each of the $M$ channels in
    $L$. Use the resulting power spectrum to calculate the shape parameters
    $A, F, B$ according to *Eq. (3)*.
  2. *FC Path*: Pass the latent embedding $L$ through a separate fully-connected
    layer to predict the 2D vectors for the phase shift. Calculate the final
    phase shift $S$ using the `atan2` function as described in *Eq. (4)*.
- *Output*: Four parameter tensors: $A, F, B, S$.
- *Tensor Shape*: Input: $L in RR^B times M times N$. Output:
  $A, F, B, S in RR^B times M$.

=== Latent Space Reconstruction
- *Description*: A "clean," perfectly periodic version of the latent space is
  reconstructed from the extracted parameters.
- *Input*: The periodic parameters $A, F, B, S$ and a constant time vector $T$.
- *Process*: Use the sinusoidal parameterization from *Eq. (5)* to compute the
  new latent space $hat(L)$.
- *Output*: The parameterized latent space batch $hat(L)$.
- *Tensor Shape*: Input: $A, F, B, S in RR^B times M$. Output:
  $hat(L) in RR^B times M times N$.

=== Decoder Forward Pass & Loss Calculation
- *Description*: The parameterized latent space is decoded back into motion
  curves, and the reconstruction error is calculated.
- *Input*: The parameterized latent space $hat(L)$ and the original motion
  window $X$.
- *Process*:
  1. Pass $hat(L)$ through the decoder network $h$ (1D deconvolutions) to get
    the reconstructed motion $Y$, as per *Eq. (6)*.
  2. Calculate the MSE loss between the original input $X$ and the
    reconstruction $Y$ using *Eq. (7)*.
  3. Perform backpropagation to update the weights of the encoder $g$, decoder
    $h$, and the phase shift FC layer.
- *Output*: The reconstructed motion batch $Y$ and a scalar loss value.
- *Tensor Shape*: Input: $hat(L) in RR^B times M times N$. Output:
  $Y in RR^B times D times N$.

=== Inference and Feature Extraction
- *Description*: After training is complete, the model is used to extract the
  final phase manifold features for any given motion frame.
- *Input*: A window of motion centered on a specific frame $t$.
- *Process*:
  1. Pass the motion window through the trained *encoder* and *parameterization*
    stages (Stages 2 & 3) to obtain the periodic parameters $A^(t)$ and $S^(t)$
    for that frame.
  2. Use these two parameters to compute the final $2M$-dimensional feature
    vector $P^(t)$ for the phase manifold, as defined in *Eq. (8)*.
- *Output*: The phase manifold feature vector $P^(t)$.
- *Tensor Shape*: Input: A motion window of shape $RR^D times N$. Output: A
  feature vector $P^(t) in RR^2M$.

== Discussion

Here is a detailed outline of the primary questions investigated in the paper,
the experiments designed to answer them, and a summary of the results and
limitations.

=== 1. How is the learned phase manifold structured, and is it superior to other embeddings?

This question investigates whether the Periodic Autoencoder successfully imposes
a meaningful and useful structure on raw motion data.

- *Experiment Design*:
  - The authors generated feature embeddings for five different motion datasets
    (Biped, Quadruped, Styles, Dance, Football) using three methods.
    1. *Phase Manifold*: The proposed method's output.
    2. *Convolutional Embedding*: A standard autoencoder without the periodic
      bottleneck, using fully connected layers instead.
    3. *Velocity Embedding*: The raw joint velocity data.
  - They used Principal Component Analysis (PCA) to project these
    high-dimensional embeddings into 2D and 3D spaces for visualization. In the
    plots, poses from the same motion clip are given the same color to show
    temporal connections.

- *Metrics Used*:
  - The evaluation is *qualitative*, based on visual inspection of the resulting
    plots (Figures 6, 7, and 8). The key criteria are the structure, clarity,
    and smoothness of the resulting manifold.

- *Results & Significance*:
  - The *Phase Manifold* consistently organizes motion into a clear,
    polar-coordinate-like structure. The angle around the center represents the
    timing or phase of a motion, while the radius corresponds to the amplitude
    or velocity. Transitions between movements appear as smooth paths between
    cycles of different sizes.
  - In contrast, the *Convolutional Embedding* produces less structured cycles,
    and the *Velocity Embedding* appears chaotic and random.
  - *Significance*: This visually demonstrates that the Periodic Autoencoder's
    inductive bias successfully transforms unstructured motion into a highly
    organized space where spatio-temporal relationships are explicit. This
    structure is far better suited for alignment and interpolation tasks than
    other common embeddings.

- *Limitations*:
  - This analysis is purely qualitative and relies on human interpretation of
    the 2D/3D projections.

=== 2. Does the phase manifold improve neural motion synthesis quality?

This question assesses if using the learned phase features as input to a
character controller leads to better-animated results compared to other methods.

- *Experiment Design*:
  1. *Vividness Test*: The authors trained a neural motion controller with their
    phase features and compared the output against baseline models (MANN, LMP)
    that use other forms of input.
  2. *Foot Skating Test*: They evaluated the amount of foot sliding during
    ground contact for motions generated by their method versus the baselines.
    The baselines were trained with explicit foot contact labels, while the
    proposed method was not.
  3. *Qualitative Comparison*: They generated animations for diverse tasks
    (biped/quadruped locomotion, stylized movements, dance, football dribbling)
    and visually compared them to results from baselines like PFNN, LMP, and
    MoGlow.

- *Metrics Used*:
  - *Average Joint Rotations per Second*: A quantitative metric for "vividness."
    Higher values indicate more dynamic movement and less stiffness or
    blurriness.
  - *Foot Skating Ratio*: The average foot speed during contact, normalized by
    the maximum foot speed observed in the dataset. Lower values are better.
  - *Visual Quality*: A qualitative assessment of the final animation's realism,
    sharpness, and physical plausibility.

- *Results & Significance*:
  - *Vividness*: The proposed method achieved higher "Average Joint Rotations
    per Second" across all tested motion categories, indicating more dynamic and
    less blurry motion (Table 2). For example, it uniquely captured subtle idle
    tail movements on the quadruped that baselines missed entirely.
  - *Foot Skating*: The method produced less foot skating than the baselines on
    most datasets (Table 3). This is significant because it shows the superior
    temporal alignment provided by the phase manifold implicitly corrects for
    artifacts that other methods need explicit labels to address.
  - *Visual Quality*: The generated motions were visibly sharper, especially in
    the upper body and during complex, multi-periodic movements. The system
    could realistically synthesize challenging motions like sharp turns, dancing
    in sync with music, and interactive football dribbling.

- *Limitations*:
  - The paper notes that for dance synthesis, the system doesn't generalize to
    create novel choreography for arbitrary new music; instead, it finds the
    closest match from its training data.

=== 3. Is the phase manifold an effective feature for motion matching?

This question tests the utility of the phase manifold as a low-dimensional
feature for searching a motion database, a core task in motion matching.

- *Experiment Design*:
  1. *Next & Future Frame Search*: They used the feature vector of a current
    frame to search for the *actual* next frame (and subsequent future frames)
    in the database. This was done using three feature types: their *Phase
    Manifold*, a manually selected *Reduced Pose* vector, and the *Full Pose*
    vector. For future frame searches, the phase vector could be predictively
    rotated forward using its learned frequency, a unique capability.
  2. *Pose Alignment Test*: For a given pose, they found the 10 nearest
    neighbors in the database using their learned phase features and compared
    the results to neighbors found using other phase-extraction methods
    (contact-based and heuristic-based).

- *Metrics Used*:
  - *k-NN Query Index*: The average rank of the correct future frame in the list
    of nearest neighbors. A lower index is better, with 1 being a perfect score.
  - *Alignment Error*: The average Euclidean distance between joint pairs of the
    10 matched poses. Lower error means the matched poses are more similar.

- *Results & Significance*:
  - *Search Accuracy*: The *Phase Manifold* found the next frame with a
    significantly lower query index (i.e., it was a much better match) than both
    reduced and full pose vectors, despite having a much lower dimension. Its
    ability to predict future frames was also far more robust, with its query
    index growing very slowly over time while the others failed rapidly (Figure
    15).
  - *Pose Alignment*: Their learned phase features resulted in a much lower
    *Alignment Error* (0.034) compared to contact-based (0.146) and
    heuristic-based (0.074) phases (Table 4).
  - *Significance*: These results show that the phase manifold is a more
    *compact, accurate, and predictive* feature for motion matching than
    traditional pose-based features. It better captures the full-body
    configuration and temporal context, leading to more accurate search results.

=== 4. What are the general limitations of the DeepPhase framework?

This synthesizes the overarching constraints and unresolved issues of the
proposed system.

- *Generalization to Novelty*: The system is excellent at aligning and
  reconstructing motions within the distribution of its training data. However,
  for creative tasks like dancing to new music, it does not have the
  functionality to combine movements in novel ways to match unseen contexts; it
  tends to find the best existing fit.

- *Ambiguity of Control*: The phase manifold resolves ambiguity about *how* to
  transition between poses smoothly, but it does not resolve the ambiguity of
  *which* skill to perform next. The system still requires user control signals
  or a higher-level probabilistic model to decide whether the character should
  walk, run, or jump.

- *Hyperparameter Tuning*: The number of phase channels is a hyperparameter that
  must be chosen based on the variety of motions in the dataset. While the paper
  notes that the system learns to assign different body part coordinations to
  different channels, there is no direct control over this process.
