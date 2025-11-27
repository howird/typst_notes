#import "../styles/notes_template.typ": *
#import "@preview/numbly:0.1.0": numbly

#show: note.with(
  title: "Learning to Play Ice Hockey in Physics Simulations",
  author: "Howard Nguyen-Huu",
  foot: none,
  margin: 1in,
  sectionnumbering: numbly(
    "{1:I}",
    "{2}.",
    "{2}.{3}.",
    "{2}.{3}.{4}.",
    "{2}.{3}.{4}.{5:A}.",
  ),
)

= Motivation <motiv>

Human beings possess a remarkable capacity to adapt to diverse and challenging environments. This adaptability is not limited to cognitive challenges, such as solving math problems, but also extends to performing physical tasks such as swimming, riding a bicycle, or ice skating. Understanding the mechanisms by which humans acquire these physical skills is essential to the fields of robotics and computer vision.

Modern advances in machine learning, as well as an increased availability of high-fidelity physics simulations, has enabled the rapid development of capable and robust robotic controllers, which learn to perform complex physical tasks entirely within computer simulations. The class of machine learning algorithms used to optimize a policy (a controller that interacts with an environment) are based on a mathematical framework called Reinforcement Learning (RL). Current state-of-the-art methods extend RL with supervised learning (Imitation Learning), where the policy learns to imitate a dataset of demonstrations, in addition to interacting with its environment. Recent works have demonstrated impressive progress in training physics-based humanoid controllers for general locomotion with Imitation Learning. However, the coordinated full-body movements, rapid reaction times, and precise control required in sports present a critical benchmark for these algorithms.

While some prior works attempt to train humanoid controllers for sports such as soccer and basketball, there is little research dedicated to ice hockey. This sport both exemplifies and _amplifies_ the aforementioned challenges faced when learning to play sports, since players must execute their movements on a low-friction surface while managing frequent interactions with the hockey stick, puck, boards, and opponents. The goal of this project is to address this gap by training control policies for ice skating and extending them to hockey.

#heading(
  level: 1,
  [Objectives: #text(black.lighten(40%), style: "italic")[Overview]],
)

#context [

  To fulfill the project goal, I will implement the following objectives (detailed in #link(locate(<timeline>).position())[III]):

  + #link(
      locate(<obj1>).position(),
    )[
      A physics-simulated hockey rink environment based on the IsaacSim simulation software
    ]
  + #link(
      locate(<obj2>).position(),
    )[A baseline ice skating controller, trained purely with RL within our rink simulation]
  + #link(
      locate(<obj3>).position(),
    )[A dataset of 3D hockey pose demonstrations, generated from hockey broadcast video]
  + #link(
      locate(<obj4>).position(),
    )[A policy trained on our hockey dataset using Imitation Learning, as well as a comparative analysis between it and the aforementioned policy (2.) trained solely with RL]
  + #link(
      locate(<obj5>).position(),
    )[An extension of our skating imitation policy (4.) that is trained to manipulate a hockey stick through weakly-supervised imitation of 2D hockey stick pose]
  // + a *jumping into rabbit hole* on offline/off-policy reinforcement learning and model/representation learning

]


#heading(
  level: 1,
  [Objectives: #text(black.lighten(40%), style: "italic")[Timeline]],
) <timeline>

== Ice Rink Simulation Environment (Dec. 2025) <obj1>

Previously, I have utilized (non-skating) humanoid models and developed prototypes of skating mechanisms for simplified robots (single and double legged robots) using both anisotropic friction in MuJoCo and wheel-based physics in IsaacLab. I will evaluate the environment by testing basic skating motions and verifying physical plausibility through qualitative observation and comparison with reference skating videos. The remaining work involves implementing boundary boards with appropriate collision dynamics, selecting and tuning ice friction parameters, and validating that the humanoid model exhibits realistic skating behavior on the low-friction surface.

== Learning to Skate with Reinforcement Learning (Dec. 2025) <obj2>

Establishing a skating controller trained purely with RL will provides a baseline comparison for the imitation learning approach. I have previously trained humanoid models on standard locomotion tasks using RL. I must now train a skating policy within the ice rink simulation, with reward functions designed to encourage forward skating velocity, balance maintenance, and turning capabilities. I will evaluate the RL-only skating policy against the later imitation-based policy on quantitative metrics including average skating speed, stability (measured by center-of-mass height variance), and successful completion of directional control tasks, as well as qualitative skating realism.

== Generating a 3D Hockey Motion Dataset (Apr. 2025) <obj3>

While most methods in Imitation Learning for humanoid control have trained their models on expensive motion capture datasets, recent advances in monocular pose estimation enable the extraction of 3D human motion from broadcast video. This potentially provides a large corpus of expert demonstrations from professional hockey games.

I have generated a 3D hockey pose dataset from broadcast video using the HMR2 (4DHumans) model. In order to determine the reconstruction accuracy and assessing temporal consistency of tracked poses, I will evaluate them by projecting the 3D pose estimates to 2D, which can then be compared to our ground truth labels. I can further refine the dataset by evaluating more recent pose estimation methods (GVHMR) and tracking models against the existing HMR2-based pipeline and selecting the best-performing approach for final dataset generation.

== Learning to Skate with Imitation Learning (Jan. 2026) <obj4>

I have implemented and validated Imitation Learning methods from previous works in locomotion (AMP), motion imitation (PHC), and basketball (SkillMimic). Next, I must integrate the hockey pose dataset and simulation environment, and train using the methodology of these adjacent works. I will evaluate the imitation learning policy trained on the hockey motion dataset through motion reconstruction error, skating speed and stability metrics, visual comparison with reference hockey footage, and comparative analysis against the RL-only baseline.

== Learning to Manipulate a Hockey Stick (Feb. 2026) <obj5>

Precise manipulation of sports equipment while maintaining balance and coordination represents a significant challenge for physics-based humanoid control. Additionally, imitation learning of stick manipulation would require 3D stick pose which human pose estimation do not produce. I aim to bypass this missing dependency with a weakly-supervised imitation learning framework that trains stick manipulation using our 2D hockey player and stick pose dataset. I hypothesize that 2D pose targets can provide a sufficient training signal to learn stick pose using the reprojection loss prevalent in 3D pose estimation methods. Initially, I plan to qualitatively evaluate  whether the agent can maintain stick contact with the puck and execute basic handling motions. However, quantitative metrics will be designed, potentially including 2D pose alignment error, puck control duration, and successful completion of specific manipulation tasks.

= Summary

