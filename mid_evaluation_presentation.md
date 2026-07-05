# Progress Presentation: A DRL Approach for Proactive Load Balancing Controller Management in SDN

Student: M. Logadharsan  
Index No: 2021/SP/098  
Course: CSC406S6 / CSC406M6  
Supervisor: Dr. K. Thabotharan  
Presentation date: 11 May 2026

---

## Slide 1: Title

**A DRL Approach for Proactive Load Balancing Controller Management in SDN**

- M. Logadharsan
- 2021/SP/098
- Department of Computer Science, University of Jaffna
- Mid Evaluation / Progress Presentation

**Speaker note:**  
Good morning. My research focuses on using Deep Reinforcement Learning to proactively manage controller load balancing and energy efficiency in distributed Software-Defined Networks.

---

## Slide 2: Introduction and Background

- Software-Defined Networking separates the control plane from the data plane.
- A single SDN controller can become a bottleneck and a single point of failure.
- Distributed multi-controller SDN improves scalability and reliability.
- However, distributed controllers introduce new problems:
  - controller load imbalance
  - increased controller-switch latency
  - unnecessary energy usage when all controllers remain active

**Speaker note:**  
SDN makes network management flexible, but as the network grows, one controller is not enough. Multi-controller SDN solves scalability, but the workload may become uneven, and keeping every controller active wastes energy during low traffic periods.

---

## Slide 3: Problem Statement

**Problem:**  
Existing SDN controller load balancing methods are often reactive and rule-based. They usually respond after overload occurs and may keep unnecessary controllers active, causing latency, uneven load distribution, and energy waste.

**This research addresses:**

- how to balance controller load dynamically
- how to reduce controller-switch latency
- how to save energy by parking underutilized controllers
- how to evoke parked controllers before overload seriously affects performance

**Speaker note:**  
The main limitation is that traditional methods react only after a threshold is crossed. My project tries to make the control plane more adaptive by allowing a learning agent to decide when to migrate switches, park controllers, evoke controllers, or do nothing.

---

## Slide 4: Research Objectives

- Design a master-slave-parked controller architecture for distributed SDN.
- Develop a DRL agent using Rainbow DQN / RDQN for proactive controller management.
- Formulate the problem as an MDP with state, action, and reward definitions.
- Evaluate the approach using real network topologies and generated traffic traces.
- Measure performance using latency, energy consumption, load balance, and proactive action behavior.

**Speaker note:**  
The main objective is not only load balancing. It is multi-objective controller management: latency reduction, energy efficiency, and stable controller utilization.

---

## Slide 5: Literature Review

| Paper | Main Focus | Method / Contribution | Limitation / Gap Used in My Work |
|---|---|---|---|
| Chen et al., 2025, **MOOO-RDQN** | Controller placement and traffic monitoring in SDN | Rainbow DQN for multi-objective optimization | Focuses mainly on placement and monitoring; my work extends the idea toward controller parking, evoking, and switch migration |
| Liu et al., 2025 | Controller load optimization survey | Reviews switch migration, controller placement, and load optimization | Shows that dynamic load imbalance remains a key challenge |
| Li et al., 2024 | SDN controller placement in MEC | CNN-LSTM traffic prediction + DRL controller placement | MEC-focused; does not directly manage parked controllers |
| Yeo et al., 2021, **SAR-LB** | RL-based switch migration in distributed SDN controllers | Uses switch-aware reinforcement learning and controller/switch resource utilization to improve load distribution | Focuses on switch migration only; my work adds parked controller evoking/parking and RDQN-based multi-objective control |
| Das et al., 2020 | Controller Placement Problem survey | Surveys CPP objectives: latency, reliability, energy, load balance | Mostly discusses placement; my work focuses on runtime controller management |
| Neghabi et al., 2018 | SDN load balancing review | Systematic review of load balancing mechanisms | Identifies need for adaptive and intelligent balancing approaches |
| Semong et al., 2020 | Intelligent load balancing in SDN | Survey of intelligent load balancing methods | Many methods remain simulation-based or reactive |

**Speaker note:**  
I used MOOO-RDQN as the closest algorithmic inspiration because it uses Rainbow DQN for SDN multi-objective optimization. I also used Yeo et al.'s SAR-LB paper because it directly supports reinforcement learning for switch migration. However, my research problem is different: I focus on runtime proactive controller management using parking, evoking, and switch migration.

---

## Slide 6: Research Gap

- Static controller placement alone cannot handle dynamic traffic changes.
- Threshold-based switch migration is usually reactive.
- Energy-aware sleeping or parking methods need safe activation decisions.
- Standard DQN may suffer from overestimation, unstable learning, and inefficient exploration.
- There is a need for a proactive DRL framework that jointly considers:
  - load balancing
  - latency
  - energy consumption
  - controller state management

**Speaker note:**  
The gap is the lack of a learning-based controller management method that can decide both load balancing actions and energy-saving actions in the same framework.

---

## Slide 7: Proposed System Architecture

**Architecture: Master-Slave-Parked Controller Model**

- **Master controller:** monitors global controller load and coordinates decisions.
- **Slave controllers:** actively manage assigned switches.
- **Parked controllers:** low-power standby controllers.
- **DRL agent:** selects proactive control actions based on network state.

**Workflow:**

Network topology and traffic -> SDN environment -> state observation -> RDQN agent -> action -> reward -> updated policy

**Speaker note:**  
The architecture contains active slave controllers and parked controllers. During low traffic, underloaded controllers can be parked to save energy. During overload, parked controllers can be evoked.

---

## Slide 8: MDP Formulation

**State space:**

- controller loads
- active or parked status of each controller
- switch-to-controller assignments

**Action space:**

- no operation
- migrate a switch from an overloaded controller
- evoke a parked controller
- park an underloaded controller

**Reward function:**

- penalizes high latency, high energy, and load variance
- rewards useful park/evoke actions only when context requires them
- penalizes no-action during overload or parkable underload
- includes anti-oscillation penalty to reduce rapid park/evoke flipping

**Speaker note:**  
The reward is context-aware. Parking is rewarded only when the network is underloaded, and evoking is rewarded when there is overload. This avoids unnecessary controller state changes.

---

## Slide 9: Algorithm Used: Rainbow DQN / RDQN

The implemented RDQN combines the main Rainbow DQN components:

- Double Q-learning: reduces Q-value overestimation.
- Prioritized experience replay: learns more from important transitions.
- Dueling network: separates state value and action advantage.
- Multi-step learning: uses n-step returns.
- Noisy networks: improves exploration.
- Distributional RL / C51: learns a distribution of possible returns.

**Implementation detail from codebase:**

- Model file: `environments/rainbow_dqn_model.py`
- Training script: `experiments/train_rainbow_fixed_rewards.py`
- Evaluation script: `experiments/evaluate_threshold_model.py`

**Speaker note:**  
I am using RDQN because the controller management problem has a large and changing action space. Rainbow components improve stability, exploration, and sample efficiency compared with ordinary DQN.

---

## Slide 10: Experimental Setup

**Topologies used from Topology Zoo:**

| Topology | Nodes | Edges |
|---|---:|---:|
| Gridnet | 9 | 20 |
| BellCanada | 48 | 64 |
| OS3E | 34 | 36 |
| Interoute | 110 | 146 |
| Cogentco | 197 | 243 |

**Traffic and environment:**

- Processed traffic traces stored under `data/traffic/processed/`
- Controller-switch delays computed using topology shortest paths
- 1 master controller, 3 active slave controllers, 2 parked controllers
- Thresholds: underload below 30%, overload above 70%

**Speaker note:**  
I evaluated the system across small, medium, and large topologies to test scalability and behavior under different network sizes.

---

## Slide 11: Implementation Progress

**Completed so far:**

- Research proposal completed.
- Literature review completed with controller placement, load balancing, and DRL papers.
- Real topology loader implemented for five network topologies.
- Traffic preprocessing and topology-specific traffic files prepared.
- Threshold-based proactive SDN environment implemented.
- Rainbow DQN / RDQN model implemented.
- Context-aware reward function added.
- Invalid action masking added.
- Training completed for five topologies.
- Initial real-test JSON results saved.

**Speaker note:**  
The current implementation has moved beyond only environment setup. I have trained topology-specific RDQN models and saved both model checkpoints and result summaries.

---

## Slide 12: Training Results

| Topology | Steps | Reward Last 100 | CS Avg Latency | Worst Latency | Energy | Load Balance |
|---|---:|---:|---:|---:|---:|---:|
| Gridnet | 30,000 | 25,111.86 | 10.08 ms | 21.37 ms | 421.41 W | 1.202 |
| BellCanada | 100,000 | 14,219.01 | 12.24 ms | 38.05 ms | 455.83 W | 2.028 |
| OS3E | 80,000 | 10,727.17 | 5.41 ms | 18.14 ms | 445.85 W | 1.627 |
| Interoute | 150,000 | 5,122.87 | 5.99 ms | 19.28 ms | 453.08 W | 1.582 |
| Cogentco | 200,000 | -28,013.38 | 19.65 ms | 59.28 ms | 453.21 W | 1.850 |

**Speaker note:**  
The model shows positive training performance for Gridnet, BellCanada, OS3E, and Interoute. Cogentco is the most difficult case because it is the largest topology with 197 nodes and a much larger action space.

---

## Slide 13: Proactive Action Behavior

| Topology | Parking Events | Evoking Events | Migration Events | Observation |
|---|---:|---:|---:|---|
| Gridnet | 12,407 | 12,428 | 1,744 | frequent park/evoke; small topology |
| BellCanada | 26,986 | 27,092 | 21,995 | strong proactive behavior |
| OS3E | 22,600 | 22,670 | 18,282 | balanced park/evoke behavior |
| Interoute | 29,604 | 29,749 | 21,475 | stable learning in larger topology |
| Cogentco | 31,627 | 31,818 | 29,029 | active but still reward-challenged |

**Speaker note:**  
The agent has learned to use all three important action categories: parking, evoking, and migrating. The park/evoke counts are close, meaning the model is dynamically scaling controller resources based on traffic conditions.

---

## Slide 14: Real-Test Preliminary Results

| Topology | CS Avg Latency | Worst Latency | Energy | Load Balance Index | Parking | Evoking |
|---|---:|---:|---:|---:|---:|---:|
| Gridnet | 8.44 ms | 12.85 ms | 415.36 W | 1.151 | 2,500 | 2,500 |
| BellCanada | 11.37 ms | 25.87 ms | 495.89 W | 2.254 | 1,000 | 1,012 |
| OS3E | 3.12 ms | 11.51 ms | 442.18 W | 1.595 | 1,000 | 1,007 |
| Interoute | 5.26 ms | 16.60 ms | 485.65 W | 1.523 | 1,621 | 1,635 |
| Cogentco | 9.69 ms | 39.99 ms | 415.00 W | 1.744 | 2,500 | 2,500 |

**Speaker note:**  
The preliminary real-test results show acceptable latency and active controller state management. However, reward values are still unstable in some tests, so further reward tuning and baseline comparison are required before final conclusions.

---

## Slide 15: Key Findings So Far

- RDQN can learn proactive controller management actions across different topology sizes.
- Context-aware reward shaping is necessary; simple reward bonuses can cause unnecessary park/evoke actions.
- Invalid action masking is important because many actions are not valid in every network state.
- Smaller and medium topologies show stronger reward behavior.
- Large topology performance, especially Cogentco, needs more tuning and possibly longer training.

**Speaker note:**  
The main technical lesson is that the action space must be controlled carefully. Without masking and context-aware reward logic, the agent may learn invalid or unstable behavior.

---

## Slide 16: Challenges

- Large action space for large topologies.
- Reward instability in some evaluations.
- Balancing latency, energy, and load objectives is difficult.
- Preventing rapid park/evoke oscillation.
- Need for stronger comparison with baselines such as random, threshold-based, and ordinary DQN.

**Speaker note:**  
These challenges are normal for DRL-based network control. The next stage is mainly about improving evaluation fairness and comparing the proposed RDQN approach with baseline methods.

---

## Slide 17: Next Steps

- Run final baseline comparisons:
  - random policy
  - threshold-based policy
  - ordinary DQN / ablation if possible
- Tune reward weights for Cogentco and other unstable cases.
- Add graphs:
  - reward curve
  - latency comparison
  - energy comparison
  - load balance comparison
  - action distribution
- Prepare final analysis and discussion.
- Complete final report writing.

**Speaker note:**  
My immediate next step is to strengthen the evaluation section. I need to compare RDQN against simple baseline policies and present the improvements clearly using graphs.

---

## Slide 18: Conclusion

- This research proposes a DRL-based proactive controller management approach for SDN.
- A master-slave-parked controller architecture has been implemented.
- RDQN is used to learn migrate, park, evoke, and no-action decisions.
- Training has been completed on five real-world topologies.
- Preliminary results show promising proactive behavior, while further tuning and baseline comparison are ongoing.

**Speaker note:**  
In conclusion, the project has reached the implementation and preliminary evaluation stage. The current results show that RDQN can control active and parked controllers, and the next phase will focus on stronger evaluation and comparison.

---

## Slide 19: Thank You

**Thank you.**  
Questions?

---

# Detailed Literature Review Notes

Use this section for preparation. You do not need to put every detail on the slides. For the presentation, show the summary table; for Q&A, use these explanations.

## Paper 1: Chen et al., 2025

**Paper name:**  
MOOO-RDQN: A Deep Reinforcement Learning Based Method for Multi-Objective Optimization of Controller Placement and Traffic Monitoring in SDN

**Main problem addressed:**  
The paper addresses the SDN Controller Placement Problem (CPP), where the goal is to decide how many controllers to deploy, where to place them, and how switches should be assigned to controllers.

**Contribution:**  

- Proposes MOOO-RDQN, a Multi-Objective Optimization Oriented Rainbow Deep Q Network.
- Optimizes controller-switch average latency, worst-case latency, load balance, and training time.
- Combines DRL-based controller placement with an FPGA-based traffic monitoring module for overload detection.
- Shows that Rainbow DQN can outperform normal DQN variants and several heuristic/meta-heuristic methods.

**Methodology:**  

- Models controller placement as a reinforcement learning problem.
- Uses Rainbow DQN improvements such as Double Q-learning, Prioritized Experience Replay, Dueling Network, Multi-step Learning, and Noisy Networks.
- Defines multi-objective reward using latency and load balance.
- Compares MOOO-RDQN with DQN, Dueling DQN, Double DQN, K-means, GOKA, PSA, MCPA, and CCA_PSO.

**Dataset / topologies used:**  

- Uses real network topologies from Internet Topology Zoo and Internet2 OS3E.
- Topologies include Gridnet, BellCanada, OS3E, Interoute, and Cogentco.
- Uses CIC-Bell-DNS 2021 as the traffic dataset.

**Key metrics:**  

- Controller-switch average latency
- Controller-switch worst-case latency
- Load balance index
- Training time
- Optimality gap from brute-force/global optimum

**Research gap / limitation:**  

- Mainly focuses on controller placement and switch-controller mapping.
- Does not focus on runtime parked-controller management.
- Does not directly handle controller parking and evoking for energy-aware controller state control.

**Connection to my research:**  

This is the closest algorithmic inspiration for my work because I also use RDQN/Rainbow DQN and similar topologies. However, my work applies RDQN to proactive runtime controller management, where the agent decides migrate, park, evoke, or no-action.

## Paper 2: Yeo et al., 2021

**Paper name:**  
Achieving Balanced Load Distribution with Reinforcement Learning-Based Switch Migration in Distributed SDN Controllers

**Main problem addressed:**  
The paper addresses uneven load distribution among distributed SDN controllers caused by static switch-controller mapping.

**Contribution:**  

- Proposes SAR-LB, a Switch-Aware Reinforcement Learning Load Balancing scheme.
- Uses switch migration to balance loads among distributed SDN controllers.
- Considers resource utilization of both controllers and switches.
- Reduces action space by treating switches as RL agents.
- Reports up to 34% improvement in normalized standard deviation of controller load distribution.

**Methodology:**  

- Uses reinforcement learning with a neural network for switch and controller selection.
- Input state includes resource utilization ratios such as CPU, RAM, and network-related resource usage.
- Selects target controller for migration.
- Compares with Round Robin, ElastiCon, and MARVEL.
- Evaluates under normal and bursty traffic conditions.

**Dataset / environment used:**  

- Uses a simulated SDN environment.
- Evaluates five scenarios with different numbers of switches and controllers.
- Uses stochastic/dynamic workload generation, including bursty traffic cases.

**Key metrics:**  

- Standard deviation of workload distribution
- Normalized standard deviation
- Decision-making time per time step
- Load balancing behavior under bursty and non-bursty traffic

**Research gap / limitation:**  

- Focuses mainly on switch migration.
- Does not include controller parking or evoking.
- Energy-aware controller state management is not the main focus.
- Performance becomes weaker when the number of controllers increases because state/action dimensions increase.

**Connection to my research:**  

This paper supports the switch migration part of my work. My work extends this idea by combining switch migration with controller parking and evoking using RDQN.

## Paper 3: Liu et al., 2025

**Paper name:**  
Controller Load Optimization Strategies in Software-Defined Networking: A Survey

**Main problem addressed:**  
The paper surveys controller load optimization strategies in SDN, mainly focusing on switch migration and controller placement.

**Contribution:**  

- Provides a comprehensive review of SDN controller load optimization strategies.
- Classifies switch migration schemes into static threshold, dynamic threshold, passive migration, and proactive migration methods.
- Classifies controller placement schemes into optimal and sub-optimal methods.
- Discusses load optimization metrics and future research challenges.

**Methodology:**  

- Systematic literature review.
- Uses research questions to study load optimization strategies, metrics, and future challenges.
- Reviews 61 relevant papers from 2017 to 2024.

**Dataset / environment used:**  

- No new dataset is used because it is a survey paper.
- It reviews datasets, topologies, and experimental environments used by previous studies.

**Key metrics discussed:**  

- Controller load
- Response time
- Number of migrated switches
- Migration cost
- Migration frequency
- Execution time
- Latency and load balance metrics

**Research gap / limitation:**  

- Current load optimization approaches still face challenges in real-time migration, seamless migration, adaptive controller placement, energy-aware optimization, and intelligent migration.
- Many existing methods are reactive and depend on thresholds.

**Connection to my research:**  

This paper justifies my problem statement. It shows that controller load optimization is still an important issue and that future work should include intelligent, adaptive, and energy-aware techniques.

## Paper 4: Li et al., 2024

**Paper name:**  
Deep Reinforcement Learning Based Controller Placement and Optimal Edge Selection in SDN-Based Multi-Access Edge Computing Environments

**Main problem addressed:**  
The paper addresses controller placement and edge selection in SDN-assisted MEC environments.

**Contribution:**  

- Proposes an SDN-assisted MEC architecture.
- Uses CNN-LSTM to predict network traffic.
- Uses DRL to solve controller placement while considering load balance and system cost.
- Uses Lyapunov optimization for dynamic edge selection.
- Reduces average communication delay, energy consumption, and improves SDN controller utilization.

**Methodology:**  

- CNN-LSTM is used for network traffic prediction.
- The controller placement problem is formulated as minimizing total communication cost while maintaining load balance.
- DRL is used to select controller placement.
- Edge selection is modeled using queuing theory and solved using Lyapunov optimization.

**Dataset / topologies used:**  

- Uses Topology Zoo networks such as Goodnet, OS3E, UUNET, Uninett2010, and TATA.
- Uses Mininet-WiFi and POX controllers in the experiment.
- Uses CIFAR-10 for image-recognition tasks in the MEC edge selection experiment.

**Key metrics:**  

- Number of controllers
- Average delay
- Control plane utilization
- Load balancing degree
- Energy consumption
- Task completion ratio

**Research gap / limitation:**  

- The focus is SDN-MEC, not general SDN controller parking/evoking.
- It solves controller placement and edge selection, but not proactive parked-controller state management.
- CNN-LSTM traffic prediction can have higher computational complexity and needs offline training.

**Connection to my research:**  

This paper supports the idea that DRL can be used for SDN controller placement and load-aware management. My work is different because I focus on runtime proactive controller management instead of MEC edge selection.

## Paper 5: Das et al., 2020

**Paper name:**  
A Survey on Controller Placement in SDN

**Main problem addressed:**  
The paper surveys the Controller Placement Problem in SDN and explains how controller placement affects latency, resiliency, energy efficiency, load balancing, and scalability.

**Contribution:**  

- Defines and explains the Controller Placement Problem.
- Discusses CPP modeling choices and evaluation metrics.
- Classifies CPP literature based on objectives and methodologies.
- Covers objectives such as latency, reliability, energy efficiency, load balancing, QoS, and multi-objective placement.
- Discusses open issues and future research directions.

**Methodology:**  

- Comprehensive survey of controller placement studies.
- Categorizes methods into optimal and heuristic/sub-optimal solutions.
- Discusses methods such as ILP, graph theory, clustering, greedy algorithms, game theory, approximation, and meta-heuristics.

**Dataset / environment used:**  

- No new dataset is used because it is a survey.
- The reviewed works often use Internet2 OS3E, Internet Topology Zoo, SNDlib, random topologies, and synthetic topologies.

**Key metrics discussed:**  

- Average switch-controller latency
- Worst-case switch-controller latency
- Number of controllers
- Controller load
- Reliability/resilience
- Energy/cost
- Flow setup time

**Research gap / limitation:**  

- Many CPP methods are static and focus on initial placement.
- Runtime adaptation, dynamic traffic changes, and ML-based adaptive control need more research.
- Placement alone cannot solve all load imbalance problems during network operation.

**Connection to my research:**  

This paper supports the background of my project. It shows why controller placement and switch-controller mapping are important, but my work focuses on runtime controller management after deployment.

# Overall Literature Gap Identified

From these papers, the main gap is:

- Controller placement is important, but static placement alone cannot handle dynamic traffic changes.
- Switch migration can balance controller load, but it does not directly save controller energy.
- Existing RL-based switch migration mainly focuses on migration, not parking/evoking controllers.
- Many methods are reactive or threshold-based.
- There is a need for an intelligent DRL-based framework that combines load balancing, latency awareness, and energy-aware controller state management.

**My research direction:**  
I use RDQN/Rainbow DQN for proactive controller management in SDN. The agent learns when to migrate switches, park underloaded controllers, evoke parked controllers, or take no action.

---

# Likely Q&A Preparation

## Q1: What is the main difference between your work and MOOO-RDQN?

MOOO-RDQN focuses on controller placement and switch-controller mapping. My work focuses on runtime proactive controller management using switch migration, controller parking, and controller evoking for load balancing and energy efficiency.

## Q2: Why did you use Rainbow DQN instead of normal DQN?

Normal DQN can suffer from unstable learning, overestimated Q-values, and inefficient exploration. Rainbow DQN improves this using double Q-learning, prioritized replay, dueling networks, multi-step learning, noisy exploration, and distributional learning.

## Q3: What are your state, action, and reward?

The state includes controller loads, controller active/parked status, and switch-controller mappings. Actions are no-action, switch migration, evoking a parked controller, and parking an underloaded controller. The reward penalizes latency, energy, and imbalance, while rewarding useful park/evoke/migrate decisions.

## Q4: What are your main results so far?

I trained RDQN models on Gridnet, BellCanada, OS3E, Interoute, and Cogentco. Four topologies achieved positive final training rewards, and all topologies show active proactive behavior through parking, evoking, and migration events.

## Q5: Why is Cogentco performance weaker?

Cogentco has 197 nodes and the largest action space in the experiment. This makes exploration and convergence harder, so it likely needs longer training, reward tuning, or stronger action abstraction.

## Q6: What are your evaluation metrics?

The main metrics are controller-switch average latency, worst-case latency, energy consumption, load balance index, reward, and proactive action counts.

## Q7: Is this fully real SDN deployment?

At the current stage, it is a simulation-based DRL environment using real topology files and processed traffic traces. Full Mininet/Ryu integration can be treated as future implementation work or final-stage extension.

## Q8: What is your contribution?

The contribution is a proactive RDQN-based controller management framework that combines load balancing and energy-aware controller state decisions through migration, parking, and evoking actions.

## Q9: Why did you select these literature review papers?

I selected them because they cover the main areas of my research: controller placement, switch migration, controller load optimization, reinforcement learning, and energy-aware SDN management. Together, they show the research gap that my work addresses.

## Q10: Which paper is closest to your research?

The closest paper is Chen et al.'s MOOO-RDQN paper because it uses Rainbow DQN for multi-objective optimization in SDN. However, my work is different because I focus on proactive controller management with parking and evoking controllers, not only controller placement.

## Q11: What did you learn from the MOOO-RDQN paper?

I learned that Rainbow DQN can be useful for SDN problems with multiple objectives such as latency and load balancing. It also showed that enhanced DQN methods can perform better than ordinary DQN or heuristic algorithms in complex SDN decision-making.

## Q12: How is your work different from MOOO-RDQN?

MOOO-RDQN focuses on selecting controller locations and switch-controller mappings. My work assumes a distributed controller architecture and manages controller states dynamically. My actions include switch migration, controller parking, controller evoking, and no action.

## Q13: What is the importance of the SAR-LB paper by Yeo et al.?

SAR-LB is important because it shows that reinforcement learning can be used for switch migration in distributed SDN controllers. It supports my idea that learning-based methods can make better migration decisions than simple static or greedy methods.

## Q14: How is your work different from SAR-LB?

SAR-LB mainly focuses on selecting switches and target controllers for migration. My work includes migration, but also adds energy-aware controller parking and evoking. Therefore, my work handles both load balancing and controller energy management.

## Q15: Why are survey papers included in the literature review?

Survey papers are useful because they summarize many existing methods and clearly show the open problems. For example, the SDN controller placement and load optimization surveys helped me identify that reactive methods and static placement are not enough for dynamic traffic conditions.

## Q16: What research gap did you identify from the literature?

The main gap is that many existing methods handle only one part of the problem, such as controller placement or switch migration. Fewer methods combine load balancing, latency reduction, and energy-aware controller state management in one proactive DRL framework.

## Q17: What are the limitations of traditional controller placement methods?

Traditional controller placement methods decide where controllers should be placed, but they may not adapt well after the network traffic changes. Even with good placement, some controllers can become overloaded while others are underused during runtime.

## Q18: What are the limitations of threshold-based load balancing?

Threshold-based methods are simple and easy to implement, but they are usually reactive. They act only after the controller load crosses a fixed threshold, so they may respond late and may not learn from previous traffic patterns.

## Q19: Why is switch migration important in SDN load balancing?

Switch migration changes the controller responsible for a switch. If one controller is overloaded, migrating some switches to less loaded controllers can reduce imbalance and improve controller response time.

## Q20: What is the limitation of switch migration alone?

Switch migration can balance load, but it does not directly save energy. During low traffic, all controllers may still remain active. That is why my work also includes parking underloaded controllers and evoking them again when traffic increases.

## Q21: What is meant by energy-aware controller management?

Energy-aware controller management means not keeping every controller fully active all the time. In my work, underloaded controllers can be moved into a parked low-power state, and they can be evoked again when the network load increases.

## Q22: Why is DRL suitable for this research problem?

DRL is suitable because the environment changes over time and the system must make sequential decisions. The result of one action, such as parking a controller, affects future load, latency, and energy. DRL can learn a policy from these repeated interactions.

## Q23: What is the weakness of normal Q-learning for this problem?

Normal tabular Q-learning is not suitable because the state space is large and continuous. Controller loads, switch assignments, and controller states create many possible combinations, so a neural-network-based method like DQN or RDQN is more practical.

## Q24: How do the literature papers justify your methodology?

The literature shows that controller load imbalance is a real problem, switch migration is a valid solution, and DRL can improve adaptive decision-making. My methodology combines these ideas and extends them with parked-controller management for energy efficiency.

## Q25: If asked to name five papers quickly, what should you say?

I can mention: Chen et al. 2025 on MOOO-RDQN, Yeo et al. 2021 on SAR-LB switch migration, Liu et al. 2025 on controller load optimization, Li et al. 2024 on DRL-based controller placement in SDN-MEC, and Das et al. 2020 on controller placement in SDN.

## Q26: What is the common limitation across the reviewed papers?

The common limitation is that most approaches focus on a specific part of SDN management, such as placement, migration, or load balancing. My work tries to combine these ideas into a proactive controller management approach that also considers energy.

## Q27: How will you explain the literature review in the presentation?

I will explain it in three parts: first, controller placement papers show why controller location and switch mapping matter; second, switch migration papers show how load can be balanced dynamically; third, DRL papers show why learning-based methods are useful for adaptive SDN control.

## Q28: What should you say if they ask whether your idea is completely new?

I should say that the individual concepts already exist in the literature, such as switch migration, controller placement, and DRL. My contribution is combining these ideas into a proactive RDQN-based framework with migration, parking, and evoking decisions for SDN controller management.

## Q29: What kind of mechanism are you currently using to generate traffic for each topology?

I am currently using topology-specific synthetic traffic generation. The topology files are real, but the traffic traces are generated. The traffic generator creates switch-level load values using day-night patterns, bursts, hotspot switches, random noise, cooldown periods, and temporal memory.

## Q30: Is the synthetic traffic generated using a built-in traffic generation library?

No. It is generated using a custom Python script with NumPy. I am not currently using Mininet traffic tools, iPerf, D-ITG, ns-3, or a separate traffic simulator for this traffic generation.

## Q31: Which file generates the synthetic traffic?

The main file is `experiments/preprocessing/create_synthetic_traffic.py`. It generates `.npy` traffic matrices and `.json` metadata files under `data/traffic/processed/<topology>/`.

## Q32: What does one generated traffic file contain?

Each traffic file contains a matrix with shape `timesteps x number_of_switches`. Each value is a normalized switch load between 0 and 1 for a particular time step and switch.

## Q33: How should you explain your traffic generation in the viva?

I should say: "I use real network topologies, but the traffic is synthetically generated using a custom NumPy-based generator. The generator simulates realistic dynamic traffic using day-night load variation, hotspot switches, burst events, random noise, and smooth temporal changes."

## Q34: How are you calculating average latency without a baseline?

Average latency is calculated directly from the topology delay matrix and the current switch-to-controller assignment. For each switch, the environment gets the delay to its assigned controller, sums all switch-controller delays, and divides by the number of switches.

## Q35: How are you calculating worst-case latency?

Worst-case latency is the maximum controller-switch delay among all current switch-controller assignments. It shows the worst communication delay experienced by any switch in the current controller mapping.

## Q36: Do you need a baseline to calculate latency?

No. A baseline is not needed to calculate latency. Latency is a metric computed from the current topology and controller assignment. A baseline is only needed to compare whether my RDQN method improves over another method.

## Q37: Do you have implemented baselines?

I have initial baseline scripts for random policy and threshold-based policy: `experiments/baseline_random.py` and `experiments/baseline_threshold.py`. However, these scripts belong to an earlier environment version and are not fully aligned with my current `ThresholdBasedProactiveSDN` environment.

## Q38: What should you honestly say if asked about baseline comparison?

I should say: "At this stage, I have implemented initial baseline scripts, but the final fair baseline comparison still needs to be aligned with my current RDQN environment. Currently, I report absolute metrics such as average latency, worst-case latency, energy, load balance, and action counts. Baseline comparison is one of my next steps."

## Q39: Where does the topology delay matrix come from?

The delay matrix is created by `utils/topology_loader.py`. It loads real `.gml` topology files, calculates edge delays using geographic distance where coordinates are available, and computes shortest-path delays from switches to selected controller host nodes.

## Q40: How should you explain the difference between metric calculation and comparison?

Metric calculation means computing values like latency, energy, and load balance for my current model. Comparison means evaluating those same metrics against another method such as random policy, threshold policy, or standard DQN.
