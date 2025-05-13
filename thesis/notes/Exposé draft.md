Extend background so that there are 3 methods described instead of 2, namely ECFP, GNNs and Neural Graph Fingerprints as in duvenaud et al. clearly state what makes neural graph fingerprints different to a standard GNN approach. Any text in braces like this {context} should be seen as added context that you will use to change this section.
### **Background**
{Incorporate NGF as a third strategy into this paragraph}
In molecular property prediction, it is crucial to develop effective molecular representations—commonly known as molecular fingerprints—to serve as inputs for machine learning models. The traditional paradigm in this area is the discrete and rule based ECFP algorithm. Another approach is using Graph Neural Networks (GNN) which apply a data driven deep learning strategy to learning powerful molecular representations. A third strategy is Neural Graph Fingerprints (NGF), which blend ECFP with the adaptive learning capabilities of GNNs.

**Extended Connectivity Fingerprint (ECFP):**
ECFP generates fixed-size binary vectors by iteratively aggregating local neighborhood information of atoms. In each iteration, the algorithm combines an atom’s features with those of its neighbors, hashing the result to create discrete structural identifiers that progressively capture larger substructures. These identifiers set corresponding bits in the fingerprint, yielding a representation that encodes the presence or absence of specific substructures.[1]

**Neural Graph Fingerprints**:
Neural Graph Fingerprints build on the concept of ECFP by replacing its discrete, non-differentiable operations with differentiable neural network components. This key innovation allows NGF to generate molecular fingerprints that retain the interpretability and structural insights of ECFP while benefiting from gradient-based optimization. In contrast to standard GNN approaches that are primarily geared toward learning representations for end-to-end tasks, NGF is specifically designed to mimic the functionality of traditional fingerprints. 

**Graph neural network (GNN):**
GNNs learn molecular representations in an end-to-end differentiable manner. They propagate and transform node and edge features through a series of message-passing layers and then aggregate the resulting information with a global pooling operation at the readout layer. 

The key difference between these approaches lies in how they generate and optimize molecular representations. ECFP relies on fixed, rule-based, and non-differentiable operations such as hashing and bit setting to create binary fingerprints. In contrast, standard GNNs employ continuous, differentiable functions that update node and edge features through iterative message passing and enable end-to-end training via gradient descent. Neural Graph Fingerprints (NGF) bridge these paradigms by substituting the discrete steps of ECFP with differentiable neural operations.
### **Motivation**
#### Research interest and questions
This thesis explores how to integrate the discrete and iterative approach of ECFP into the design of Graph Neural Networks aiming at producing high quality embeddings for downstream tasks like molecular property prediction. This research seeks to answer the following questions:
- How can discrete iterative methods like ECFP be integrated into the continuous and differentiable nature of Graph Neural Networks?
- What benefits arise from leveraging the concatenation of intermediate representations in Graph Neural networks?

#### Goal of the Project
The primary goal of this project is to investigate how combining interim iteration results—as employed in ECFP—can be integrated into the design of Graph Neural Networks. Specifically, the project aims to develop a GNN architecture that generates molecular embeddings by explicitly incorporating and aggregating intermediate node-level embeddings and combining them with concatenation. 
### **Approach**
The initial phase of the project involves using a frozen, randomly initialized Graph Isomorphism Network (GIN) [1] as a fixed feature extractor for molecular graphs. This GIN converts each molecular graph into a corresponding embedding, which is then used as input to train an MLP classifier. The performance of this classifier—measured by ROC-AUC—will be directly compared to that obtained using traditional ECFP bit vectors as molecular representations. Additionally, a frozen Neural Graph Fingerprint (NGF) [4] model will serve as another baseline, with its embeddings used to train a separate MLP classifier.

In the second phase, the focus shifts to designing an enhanced GIN that integrates mechanisms such as Jumping Knowledge concatenation [3] to aggregate interim iteration outputs into the final graph embedding. The aim is to assess how closely the embeddings produced by this enhanced, frozen GIN match the discriminative power of ECFP when used for downstream tasks via an MLP classifier.

Additionally, the expressivity of the embeddings and fingerprints produced by their respective methods. For this, it will be investigated how many unique embeddings/fingerprints are produced from a dataset of molecules. 
#### Challenges to consider

- A differentiable neighbourhood aggregation method
	 In ECFP, the local neighborhood of each atom is aggregated by hashing the atom's feature array along with its neighbors' features into a new substructure identifier. This process, while effective for generating fixed binary fingerprints, is inherently non-differentiable. For a GNN to support end-to-end learning via gradient descent, the neighborhood aggregation must be implemented using differentiable operations. [4]

- Duplicate removal:
	 In traditional ECFP fingerprint generation, if the same identifier is produced for different  atoms or substructures, the fingerprint simply marks the corresponding bit as active, effectively collapsing duplicates into a single indicator. This non-differentiable process is acceptable for fixed, binary fingerprints. However, in a GNN, every operation must be differentiable to allow for gradient-based learning. Therefore, the GNN must employ differentiable aggregation methods that can handle duplicate or highly similar intermediate representations without disrupting the learning process.

- Transforming graph embeddings into fixed size fingerprints
	 When using concatenation for neighborhood aggregation, the resulting graph-level  embedding may vary in length from one molecule to another, as it depends on the number of nodes (atoms). However, downstream tasks such as classification require input vectors of a consistent dimensionality. To address this, a differentiable operation must be applied to transform the variable-length, concatenated embeddings into fixed-size neural fingerprints.
### **Milestones**
#### 01.03.2025 - 01.04.2025 - Milestone 1
- Reading relevant literature
- Setup of a training pipeline and the benchmark experiment
#### 01.04.2025 - 15.05.2025 - Milestone 2
- Compare frozen downstream task performance between ECFP4, ECFP6, NGF2, NGF3, GIN2 and GIN3
#### 15.05.2025  - 15.06.2025 Milestone 3
- Develop enhanced GIN and assess frozen downstream task performance
- Train the enhanced GIN and assess downstream task performance
#### 15.06.2025 - 30.06.2025 Milestone 4
- Presentation of the findings - Date: 25.06.2025
- Submission of the written report
### **Literature**
[1] David Rogers, Mathew Hahn, Extended Connectivity Fingerprints, J. Chem. Inf. Model. 2010, 50, 742–754
[2] Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, How powerful are Graph Neural Networks? Published as a conference paper at ICLR 2019
[3] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka, Representation Learning on Graphs with Jumping Knowledge Networks, Proceedings of the 35th International Conference on Machine Learning, Stockholm, Sweden, PMLR 80, 2018
[4] David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, Ryan P. Adams, Convolutional Networks on Graphs for Learning Molecular Fingerprints, NIPS'15: Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 2, 07 December 2015