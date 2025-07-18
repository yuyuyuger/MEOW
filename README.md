# 🚀 MEOW
Aligning Multi-level Semantics for Multi-modal Knowledge Graph Completion.

# Abstract:
Current multi-modal knowledge graph completion often incorporates simple fusion neural networks to achieve multi-modal alignment and knowledge completion tasks, which face three major challenges: 1) Inconsistent semantics between images and texts corresponding to the same entity; 2) Discrepancies in semantic spaces resulting from the use of diverse uni-modal feature extractors; 3) Inadequate evaluation of semantic alignment using only energy functions or basic contrastive learning losses. To address these challenges, we propose the Multi-modal Entity in One Word (MEOW) model. This model ensures alignment at various levels, including text-image match alignment, feature alignment and distribution alignment. Specificially, the entity image filtering module utilizes a visual-language model to exclude unrelated images by aligning their captions with corresponding text descriptions. A pre-trained CLIP-based encoder is utilized for encoding dense semantic relationships, while a graph attention network based structure encoder handles sparse semantic relationships, yielding a comprehensive  semantic representation and enhancing convergence speed. Additionally, a diffusion model is integrated to enhance denoising capabilities. The proposed MEOW further includes a distribution alignment module equipped with dense alignment constraint, integrity alignment constraint, and fusion fidelity constraint to effectively align multi-modal representations. Experiments on two public multi-modal knowledge graph datasets show that MEOW significantly improves link prediction performance.

---

## Requrements
torch==1.10.0 numpy==1.21.4 pandas==1.2.4 pickleshare==0.7.5
