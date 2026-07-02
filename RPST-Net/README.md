# TOMBS-Mural and RSPT-Net

RSPT-Net (Reference-guided Style Prior Transformer Network) is designed for component-level restoration of Tang dynasty tomb murals. The project focuses on evidence-driven mural inpainting, where restoration is guided not only by the damaged image itself but also by historically and stylistically relevant mural components. Built around TOMBS-Mural, the framework links professional archaeological annotation, component-level reference organization, structure-prior construction, and transformer-based restoration into a unified workflow. Experiments on TOMBS-Mural and MuralDH show that RSPT-Net improves structural fidelity, stylistic consistency, and perceptual quality, offering a practical framework for knowledge-guided cultural heritage restoration.

## Method Overview

The framework starts from a structured component-level reference mechanism built on TOMBS-Mural. A multi-factor retrieval strategy selects semantically and stylistically consistent reference components, forming a traceable historical evidence bank. The geometry-aware structure prior refines contour alignment between references and target regions. A multi-scale style encoder then extracts hierarchical representations from retrieved murals, which are fused into the restoration backbone through cross-attention and FiLM-based modulation. This design enables simultaneous modeling of structural morphology, local texture continuity, and global stylistic consistency. Extensive experiments demonstrate that RSPT-Net significantly improves restoration quality and provides a more interpretable and evidence-driven paradigm for mural image inpainting.

## Data release policy

The complete dataset is available from the corresponding author upon reasonable request. Additional data will be released in future updates.