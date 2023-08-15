# Code-Review-of-BiomedGPT Based on LLM

In this repo Provide General Overview how does BiomedGPT Built to tackle Multi-Modalities Input to generate Textual Output relate to BioMedical handle multi-Task such :

- Classification
- Caption
- V&Q
- image filling
- masked language modeling

<div align="center">
    <img src="assets/modeling.png" width="600" height="350" />
</div>

**Notation**: WE WILL SPLIT OUR REVIEW PAPER AND CODE FOLLOWING SECTION ..

1. [background](#background)
2. [paper-insghits](#paper-insghits)
3. [BiomedGPT-Pipeline](#BiomedGPT-Pipeline)
4. [ENVIREMNET-SETUP](#ENVIREMNET-SETUP)
5. [RESULTS](#Results)
6. [CONCLUSION](#CONCLUSION)

### background

Introducing BiomedGPT **(Biomedical Generative Pre-trained Transformer)**, the paper presents a versatile model tailored for diverse biomedical data and tasks. Leveraging self-supervision on extensive datasets, BiomedGPT exhibits superior performance over leading benchmarks. **The study encompasses five tasks and 20 public datasets spanning 15 distinct biomedical modalities, highlighting BiomedGPT's broad applicability.** Notably, the authors' innovative multi-modal, multitask pretraining approach showcases effective knowledge transfer to novel data. This contribution marks a significant stride in creating adaptable and comprehensive biomedicine models, offering potential enhancements to healthcare outcomes.

### paper-insghits

we will include the most interesting ideas in the paper the Co-authors discussed and explained in their paper

1. Capabilities of the BiomedGPT Model : 

    The capabilities of the BiomedGPT model have been realized as a result of the study, which has opened up new avenues in biomedical research. This advancement aims to enhance the collaboration between AI and medicine by fostering a deeper understanding of the intricate biological mechanisms that underlie both human health and disease. These contributions are summarized as follows:

    * **Versatility Across Domains**: BiomedGPT spans various biomedical domains, setting a new benchmark for pretraining effectiveness. It excels in pathology, radiology, and academic literature, demonstrating proficiency in different body parts across modalities.

    * **In-depth Insights**: BiomedGPT is designed to encompass a wide range of domains in biomedicine. Our experimental results set a new benchmark, illustrating the feasibility of pretraining across diverse biomedical fields such as pathology, radiology, and academic literature. This is coupled with an ability to handle various body parts across different modalities.

### BiomedGPT-Pipeline

The proposed BiomedGPT model adopts an encoder-decoder architecture, which effectively maps various modalities into a unified semantic representation space and handles diverse tasks.

* **Architecture select**

BiomedGPT is designed based on the [BART architecture](https://huggingface.co/docs/transformers/model_doc/bart), a sequence-to-sequence model with a [BERT-style encoder](https://arxiv.org/abs/2306.03678) and a [GPT-style decoder](https://jalammar.github.io/illustrated-gpt2/). The authors make architectural changes to improve convergence efficiency and stability in pretraining: they add normalization operations, incorporate absolute and relative position embeddings for text and images, and implement decoupling to separate position correlation. These modifications enhance the model’s ability to encode positional information and handle different modalities.


* **Input/Output Unification**

