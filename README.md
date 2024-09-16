### Embedding-based classifiers can detect prompt injection attacks
In this project, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against prompt injection attacks. We leverage three commonly used embedding models, such as API-only OpenAI `text-embedding-3-small`, and the open-source models `gte-large`, and `all-MiniLM-L6-v2`, to generate embeddings of malicious and benign prompts. Then, we utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in the open-source that use encoder-only neural networks.

The research project has been published at the Conference on Applied Machine Learning in Information Security [(CAMLIS 2024)](https://www.camlis.org/).

### Citing this work
If you use our implementation for scientific research, you are highly encouraged to cite our paper.

```
@article{ayub2024embedding,
  title={Embedding-based classifiers can detect prompt injection attacks},
  author={Ayub, Md Ahsan and Majumdar, Subhabrata},
  booktitle={CAMLIS},
  year={2024}
}
```
