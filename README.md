### Embedding-based classifiers can detect prompt injection attacks
In this project, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against prompt injection attacks. We leverage three commonly used embedding models, such as API-only OpenAI `text-embedding-3-small`, and the open-source models `gte-large`, and `all-MiniLM-L6-v2`, to generate embeddings of malicious and benign prompts. Then, we utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in the open-source that use encoder-only neural networks.

The research project has been published at the Conference on Applied Machine Learning in Information Security [(CAMLIS 2024)](https://www.camlis.org/).


#### Dataset
The dataset used in our experiments is curated from open-source datasets containing malicious and benign prompts pertaining to prompt injection attacks. In total, we acquire a total of 553,185 numbers of malicious and benign prompts. After deduplication, we end up with a total of 467,057 unique prompts, of which 109,934 (23.54%) are malicious. Each prompt is assigned a unique identifier and a source to indicate its origin. Therefore, the dataset columns appear as follows: ID, Source, Text, and Label (0 to denote benign, 1 for malicious). Please go to [dataset](https://github.com/AhsanAyub/malicious-prompt-detection/tree/main/dataset) folder to access them.

| Dataset (User: Title)                     | # fo Prompts |
| ----------------------------------------- | ------------ |
| imoxto: Prompt Injection cleaned dataset  | 535,105      |
| reshabhs: SPML Chatbot Prompt Injection   | 16,012       |
| Harelix: Prompt Injection Mixed Techniques| 1,174        |
| JasperLS: Prompt Injections               | 662          |
| fka: Awesome Chatgpt Prompts              | 153          |
| rubend18: ChatGPT Jailbreak Prompts       | 79           |

We develop a data pipeline using Python 3.11 to generate the embeddings for all prompts. With OpenAI's API key, we submit each prompt to get its embedding through `text-embedding-3-small` model. To obtain the GTE embeddings, we use the `thenlper/gte-large`, accessed remotely through the serverless endpoint on OctoAI. For the MiniLM embeddings, we download the `sentence-transformers/all-MiniLM-L6-v2` model and host it locally. This approach allowed us to construct three separate tabular datasets composed of embeddings based on each of the embedding models. Please go to [embeddings](https://github.com/AhsanAyub/malicious-prompt-detection/tree/main/embeddings) folder to access them.


#### Citing this work
If you use our implementation for scientific research, you are highly encouraged to cite our paper.

```
@article{ayub2024embedding,
  title={Embedding-based classifiers can detect prompt injection attacks},
  author={Ayub, Md Ahsan and Majumdar, Subhabrata},
  booktitle={CAMLIS},
  year={2024}
}
```
