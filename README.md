# Enhanced TF-IDF for Knowledge Graph based Abstractive Summarization of Multilingual Documents

## Architecture

<img src="/archive/architecture.png" width="100%" />

This code is based on the thesis **Enhanced TF-IDF for Knowledge Graph based Abstractive Summarization of Multilingual Documents**.

## Running the Model

Seed data is available in the `data` folder of this repository. More data can be uploaded in the same format. However, only English, Hindi and Marathi are supported at the moment. 

To summarize and generate the knowledge graph for a particular topic, execute the following command.

```
python run_summarization.py [topic-name]
```

### Knowledge Graph

![Knowledge Graph](/data/knowledge_graphs/kg_war.png)

### Abstractive Summary

```---summariztion done::  Vladimir V. Putin’s ordered Russian forces to invade Ukraine. The Largest Mobilization of Forces Europe has seen since 1945 is underway. So far, Moscow has been denied the swift victory it anticipated. It has failed to capture major cities across the country, including Kyiv, the capital.```

To view the knowledge graphs, final abstractive summaries and the intermediate summaries, check the respective sub-folders under the `data` folder.

## Evaluation

We evaluate using multiple metrics for the summaries - intermediate as well as final abstractive summary. Run the following command to begin the evaluation process.

```
python evaluation.py
```

Expected output should be something like this:

```
--file name:  war

---BLEU score:  0.09504132231404959
---ROUGE score:  [{'rouge-1': {'r': 0.6933333333333334, 'p': 0.35494880546075086, 'f': 0.4695259548889422}, 'rouge-2': {'r': 0.5550239234449761, 'p': 0.25663716814159293, 'f': 0.35098335422339516}, 'rouge-l': {'r': 0.6733333333333333, 'p': 0.3447098976109215, 'f': 0.45598193683025146}}]
---embedded cosine score:  0.26103894242379827
---frequency cosine score:  0.8052628076498394
---keyBERT score:  0.2857142857142857
```
