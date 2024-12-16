## Winograd Natural Language Inference
### UW CSE 447 Extra Credit

Code trains a neural network to solve a natural language inference task using the WNLI dataset. It compares the performance of two embedding techniques (GloVe and SentenceTransformer) post-hyperparameter optimization.

Description of Winograd Schema Challenge:
[Full paper here](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf)

The Winograd Natural Language Inference (WNLI) task is a disambiguation task that is part of the GLUE benchmark. The dataset contains two sentences - the first contains a pronoun like “it” or “they”, and the second attempts to disambiguate the first sentence. Finally, there’s a label of 1 or 0 for whether or not the disambiguation attempt is correct.

For example, consider the pair of sentences:

1. “I could not put the pot on the shelf because it was too high.”
2. “The pot was too high.”

This pairing would have a label of 0 because it’s the shelf that’s too high, not the pot.