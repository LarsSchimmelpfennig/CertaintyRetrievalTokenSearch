# Certainty Retrieval Token Search (CeRTS)

LLM uncertainty estimator for structured information extraction.

# Description

Aimed at use with LLM JSON extraction tasks, CeRTS explores the token tree space to find all outputs above some probability threshold. This is intended for extracting one feature at a time but multiple is supported. The user specifies the minimum cumulative probability threshold and max inference steps to restrict the search space.

<p align="center">
  <img src="https://github.com/user-attachments/assets/eaf372b4-e622-4faa-949e-3034f3b1a293" height="600px">
</p>

This tool allows the user to quickly explore the entire output space with minimal computational cost. The measured output distribution can then be used to estimate confidence using the probability difference between the two most likely output sequences (Top-2 Delta).

Use the CeRTS function defined in src/CeRTS.py 

Future Work
- Multiple feature extraction. Currently only intended for one feature at a time.
  - Detect and handle entitiy dependencies.
- Use beam search to retrieve the top-2 answers instead of measuring the entire output distribution.
- Allow for LLM-assisted or human in the loop validation of the output distribution to allow merging of identical answers.


