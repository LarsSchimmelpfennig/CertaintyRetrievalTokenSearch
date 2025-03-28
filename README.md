# Certainty Retrieval Token Search (CeRTS)

LLM uncertainty estimator for structured information extraction.

# Description

Aimed at use with LLM JSON extraction tasks, CeRTS explores the token tree space to find all outputs above some probability threshold. This is intended for extracting one feature at a time but multiple is supported. The user specifies the minimum cumulative probability threshold and max inference steps to restrict the search space.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d99f4be-925e-4c17-80f8-ac4a2f21e278" height="300px">
</p>

This tool allows the user to quickly explore the entire output space with minimal computational cost. The measured output distribution can then be used to estimate confidence using the probability difference between the two most likely output sequences (Top-2 Delta).


