# StructuredTokenSearch

Structured Token Search with nucleus sampling.

# Description

Aimed at use with LLM JSON extraction tasks, this tool explores the token tree space to find all outputs above some probability threshold. This is intended for extracting one feature at a time but multiple is supported. The user specifies top_p and the minimum cumulative probability threshold to control how thorough the search is.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e1135d3-31a3-42a5-80f8-9951850f776a" height="400px">
</p>

This tool allows the user to quickly explore the entire output space with minimal computational cost. The measured output distribution can then be used to estimate confidence.


