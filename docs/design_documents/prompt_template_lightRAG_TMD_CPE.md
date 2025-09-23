Prompt template and Modify LightRAG for LNSP Use

Structured Prompt Template and Fine-Tuning Guide for LiteRAG Enhancements

9/21/2025
Trent Carter

Overview
This document provides a comprehensive guide based on our previous discussion. It includes:
* A prompt template for extracting propositions and additional parameters (domain, task, modifier, mission text, expected answer) from text chunks using a large language model (e.g., Llama-3-70B or similar open-source models like Mixtral-8x7B).
* Details on JSON fine-tuning, including example data pairs and training recommendations.
* Information on the tool (Outlines) that enforces structured generation to prevent hallucinations.
The goal is to perform a single-pass inference on your data chunks, outputting structured JSON that can be directly integrated into Faiss vectors (e.g., by concatenating encoded bits) and Neo4j graphs. This optimizes for efficiency in a propositional chunking setup within LiteRAG.
Prompt Template
Use this template for guided prompting. It's designed for models like Llama-3-70B, integrated with Outlines for schema enforcement. The prompt ensures structured JSON output only, reducing hallucinations by constraining the model's token generation to valid schema paths.
Base Prompt Structure
text

You are an expert extractor of propositional knowledge from text. Given the input text, extract the following in a single JSON object:

- `concept_text`: The core atomic concept, phrased as a concise, standalone statement (string).
- `probe_question`: A question that the `concept_text` directly and completely answers (string).
- `expected_answer`: The expected answer to the `probe_question`, often a short phrase or entity (string).
- `domain`: The primary domain category from the official TMD schema (enum).
- `task`: The primary cognitive task from the official TMD schema (enum).
- `modifier`: The descriptive modifier from the official TMD schema (enum).
- `mission_text`: The original extraction prompt or mission that guided this process (string).


Output ONLY the JSON object. No explanations or additional text.

Input text: {input_text}


Example Usage
Replace {input_text} with your actual chunk, e.g.:
* Input: "Albert Einstein developed the theory of general relativity in 1915, revolutionizing our understanding of gravity."
* Expected Output JSON:text{
*   "prop": "Albert Einstein developed the theory of general relativity in 1915.",
*   "domain": "physics",
*   "task": "develop",
*   "modifier": "revolutionizing",
*   "mission": "Revolutionize understanding of gravity.",
*   "expected": "Gravity as curvature of spacetime."
* }
This template can be fed directly into the model during inference. For the 16-bit encoding: After extraction, bit-shift the enum indices (e.g., domain: 4 bits for 16 options, task: 6 bits for 64, modifier: 5 bits for 32 – totaling ~15 bits) into a uint16 value and concatenate it to your 768-dim embeddings (assuming standard like from Sentence Transformers) for Faiss storage.
JSON Fine-Tuning Recommendations
Fine-tune the model on JSON input-output pairs to adapt it for your specific propositional extraction task. This ensures high accuracy on your dataset (e.g., Wikipedia dumps or custom corpora).
Fine-Tuning Setup
* Model: Start with a large open-source model like Llama-3-70B, Mixtral-8x7B, or even larger (e.g., 405B variants if hardware allows). These handle complex extractions well without needing proprietary APIs.
* Technique: Use LoRA (Low-Rank Adaptation) for efficiency – fine-tune only adapters, not the full model. Tools like Unsloth make this fast on consumer GPUs (e.g., RTX 4090 can handle 70B with 4-bit quantization).
* Dataset: Create ~1,000-10,000 JSON pairs from your data. Manually label a subset, then use a base model to semi-automate the rest.
    * Example Pair (Input JSON for fine-tuning prompt):text{
    *   "input": "The quick brown fox jumps over the lazy dog.",
    *   "output": {
    *     "prop": "The quick brown fox jumps over the lazy dog.",
    *     "domain": "linguistics",
    *     "task": "jump",
    *     "modifier": "quickly",
    *     "mission": "Demonstrate pangram sentence.",
    *     "expected": "All letters of the alphabet used."
    *   }
    * }
    * Another Example:text{
    *   "input": "Photosynthesis converts light energy into chemical energy in plants.",
    *   "output": {
    *     "prop": "Photosynthesis converts light energy into chemical energy in plants.",
    *     "domain": "biology",
    *     "task": "convert",
    *     "modifier": "efficiently",
    *     "mission": "Enable plant growth via energy transformation.",
    *     "expected": "Glucose and oxygen produced."
    *   }
    * }
* Training Parameters:
    * Epochs: 5-10 (monitor for overfitting).
    * Batch Size: 4-16 (depending on GPU VRAM).
    * Learning Rate: 2e-4.
    * Optimizer: AdamW.
    * Use Hugging Face's transformers library with peft for LoRA.
* Enum Handling: During fine-tuning, include the enum lists in the prompt to teach the model constraints (e.g., "domain: one of ['physics', 'biology', ..., 'economics']").
* Tools for Fine-Tuning:
    * Unsloth: Speeds up training by 2-5x, supports quantization.
    * Dataset Format: Use Hugging Face Datasets for loading JSONL files.
After fine-tuning, the model will reliably output structured JSON in one pass, incorporating all elements (propositions, classifications, mission, expected answer).
Tool for Preventing Hallucinations: Outlines
Outlines (from the Outlines library on GitHub) is a guided generation tool that enforces structured outputs during inference. It prevents hallucinations by restricting the model's logits to only valid tokens that match your schema.
How It Works
* Integration: Wrap your model with Outlines. It uses regex or JSON schema to guide sampling – e.g., force JSON structure, enum values, and string formats.
* Example Code Snippet (Python with Hugging Face):pythonfrom outlines import models, generate, samplers
* from outlines.integrations import transformers
* from outlines.guide import json_guide
* 
* # Load fine-tuned model
* model = models.transformers("your/fine-tuned-llama-70b")
* 
* # Define JSON schema (simplified)
* schema = {
*   "type": "object",
*   "properties": {
*     "prop": {"type": "string"},
*     "domain": {"enum": ["physics", "biology", ...]},  # Your 16 options
*     "task": {"enum": ["propose", "discover", ...]},   # 64 options
*     "modifier": {"enum": ["boldly", "cautiously", ...]},  # 32 options
*     "mission": {"type": "string"},
*     "expected": {"type": "string"}
*   },
*   "required": ["prop", "domain", "task", "modifier", "mission", "expected"]
* }
* 
* # Generate with guidance
* prompt = f"From text: {input_text} Extract: ..."
* output = generate.text(model, prompt, sampler=samplers.greedy(), guide=json_guide(schema))
* Benefits:
    * Zero extra inference cost – it prunes invalid paths in the logit processor.
    * "Handcuffs" the model: No drifting into irrelevant text; outputs are always valid JSON.
    * Compatible with vLLM or Hugging Face for batching.
* Repo and Resources: Check the Outlines GitHub repo (github.com/outlines-dev/outlines). They have Colab notebooks for guided decoding with Llama – fork one, swap in your schema and dataset.
This setup ensures everything is generated in one model call, minimizing token usage and latency.
