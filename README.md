---
title: Kgp Finance Advisor
emoji: 💲
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
python_version: 3.10.13
---

# 📉 ChatKGP: The IIT Kharagpur AI Finance Advisor

An iterative, domain-specific Large Language Model (LLM) pipeline designed to provide hyper-local, conversational financial advice for students at IIT Kharagpur.

## 🚀 Overview & The Problem

Personal finance is incredibly localized. General AI models give generic, high-income advice ("Save 20% of your salary in an index fund"), which is entirely useless to college students. Students need actionable advice based on their actual economic reality ("How do I budget for PFC, Chedis, and Hall expenses on a ₹5,000 monthly allowance?"). 

ChatKGP was built to solve this by creating an AI that intimately understands the financial ecosystem, campus slang, and constraints of an IIT KGP student.

## 🧬 Project Evolution: From Fine-Tuning to Edge Optimization

### Phase 1: Domain Adaptation (The Fine-Tuning Origin)
The project began as an NLP fine-tuning experiment. Base models lacked context about IIT KGP. To bridge this gap, a custom dataset of local financial scenarios was curated and injected into a base LLM using **LoRA (Low-Rank Adaptation)** via the `Unsloth` framework. This successfully taught the model the required campus-specific knowledge and colloquialisms without retraining the entire neural network from scratch.

### Phase 2: The Edge Deployment Reality
While fine-tuning succeeded in a high-VRAM Google Colab environment, deploying the heavy model to a free-tier, serverless cloud environment (16GB RAM, 2 vCPUs) introduced severe infrastructure bottlenecks, forcing a complete architectural pivot toward edge-optimization.

## 🧠 Current Architecture & Tech Stack (V1)

* **Base Engine:** Microsoft `Phi-3-Mini` (3.8B Parameters)
* **Training Pipeline:** `Unsloth` (LoRA Fine-Tuning)
* **Optimization:** 4-bit Quantized `.gguf` format via `llama.cpp`
* **Frontend UI:** Gradio with custom CSS (Dark Mode/Gradient styling)
* **Hosting:** Hugging Face Spaces (Serverless CPU Tier)

## 🛠️ Technical Challenges & Architectural Pivots

Building and deploying a generative AI application on constrained hardware presented significant DevOps challenges. Here is how the infrastructure was stabilized:

### 1. The "OOM" (Out of Memory) Bottleneck
* **The Problem:** The initial fine-tuning phase utilized Llama-3 (8B parameters). During cold starts on the serverless host, the model's memory footprint combined with context-window allocation exceeded the 16GB container limit, resulting in fatal `Exit Code 137 (OOMKilled)` server crashes.
* **The Solution:** Executed an architectural "Right-Sizing" pivot. The training pipeline was migrated from Llama-3 to Microsoft's `Phi-3-Mini`. By quantizing the fine-tuned Phi-3 model to a 4-bit `Q4_K_M.gguf` format, the total footprint was reduced to ~2.3GB. This eliminated OOM errors entirely while preserving the injected KGP knowledge.

### 2. CI/CD and Deployment Latency
* **The Problem:** Dependency resolution for `llama-cpp-python` triggered a fallback to source-code compilation, forcing the free-tier CPU to compile C++ during every deployment. This caused agonizing 30+ minute build times.
* **The Solution:** Implemented strict dependency pinning. By locking the environment to `python_version: 3.10.13` and `llama-cpp-python==0.2.85`, the CI/CD pipeline was forced to utilize pre-compiled CPU wheels. Deployment container build times were reduced from 30+ minutes to less than 15 seconds.

### 3. UX Latency Masking
* **The Problem:** Running LLM inference strictly on a CPU inherently results in high Time-To-First-Token (TTFT) and slow generation speeds.
* **The Solution:** Implemented Python generators to stream tokens asynchronously to the Gradio frontend (`stream=True`). This provides immediate visual feedback to the user, effectively masking the hardware latency and creating a fluid conversational experience.

## 💻 Local Installation

To run the optimized ChatKGP inference engine locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/ChatKGP.git](https://github.com/your-username/ChatKGP.git)
   cd ChatKGP