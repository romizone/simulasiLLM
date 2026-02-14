# Transformer Explainer — LLM Attention Simulation (GPT-2 Style)

An interactive, educational web-based simulation that visualizes how the **attention mechanism** works inside transformer-based large language models, modeled after the **GPT-2 Small** architecture.

> **Live Demo:** [simulasillm.vercel.app](https://simulasillm.vercel.app/)
>
> **Research Paper:** [paper-llm-attention.vercel.app](https://paper-llm-attention.vercel.app/)

---

## Overview

This project provides a hands-on way to explore the inner workings of transformer models. Instead of reading equations on paper, users can watch each stage of the pipeline in real time — from tokenization and embedding, through Query/Key/Value projections, to masked self-attention and next-token sampling.

### Key Features

- **Token Embedding Visualization** — See how input text is split into tokens and mapped to embedding vectors with positional encoding
- **Q/K/V Projection Inspector** — Examine Query, Key, and Value vectors for each attention head
- **Masked Self-Attention Heatmap** — Interactive attention matrix with causal masking, showing which tokens attend to which
- **Next-Token Probability Distribution** — Real-time softmax output displaying candidate tokens and their probabilities
- **Autoregressive Text Generation** — Generate tokens step-by-step and observe how each new token influences subsequent attention patterns
- **Adjustable Sampling Parameters** — Control Temperature (0.3–1.5), Top-k (1–12), and generation length to see how they affect output diversity
- **Multi-Head Attention** — Switch between attention heads to compare different learned attention patterns

### Model Specifications (Simulated)

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 Small |
| Layers | 12 |
| Attention Heads | 4 (visual) |
| Hidden Size | 16 |
| Head Dimension | 4 |
| Causal Mask | Active |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Markup | HTML5 |
| Styling | CSS3 (Custom Properties, Grid, Flexbox) |
| Logic | Vanilla JavaScript (ES2020+) |
| Fonts | Google Fonts (Orbitron, Space Grotesk, IBM Plex Mono) |
| Hosting | Vercel (static) |

Zero dependencies — no frameworks, no build step, no bundler.

---

## Project Structure

```
simulasiLLM/
├── index.html        # Main UI layout — toolbar, attention SVG, token stream, output panels
├── style.css         # Full styling with CSS custom properties
├── app.js            # Simulation engine — tokenizer, attention math, rendering, generation
├── DEPLOY.md         # Deployment guide for Vercel, Cloudflare Pages, Netlify, GitHub Pages
└── .github/
    └── workflows/    # CI/CD configuration
```

---

## Getting Started

### Run Locally

No install required — just serve the static files:

```bash
git clone https://github.com/romizone/simulasiLLM.git
cd simulasiLLM
python3 -m http.server 8081
```

Open [http://127.0.0.1:8081](http://127.0.0.1:8081) in your browser.

Alternatively, use any static file server:

```bash
# Node.js
npx serve .

# PHP
php -S localhost:8081
```

### Deploy

See [DEPLOY.md](./DEPLOY.md) for deployment instructions on Vercel, Cloudflare Pages, Netlify, and GitHub Pages.

---

## How It Works

```
Input Text
    │
    ▼
┌─────────────┐
│  Tokenizer  │  Split text into tokens using Unicode-aware regex
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  Map tokens to dense vectors + positional encoding
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Q / K / V       │  Project embeddings into Query, Key, Value spaces
│  Projections     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Scaled Dot-     │  Compute attention scores with causal mask
│  Product         │  score = (Q · Kᵀ) / √d_k, then mask future tokens
│  Attention       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Softmax +       │  Convert scores to probabilities,
│  Sampling        │  apply temperature & top-k filtering
└──────┬───────────┘
       │
       ▼
  Next Token (autoregressive loop)
```

---

## Research Paper

The accompanying research paper provides the theoretical foundation and detailed system architecture behind this simulation:

**"Simulating the Attention Mechanism in Large Language Models Based on the GPT-2 Architecture"**

Read the full paper at [paper-llm-attention.vercel.app](https://paper-llm-attention.vercel.app/)

Topics covered:
- Token processing pipeline (BPE tokenization, embedding, positional encoding)
- Mathematical formulation of scaled dot-product attention
- Causal masking in autoregressive generation
- Temperature scaling and top-k sampling strategies
- GPT-2 Small architecture specifications

---

## Screenshots

### Attention Heatmap & Token Stream
The main interface displays the token embedding stream on the left, the attention core (Q/K/V projections and masked self-attention matrix) in the center, and the next-token probability distribution on the right.

### Autoregressive Generation
Generated tokens appear below the main panel, showing the model's step-by-step text continuation with probability annotations.

---

## License

This project is open source and available for educational purposes.

---

## Author

**Romin Urismanto** — [@romizone](https://github.com/romizone)
