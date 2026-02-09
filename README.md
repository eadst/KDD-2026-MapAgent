# MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation

This repository is created for the paper:

**MapAgent: An Industrial-Grade Agentic Framework for City-scale Lane-level Map Generation**

We are currently preparing a public release. **A subset of the codebase and runnable examples will be added soon**, including key components and minimal scripts for reproduction and demonstration.

---

## Overview

MapAgent is an agentic refinement framework built on top of a frozen BEV vectorization backbone.  
It introduces a bounded **Judge–Planner–Worker** loop to verify mapping specifications and perform deterministic, tool-grounded edits for lane-level map refinement.

---

## Status

🚧 **Work in progress**  
- Code and examples will be released in phases.
- Additional documentation (installation, data format, and usage) will be provided along with the first release.

---

## Planned Contents (to be released)

- Core pipeline skeleton (Quality Agent + bounded refinement loop)
- Judge Agent prompts / interfaces and structured outputs
- Deterministic worker tools (e.g., delete / smooth / regenerate) and minimal demos
- Example configurations and small, runnable scripts

---

## Citation

If you find this project useful, please consider citing our paper (BibTeX will be provided after publication).
