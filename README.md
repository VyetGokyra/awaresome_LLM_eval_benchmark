# 📊 250 LLM Benchmarks & Evaluation Datasets

A curated list of **250+ benchmarks** for **Large Language Models (LLMs)** evaluation.

---

## 📂 Overview
- **Total entries:** 250
- **Categories:** Language & Reasoning, Safety, Retrieval, Multilingual, Conversation, Domain-Specific, Others
- **Format:** Markdown list grouped by category
- **Use cases:** Model evaluation, research, leaderboard building

---
## 🗂 Agents & tools use

### ColBench
- **Description:** A new benchmark, where an LLM agent interacts with a human collaborator over multiple turns to solve realistic tasks in backend programming and frontend design.
- **Paper:** SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks 
https://arxiv.org/abs/2503.15478
- **Code:** https://github.com/facebookresearch/sweet_rl
- **Dataset:** https://huggingface.co/datasets/facebook/collaborative_agent_bench
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2025

### BFCL (Berkeley Function-Calling Leaderboard)
- **Description:** A set of function-calling tasks, including multiple and parallel function calls.
- **Paper:** Berkeley Function-Calling Leaderboard
https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html
- **Code:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **Dataset:** https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- **Examples:** 2000
- **License:** Apache-2.0 license
- **Year:** 2024

### FlowBench
- **Description:** A benchmark for workflow-guided planning that covers 51 different scenarios from 6 domains, with knowledge presented in text, code, and flowchart formats.
- **Paper:** FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents 
https://arxiv.org/abs/2406.14884
- **Code:** https://github.com/Justherozen/FlowBench
- **Dataset:** see repo
- **Examples:** 5313
- **License:** see dataset page
- **Year:** 2024

### AutoTools
- **Description:** A framework that enables LLMs to automate the tool-use workflow.
- **Paper:** Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents 
https://arxiv.org/abs/2405.16533
- **Code:** https://github.com/mangopy/Tool-learning-in-the-wild
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

### WorfBench
- **Description:** Unified workflow generation benchmark with multi-faceted scenarios and graph workflow structures.
- **Paper:** Benchmarking Agentic Workflow Generation 
https://arxiv.org/abs/2410.07869 
- **Code:** https://github.com/zjunlp/WorfBench
- **Dataset:** https://huggingface.co/collections/zjunlp/worfbench-66fc28b8ac1c8e2672192ea1
- **Examples:** 21000
- **License:** Apache-2.0 license
- **Year:** 2024

### API-Bank
- **Description:** Specifically designed for tool-augmented LLMs.
- **Paper:** API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
https://arxiv.org/abs/2304.08244
- **Code:** https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank
- **Dataset:** https://huggingface.co/datasets/liminghao1630/API-Bank
- **Examples:** nan
- **License:** MIT License
- **Year:** 2023

### ToolLLM
- **Description:** An instruction-tuning dataset for tool use.
- **Paper:** ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
https://arxiv.org/abs/2307.16789
- **Code:** https://github.com/OpenBMB/ToolBench
- **Dataset:** https://github.com/OpenBMB/ToolBench?tab=readme-ov-file#data-release
- **Examples:** nan
- **License:** Apache-2.0 license
- **Year:** 2023

### ToolBench
- **Description:** A tool manipulation benchmark consisting of software tools for real-world tasks.
- **Paper:** On the Tool Manipulation Capability of Open-source Large Language Models
https://arxiv.org/abs/2305.16504
- **Code:** https://github.com/sambanova/toolbench/tree/main
- **Dataset:** https://github.com/sambanova/toolbench/tree/main
- **Examples:** nan
- **License:** Apache-2.0 license
- **Year:** 2023

### AgentBench
- **Description:** Evaluate LLM-as-Agent across 8 environments, including Operating System (OS)
Database (DB), Knowledge Graph (KG), Digital Card Game (DCG), and Lateral Thinking Puzzles (LTP).
- **Paper:** AgentBench: Evaluating LLMs as Agents
https://arxiv.org/abs/2308.03688
- **Code:** https://github.com/THUDM/AgentBench
- **Dataset:** https://github.com/THUDM/AgentBench/tree/main/data
- **Examples:** 1360
- **License:** Apache-2.0 license
- **Year:** 2023

### MetaTool
- **Description:** A set of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-tool scenarios.
- **Paper:** MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use
https://arxiv.org/abs/2310.03128
- **Code:** https://github.com/HowieHwong/MetaTool
- **Dataset:** https://github.com/HowieHwong/MetaTool/tree/master/dataset
- **Examples:** 20879
- **License:** MIT License
- **Year:** 2023

### Webarena
- **Description:** An environment for autonomous agents that perform tasks on the web.
- **Paper:** WebArena: A Realistic Web Environment for Building Autonomous Agents
https://arxiv.org/abs/2307.13854
- **Code:** https://github.com/web-arena-x/webarena
- **Dataset:** https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json
- **Examples:** nan
- **License:** Apache-2.0 license
- **Year:** 2023

### ToolQA
- **Description:** A new dataset to evaluate the capabilities of LLMs in answering challenging questions with external tools. It offers two levels (easy/hard) across eight real-life scenarios.
- **Paper:** ToolQA: A Dataset for LLM Question Answering with External Tools 
https://arxiv.org/abs/2306.13304
- **Code:** https://github.com/night-chen/ToolQA 
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

### T-Eval
- **Description:** Decomposes the tool utilization capability into multiple sub-processes, including instruction following, planning, reasoning, retrieval, understanding, and review.
- **Paper:** T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step 
https://arxiv.org/abs/2312.14033 
- **Code:** https://github.com/open-compass/T-Eval
- **Dataset:** https://huggingface.co/datasets/lovesnowbest/T-Eval
- **Examples:** 23305
- **License:** Apache-2.0 license
- **Year:** 2023

### GAIA
- **Description:** Presents real-world questions requiring reasoning, multi-modality handling, and tool-use proficiency to evaluate general AI assistants.
- **Paper:** GAIA: A Benchmark for General AI Assistants 
https://arxiv.org/pdf/2311.12983 
- **Code:** https://huggingface.co/gaia-benchmark
- **Dataset:** https://huggingface.co/datasets/gaia-benchmark/GAIA
- **Examples:** 450
- **License:** see dataset page
- **Year:** 2023

### MINT
- **Description:** Evaluates LLMs' ability to solve tasks with multi-turn interactions by using tools and leveraging natural language feedback.
- **Paper:** MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback
https://arxiv.org/abs/2309.10691 
- **Code:** https://github.com/xingyaoww/mint-bench
- **Dataset:** https://github.com/xingyaoww/mint-bench/blob/main/docs/DATA.md
- **Examples:** 586
- **License:** see dataset page
- **Year:** 2023

### AgentBench 
- **Description:** A multi-dimensional benchmark to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting.
- **Paper:** AgentBench: Evaluating LLMs as Agents 
https://arxiv.org/abs/2308.03688
- **Code:** https://github.com/THUDM/AgentBench
- **Dataset:** https://github.com/THUDM/AgentBench/tree/main/data
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

### Webshop
- **Description:** A simulated e-commerce website environment with 1.18 million real-world products and 12,087 crowd-sourced text instructions. An agent needs to navigate multiple types of webpages, find, customize, and purchase an item.
- **Paper:** WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents 
https://arxiv.org/abs/2207.01206
- **Code:** https://github.com/princeton-nlp/webshop
- **Dataset:** https://huggingface.co/datasets/jyang/webshop_inst_goal_pairs_truth
- **Examples:** 529107
- **License:** MIT License
- **Year:** 2022

### PaperBench
- **Description:** A benchmark evaluating the ability of AI agents to replicate state-of-the-art AI research. Agents must replicate 20 ICML 2024 Spotlight and Oral papers from scratch, including understanding paper contributions, developing a codebase, and successfully executing experiments. 
- **Paper:** PaperBench: Evaluating AI's Ability to Replicate AI Research 
https://arxiv.org/abs/2504.01848 
- **Code:** https://github.com/openai/preparedness/blob/main/project/paperbench/README.md
- **Dataset:** https://github.com/openai/preparedness/blob/main/project/paperbench/README.md
- **Examples:** 8316
- **License:** see dataset page
- **Year:** 2025

### LLF-Bench
- **Description:** A benchmark that evaluates the ability of AI agents to interactively learn from natural language feedback and instructions.
- **Paper:** LLF-Bench: Benchmark for Interactive Learning from Language Feedback
https://arxiv.org/abs/2312.06853
- **Code:** https://github.com/microsoft/LLF-Bench
- **Dataset:** https://github.com/microsoft/LLF-Bench
- **Examples:** nan
- **License:** MIT License
- **Year:** 2023

### MultiAgentBench
- **Description:** A comprehensive benchmark designed to evaluate LLM-based multi-agent systems across diverse, interactive scenarios. It measures task completion and the quality of collaboration and competition.
- **Paper:** MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents
https://arxiv.org/html/2503.01935v1
- **Code:** https://github.com/ulab-uiuc/MARBLE
- **Dataset:** https://github.com/MultiagentBench/MARBLE/tree/main/multiagentbench
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2025

### CRMArena
- **Description:** A benchmark designed to evaluate AI agents on realistic tasks grounded in professional work environments. 
- **Paper:** CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments
https://arxiv.org/abs/2411.02305
- **Code:** https://github.com/SalesforceAIResearch/CRMArena
- **Dataset:** https://huggingface.co/datasets/Salesforce/CRMArena
- **Examples:** 1186
- **License:** CC-BY-NC-4.0
- **Year:** 2024

### CRMArena-Pro
- **Description:** A benchmark developed by Salesforce AI Research to evaluate LLM agents in realistic CRM (Customer Relationship Management) tasks
- **Paper:** CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions 
https://arxiv.org/abs/2505.18878
- **Code:** https://github.com/SalesforceAIResearch/CRMArena
- **Dataset:** https://huggingface.co/datasets/Salesforce/CRMArenaPro
- **Examples:** 8614
- **License:** CC-BY-NC-4.0
- **Year:** 2025

### FutureBench
- **Description:** A benchmarking system that tests agents’ ability to predict real-world outcomes using fresh news and prediction market events.
- **Paper:** Back to The Future: Evaluating AI Agents on Predicting Future Events 
https://huggingface.co/blog/futurebench
- **Code:** https://huggingface.co/spaces/togethercomputer/FutureBench
- **Dataset:** https://huggingface.co/spaces/togethercomputer/FutureBench
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2025

### SpreadsheetBench
- **Description:** A challenging spreadsheet manipulation benchmark exclusively derived from real-world scenarios.
- **Paper:** SpreadsheetBench: Towards Challenging Real World Spreadsheet Manipulation

https://arxiv.org/abs/2406.14991
- **Code:** https://github.com/RUCKBReasoning/SpreadsheetBench
- **Dataset:** https://github.com/RUCKBReasoning/SpreadsheetBench/tree/main/data
- **Examples:** 912
- **License:** CC-BY-SA-4.0
- **Year:** 2024

### TheAgentCompany
- **Description:** An extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. 
- **Paper:** TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks 

https://arxiv.org/abs/2412.14161
- **Code:** https://github.com/TheAgentCompany/TheAgentCompany
- **Dataset:** https://github.com/TheAgentCompany/TheAgentCompany/blob/main/workspaces/README.md
- **Examples:** 175
- **License:** see dataset page
- **Year:** 2024

### DSBench
- **Description:** DSBench evaluates large language and vision-language models on realistic data science tasks, including data analysis and data modeling tasks.
- **Paper:** DSBench: How Far Are Data Science Agents to Becoming Data Science Experts?
https://arxiv.org/abs/2409.07703
- **Code:** https://github.com/LiqiangJing/DSBench
- **Dataset:** https://github.com/LiqiangJing/DSBench?tab=readme-ov-file#usage
- **Examples:** 540
- **License:** see dataset page
- **Year:** 2024

### BrowseComp
- **Description:** A benchmark for measuring the ability of AI agents to browse the web. Comprises of questions that require persistently navigating the internet in search of hard-to-find, entangled information. 
- **Paper:** BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents
https://arxiv.org/abs/2504.12516
- **Code:** https://github.com/openai/simple-evals
- **Dataset:** https://github.com/openai/simple-evals
- **Examples:** 1266
- **License:** MIT License
- **Year:** 2025

### MLE-bench
- **Description:** A benchmark for measuring how well AI agents perform at machine learning engineering.
- **Paper:** MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
https://arxiv.org/abs/2410.07095 

- **Code:** https://github.com/openai/mle-bench
- **Dataset:** https://github.com/openai/mle-bench
- **Examples:** 75
- **License:** see dataset page
- **Year:** 2024

## 🗂 Agents & tools use,domain-specific

### TaxCalcBench
- **Description:** A benchmark for determining models' abilities to calculate personal income tax returns given all of the necessary information. 
- **Paper:** TaxCalcBench: Evaluating Frontier Models on the Tax Calculation Task 

https://arxiv.org/abs/2507.16126
- **Code:** https://github.com/column-tax/tax-calc-bench
- **Dataset:** https://github.com/column-tax/tax-calc-bench?tab=readme-ov-file#the-taxcalcbench-eval-ty24-dataset
- **Examples:** 51
- **License:** see dataset page
- **Year:** 2025

### SciGym
- **Description:** A benchmark that assesses LLMs’ iterative experiment design and analysis abilities in open-ended scientific discovery tasks. It challenges models to uncover biological mechanisms by designing and interpreting simulated experiments.
- **Paper:** Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab
https://arxiv.org/html/2507.02083v1 
- **Code:** https://github.com/h4duan/SciGym
- **Dataset:** https://huggingface.co/datasets/h4duan/scigym-sbml
- **Examples:** 350
- **License:** see dataset page
- **Year:** 2025

## 🗂 Agents & tools use,language & reasoning

### ACPBench
- **Description:** A benchmark for evaluating the reasoning tasks in the field of planning. The benchmark consists of 7 reasoning tasks over 13 planning domains.
- **Paper:** ACPBench: Reasoning about Action, Change, and Planning
https://arxiv.org/abs/2410.05669
- **Code:** https://github.com/ibm/ACPBench
- **Dataset:** https://huggingface.co/datasets/ibm-research/acp_bench
- **Examples:** 3210
- **License:** CDLA-Permissive-2.0
- **Year:** 2024

## 🗂 Bias & ethics

### Global MMLU
- **Description:** Translated MMLU, that also includes cultural sensitivity annotations for a subset of the questions, with evaluation coverage across 42 languages.
- **Paper:** Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation 
https://arxiv.org/abs/2412.03304 
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/CohereForAI/Global-MMLU
- **Examples:** 601734
- **License:** Apache-2.0 license
- **Year:** 2024

### Civil Comments
- **Description:** A suite of threshold-agnostic metrics for unintended bias and a test set of online comments with crowd-sourced annotations for identity references.
- **Paper:** Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification 
https://arxiv.org/abs/1903.04561
- **Code:** https://github.com/conversationai/conversationai.github.io/tree/main
- **Dataset:** https://huggingface.co/datasets/google/civil_comments
- **Examples:** 1999514
- **License:** CC0-1.0
- **Year:** 2019

## 🗂 Bias & ethics,knowledge

### SOCKET
- **Description:** A theory-driven benchmark containing 58 NLP tasks testing social knowledge, including humor, sarcasm, offensiveness, sentiment, emotion, and trustworthiness.
- **Paper:** Do LLMs Understand Social Knowledge? Evaluating the Sociability of Large Language Models with the SOCKET Benchmark. https://arxiv.org/pdf/2305.14938  
- **Code:** https://github.com/minjechoi/SOCKET
- **Dataset:** https://huggingface.co/datasets/Blablablab/SOCKET/tree/main/SOCKET_DATA 
- **Examples:** 58
- **License:** CC-BY-4.0
- **Year:** 2023

## 🗂 Coding

### CRUXEval (Code Reasoning, Understanding, and Execution Evaluation)
- **Description:** A set of Python functions and input-output pairs that consists of two tasks: input prediction and output prediction.
- **Paper:** CRUXEval: A Benchmark for Code Reasoning,
Understanding and Execution
https://arxiv.org/abs/2401.03065
- **Code:** https://github.com/facebookresearch/cruxeval
- **Dataset:** https://huggingface.co/datasets/cruxeval-org/cruxeval
- **Examples:** 800
- **License:** MIT License
- **Year:** 2024

### BigCodeBench
- **Description:** Function-level code generation tasks with complex instructions and diverse function calls.
- **Paper:** BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
https://arxiv.org/abs/2406.15877
- **Code:** https://github.com/bigcode-project/bigcodebench
- **Dataset:** https://github.com/bigcode-project/bigcodebench
- **Examples:** 1140
- **License:** Apache-2.0 license
- **Year:** 2024

### SWE-bench verified
- **Description:** A subset of SWE-bench, consisting of 500 samples verified to be non-problematic by our human annotators.
- **Paper:** Introducing SWE-bench Verified 
https://openai.com/index/introducing-swe-bench-verified/
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
- **Examples:** 500
- **License:** see dataset page
- **Year:** 2024

### CrossCodeEval
- **Description:** Multilingual code completion tasks built on built on real-world GitHub repositories in Python, Java, TypeScript, and C#.
- **Paper:** CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
https://arxiv.org/abs/2310.11248
- **Code:** https://github.com/amazon-science/cceval
- **Dataset:** https://github.com/amazon-science/cceval/tree/main/data
- **Examples:** 10000
- **License:** Apache-2.0 license
- **Year:** 2023

### EvalPlus
- **Description:** Extended HumanEval & MBPP by 80x/35x for rigorous eval.
- **Paper:** Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
https://arxiv.org/abs/2305.01210
- **Code:** https://github.com/evalplus/evalplus
- **Dataset:** https://github.com/evalplus/evalplus/tree/master/evalplus/data
- **Examples:** nan
- **License:** Apache-2.0 license
- **Year:** 2023

### ClassEval
- **Description:** Class-level Python code generation tasks.
- **Paper:** ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"
https://arxiv.org/abs/2308.01861
- **Code:** https://github.com/FudanSELab/ClassEval
- **Dataset:** https://huggingface.co/datasets/FudanSELab/ClassEval
- **Examples:** 100
- **License:** MIT License
- **Year:** 2023

### Repobench
- **Description:** Consists of three interconnected evaluation tasks: retrieve the most relevant code snippets, predict the next line of code, and handle complex tasks that require a combination of both retrieval and next-line prediction. Supports both Python and Java.
- **Paper:** RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems
https://arxiv.org/abs/2306.03091
- **Code:** https://github.com/Leolty/repobench
- **Dataset:** https://huggingface.co/datasets/tianyang/repobench-r 
https://huggingface.co/datasets/tianyang/repobench-c 
https://huggingface.co/datasets/tianyang/repobench-p
- **Examples:** unspecified
- **License:** CC-BY-NC-ND 4.0
- **Year:** 2023

### SWE-bench
- **Description:** Real-world software issues collected from GitHub.
- **Paper:** SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
https://arxiv.org/abs/2310.06770
- **Code:** https://github.com/princeton-nlp/SWE-bench
- **Dataset:** https://huggingface.co/datasets/princeton-nlp/SWE-bench
- **Examples:** 2200
- **License:** MIT License
- **Year:** 2023

### Code Lingua
- **Description:** Compares the ability of LLMs to understand what the code implements in source language and translate the same semantics in target language.
- **Paper:** Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code
https://arxiv.org/abs/2308.03109
- **Code:** https://github.com/codetlingua/codetlingua
- **Dataset:** https://huggingface.co/iidai
- **Examples:** 1700
- **License:** MIT License
- **Year:** 2023

### DS-1000
- **Description:** Code generation benchmark with data science problems spanning seven Python libraries, such as NumPy and Pandas.
- **Paper:** DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation
https://arxiv.org/abs/2211.11501
- **Code:** https://github.com/xlang-ai/DS-1000
- **Dataset:** https://huggingface.co/datasets/xlangai/DS-1000
- **Examples:** 1000
- **License:** CC-BY-SA-4.0
- **Year:** 2022

### CodeXGLUE
- **Description:** 14 datasets for program understanding and generation and three baseline systems, including the BERT-style, GPT-style, and Encoder-Decoder models.
- **Paper:** CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation
https://arxiv.org/abs/2102.04664
- **Code:** https://github.com/microsoft/CodeXGLUE
- **Dataset:** https://huggingface.co/datasets?search=code_x_glue
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2021

### APPS (Automated Programming Progress Standard)
- **Description:** A dataset for code generation, including introductory to competitive programming problems.
- **Paper:** Measuring Coding Challenge Competence With APPS
https://arxiv.org/abs/2105.09938
- **Code:** https://github.com/hendrycks/apps
- **Dataset:** https://huggingface.co/datasets/codeparrot/apps
- **Examples:** 10000
- **License:** MIT License
- **Year:** 2021

### MBPP (Mostly Basic Programming Problems)
- **Description:** Crowd-sourced entry-level programming tasks.
- **Paper:** Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732
- **Code:** https://github.com/google-research/google-research/blob/master/mbpp/README.md
- **Dataset:** https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl
- **Examples:** 974
- **License:** CC-BY-SA-4.0
- **Year:** 2021

### HumanEval
- **Description:** Programming tasks and unit tests to check model-generated code.
- **Paper:** Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374 
- **Code:** https://github.com/openai/human-eval
- **Dataset:** https://huggingface.co/datasets/openai/openai_humaneval
- **Examples:** 164
- **License:** MIT License
- **Year:** 2021

### APPS
- **Description:** A benchmark dataset for code generation and completion tasks, containing coding problems and solutions.
- **Paper:** Measuring Coding Challenge Competence With
APPS
https://arxiv.org/pdf/2105.09938
- **Code:** https://github.com/hendrycks/apps
- **Dataset:** https://huggingface.co/datasets/codeparrot/apps
- **Examples:** 10000
- **License:** MIT License
- **Year:** 2021

### LiveCodeBench 
- **Description:** A benchmark that evaluates the coding abilities of LLMs and contains problems from contests across three competition platforms - LeetCode, AtCoder, and CodeForces.
- **Paper:** LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code
https://arxiv.org/abs/2403.07974
- **Code:** https://livecodebench.github.io/
- **Dataset:** https://huggingface.co/livecodebench
- **Examples:** 1882
- **License:** see dataset page
- **Year:** 2024

### LiveCodeBench Pro
- **Description:** A benchmark composed of problems from Codeforces, ICPC, and IOI that are continuously updated to reduce the likelihood of data contamination. A team of Olympiad medalists annotates every problem.
- **Paper:** LiveCodeBench Pro: How Do Olympiad Medalists Judge LLMs in Competitive Programming?
https://arxiv.org/abs/2506.11928 
- **Code:** https://github.com/GavinZhengOI/LiveCodeBench-Pro
- **Dataset:** https://huggingface.co/datasets/anonymous1926/anonymous_dataset
- **Examples:** 785
- **License:** MIT License
- **Year:** 2025

### CodeElo
- **Description:** A standardized competition-level code generation benchmark.
- **Paper:** CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings 
https://arxiv.org/abs/2501.01257
- **Code:** https://github.com/QwenLM/CodeElo
- **Dataset:** https://huggingface.co/datasets/Qwen/CodeElo
- **Examples:** 408
- **License:** Apache-2.0 license
- **Year:** 2025

### ResearchCodeBench
- **Description:** A benchmark that evaluates LLMs’ ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code.
- **Paper:** ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code 

https://arxiv.org/html/2506.02314v1
- **Code:** https://github.com/PatrickHua/ResearchCodeBench
- **Dataset:** https://researchcodebench.github.io/leaderboard/index.html
- **Examples:** 212
- **License:** see dataset page
- **Year:** 2025

### Spider 2.0
- **Description:** An evaluation framework comprising real-world text-to-SQL workflow problems derived from enterprise-level database use cases. 
- **Paper:** Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows
https://arxiv.org/abs/2411.07763
- **Code:** https://github.com/xlang-ai/Spider2
- **Dataset:** https://github.com/xlang-ai/Spider2?tab=readme-ov-file#data
- **Examples:** 632
- **License:** see dataset page
- **Year:** 2024

### SciCode
- **Description:** A benchmark that challenges language models to code solutions for scientific problems.
- **Paper:** SciCode: A Research Coding Benchmark Curated by Scientists
https://arxiv.org/abs/2407.13168 
- **Code:** https://github.com/scicode-bench/SciCode
- **Dataset:** https://huggingface.co/datasets/SciCode1/SciCode
- **Examples:** 80
- **License:** Apache-2.0 license
- **Year:** 2024

## 🗂 Conversation & chatbots

### MultiChallenge
- **Description:** Evaluates LLMs on conducting multi-turn conversations with human users across 4 challenges: instruction retention, inference memory, reliable versioned editing, and self-coherence. 
- **Paper:** MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs 
https://arxiv.org/abs/2501.17399
- **Code:** https://github.com/ekwinox117/multi-challenge
- **Dataset:** https://github.com/ekwinox117/multi-challenge/tree/main/data
- **Examples:** 273
- **License:** see dataset page
- **Year:** 2025

### MT-Bench-101
- **Description:** Multi-turn dialogues.
- **Paper:** MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues
https://arxiv.org/abs/2402.14762 
- **Code:** https://github.com/mtbench101/mt-bench-101
- **Dataset:** https://github.com/mtbench101/mt-bench-101/tree/main/data/subjective
- **Examples:** 4208
- **License:** Apache-2.0 license
- **Year:** 2024

### Chatbot Arena
- **Description:** Open-source platform for comparing LLMs in a competitive environment.
- **Paper:** Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference
https://arxiv.org/abs/2403.04132

- **Code:** https://github.com/lm-sys/FastChat/tree/main
- **Dataset:** https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
- **Examples:** 33000
- **License:** see dataset page
- **Year:** 2024

### MixEval
- **Description:** A ground-truth-based dynamic benchmark derived from off-the-shelf benchmark mixtures.
- **Paper:** MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures
https://arxiv.org/abs/2406.06565
- **Code:** https://github.com/Psycoy/MixEval
- **Dataset:** https://huggingface.co/datasets/MixEval/MixEval
- **Examples:** 5000
- **License:** Apache-2.0 license
- **Year:** 2024

### WildChat
- **Description:** A collection of 1 million conversations between human users and ChatGPT, alongside demographic data (https://wildchat.allen.ai/about). 
- **Paper:** WildChat: 1M ChatGPT Interaction Logs in the Wild
https://arxiv.org/abs/2405.01470
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/allenai/WildChat-1M
- **Examples:** 1,000,000+
- **License:** ODC-BY license
- **Year:** 2024

### Arena-Hard 
- **Description:** Automatic evaluation tool for instruction-tuned LLMs, contains 500 challenging user queries sourced from Chatbot Arena. 
- **Paper:** From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline
https://arxiv.org/abs/2406.11939
- **Code:** https://github.com/lmarena/arena-hard-auto
- **Dataset:** https://huggingface.co/spaces/lmarena-ai/arena-hard-browser
- **Examples:** 500
- **License:** see dataset page
- **Year:** 2024

### MT-Bench
- **Description:** Multi-turn questions: an open-ended question and a follow-up question.
- **Paper:** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena 
https://arxiv.org/abs/2306.05685
- **Code:** https://github.com/lm-sys/FastChat/tree/main
- **Dataset:** https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
- **Examples:** 3300
- **License:** CC-BY-4.0
- **Year:** 2023

### OpenDialKG
- **Description:** A dataset of conversations between two crowdsourcing
agents engaging in a dialog about a given topic.
- **Paper:** OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs
https://aclanthology.org/P19-1081/
- **Code:** https://github.com/facebookresearch/opendialkg
- **Dataset:** https://github.com/facebookresearch/opendialkg/tree/main/data
- **Examples:** 15000
- **License:** CC-BY-NC-4.0
- **Year:** 2019

### CoQA (Conversational Question Answering)
- **Description:** Questions with answers collected from 8000+ conversations.
- **Paper:** CoQA: A Conversational Question Answering Challenge
https://arxiv.org/abs/1808.07042
- **Code:** https://stanfordnlp.github.io/coqa/
- **Dataset:** https://stanfordnlp.github.io/coqa/
- **Examples:** 127000
- **License:** see dataset page
- **Year:** 2018

### QuAC (Question Answering in Context)
- **Description:** Question-answer pairs, simulating student-teacher interactions.
- **Paper:** QuAC : Question Answering in Context
https://arxiv.org/abs/1808.07036
- **Code:** https://quac.ai/
- **Dataset:** https://quac.ai/
- **Examples:** 100000
- **License:** CC-BY-SA-4.0
- **Year:** 2018

### SPC (Synthetic-Persona-Chat Dataset)
- **Description:** A persona-based conversational dataset, consisting of synthetic personas and conversations.
- **Paper:** Faithful Persona-based Conversational Dataset Generation with Large Language Models  https://arxiv.org/abs/2312.10007 
- **Code:** https://github.com/google-research-datasets/Synthetic-Persona-Chat/tree/main 
- **Dataset:** https://huggingface.co/datasets/google/Synthetic-Persona-Chat 
- **Examples:** 10000+
- **License:** CC-BY-4.0
- **Year:** 2023

### Wildbench
- **Description:** An automated evaluation framework designed to benchmark LLMs on real-world user queries. It consists of 1,024 tasks selected from over one million human-chatbot conversation logs. 
- **Paper:** WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild 
https://arxiv.org/abs/2406.04770 
- **Code:** https://github.com/allenai/WildBench
- **Dataset:** https://huggingface.co/datasets/allenai/WildBench
- **Examples:** 1024
- **License:** CC-BY-4.0
- **Year:** 2024

### SocialDial
- **Description:** A socially-aware dialogue corpus that covers five categories of social norms, including social relation, context, and social distance.
- **Paper:** SocialDial: A Benchmark for Socially-Aware Dialogue Systems 
https://arxiv.org/abs/2304.12026
- **Code:** https://github.com/zhanhl316/SocialDial
- **Dataset:** https://github.com/zhanhl316/SocialDial/blob/main/human_dialogue_data.json
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

## 🗂 Decision-making

### Contrast Sets
- **Description:** Annotation paradigm for NLP that helps to close systematic gaps in the test data. Contrast sets provide a local view of a model's decision boundary, which can be used to more accurately evaluate a model's true linguistic capabilities.
- **Paper:** Evaluating Models' Local Decision Boundaries via Contrast Sets 
https://arxiv.org/abs/2004.02709
- **Code:** https://github.com/allenai/contrast-sets
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2020

## 🗂 Domain-specific

### ClinicBench
- **Description:** Datasets and clinical tasks that are common in real-world medical practice, e.g., open-ended decision-making, long document processing, and emerging drug analysis.
- **Paper:** Large Language Models in the Clinic: A Comprehensive Benchmark 
https://arxiv.org/abs/2405.00716
- **Code:** https://github.com/AI-in-Health/ClinicBench
- **Dataset:** https://github.com/AI-in-Health/ClinicBench
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

### LegalBench
- **Description:** Collaboratively curated tasks for evaluating legal reasoning in English LLMs.
- **Paper:** LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models
https://arxiv.org/abs/2308.11462
- **Code:** https://github.com/HazyResearch/legalbench/
- **Dataset:** https://huggingface.co/datasets/nguha/legalbench
- **Examples:** 162
- **License:** see dataset page
- **Year:** 2023

### MedMCQA
- **Description:** Four-option multiple-choice questions from Indian medical entrance examinations. Covers 2,400 healthcare topics and 21 medical subjects.
- **Paper:** MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering
https://arxiv.org/abs/2203.14371
- **Code:** https://github.com/medmcqa/medmcqa
- **Dataset:** https://github.com/medmcqa/medmcqa
- **Examples:** 194000
- **License:** MIT License
- **Year:** 2022

### TAT-QA 
- **Description:** Questions and associated hybrid contexts from real-world financial reports.
- **Paper:** TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
https://arxiv.org/abs/2105.07624
- **Code:** https://github.com/NExTplusplus/TAT-QA
- **Dataset:** https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw
- **Examples:** 16552
- **License:** MIT License
- **Year:** 2021

### CUAD
- **Description:** A dataset for legal contract review with over 13,000 annotations.
- **Paper:** CUAD: An Expert-Annotated NLP Dataset for
Legal Contract Review
https://arxiv.org/pdf/2103.06268
- **Code:** https://github.com/TheAtticusProject/cuad
- **Dataset:** https://huggingface.co/datasets/theatticusproject/cuad-qa
- **Examples:** 13000
- **License:** CC-BY-4.0
- **Year:** 2021

### MedQA
- **Description:** Free-form multiple-choice OpenQA dataset for solving medical problems collected from the professional medical board exams.
- **Paper:** What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/abs/2009.13081
- **Code:** https://github.com/jind11/MedQA
- **Dataset:** https://github.com/jind11/MedQA
- **Examples:** 12723
- **License:** MIT License
- **Year:** 2020

### PubMedQA
- **Description:** A dataset for biomedical research question answering.
- **Paper:** PubMedQA: A Dataset for Biomedical Research Question Answering
https://arxiv.org/abs/1909.06146
- **Code:** https://github.com/pubmedqa/pubmedqa
- **Dataset:** https://github.com/pubmedqa/pubmedqa
- **Examples:** 270000
- **License:** MIT License
- **Year:** 2019

### MedConceptsQA
- **Description:** MedConceptsQA measures the ability of models to interpret and distinguish between medical codes for diagnoses, procedures, and drugs. 
- **Paper:** MedConceptsQA: Open source medical concepts QA benchmark 
https://www.sciencedirect.com/science/article/pii/S0010482524011740 
- **Code:** https://github.com/nadavlab/MedConceptsQA
- **Dataset:** https://huggingface.co/datasets/ofir408/MedConceptsQA
- **Examples:** 819829
- **License:** Apache-2.0 license
- **Year:** 2024

### CUPCase
- **Description:** CUPCase is based on 3,563 real-world clinical case reports formulated into diagnoses in open-ended textual format and as multiple-choice options with distractors.
- **Paper:** CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset 
https://arxiv.org/abs/2503.06204
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/ofir408/CupCase
- **Examples:** 3562
- **License:** Apache-2.0 license
- **Year:** 2025

### LAB-Bench (Language Agent Biology Benchmark)
- **Description:** An evaluation dataset for AI systems intended to benchmark capabilities foundational to scientific research in biology. 
- **Paper:** LAB-Bench: Measuring Capabilities of Language Models for Biology Research 
https://arxiv.org/abs/2407.10362 
- **Code:** https://github.com/Future-House/LAB-Bench
- **Dataset:** https://huggingface.co/datasets/futurehouse/lab-bench
- **Examples:** 2000
- **License:** CC-BY-SA-4.0
- **Year:** 2024

### PERRECBENCH
- **Description:** A novel benchmark for evaluating how well LLMs understand user preferences in recommendation systems.
- **Paper:** https://arxiv.org/abs/2501.13391
- **Code:** https://github.com/TamSiuhin/PerRecBench
- **Dataset:** https://github.com/TamSiuhin/PerRecBench?tab=readme-ov-file#download-data
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2025

## 🗂 Domain-specific,agents & tools use,information retrieval & rag

### DIBS (Domain Intelligence Benchmark Suite)
- **Description:** DIBS measures LLM performance on datasets curated to reflect specialized domain knowledge and common enterprise use cases that traditional academic benchmarks often overlook.
- **Paper:** Benchmarking Domain Intelligence 
https://www.databricks.com/blog/benchmarking-domain-intelligence
- **Code:** _No repository provided_
- **Dataset:** _No dataset link provided_
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

## 🗂 Domain-specific,language & reasoning

### MediQ
- **Description:** A framework for simulating realistic clinical interactions, where an Expert model asks information-seeking questions when needed and respond reliably. 
- **Paper:** MediQ: Question-Asking LLMs and a Benchmark for Reliable Interactive Clinical Reasoning

https://arxiv.org/abs/2406.00922
- **Code:** https://github.com/stellalisy/mediQ
- **Dataset:** https://drive.google.com/drive/folders/1ZPGfr-iftLsQDLkwyNYRg5ERwpuCtLg_
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

## 🗂 Empathy

### EmotionBench
- **Description:** Evaluates the empathy ability of LLMs across 8 emotions: anger, anxiety, depression, frustration, jealousy, guilt, fear, embarrassment.
- **Paper:** Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench 
https://arxiv.org/abs/2308.03656
- **Code:** https://github.com/CUHK-ARISE/EmotionBench
- **Dataset:** https://huggingface.co/datasets/CUHK-ARISE/EmotionBench
- **Examples:** 400
- **License:** Apache-2.0 license
- **Year:** 2023

### EQ-Bench
- **Description:** Assesses the ability of LLMs to understand complex emotions and social interactions by asking them to predict the intensity of emotional states of characters in a dialogue.
- **Paper:** EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models 
https://arxiv.org/abs/2312.06281
- **Code:** https://github.com/EQ-bench/EQ-Bench
- **Dataset:** https://huggingface.co/datasets/pbevan11/EQ-Bench
- **Examples:** 171
- **License:** MIT License
- **Year:** 2023

## 🗂 Image generation

### MC-Bench (Minecraft AI Benchmark)
- **Description:** A platform for evaluating and comparing AI models by challenging them to create Minecraft builds.
- **Paper:** https://mcbench.ai/
- **Code:** https://github.com/mc-bench
- **Dataset:** Not dataset-based
- **Examples:** nan
- **License:** MIT License
- **Year:** 2024

## 🗂 Information retrieval & rag

### NoLiMa
- **Description:** Extended NIAH, where questions and needles have minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack.
- **Paper:** NoLiMa: Long-Context Evaluation Beyond Literal Matching 
https://arxiv.org/abs/2502.05167
- **Code:** https://github.com/adobe-research/NoLiMa
- **Dataset:** https://huggingface.co/datasets/amodaresi/NoLiMa
- **Examples:** 7540
- **License:** Adobe Research License
- **Year:** 2025

### RULER
- **Description:** A synthetic benchmark with flexible configurations for customized sequence length and task complexity. RULER expands upon the vanilla NIAH test to encompass variations with diverse types and quantities of needles.
- **Paper:** RULER: What's the Real Context Size of Your Long-Context Language Models? 
https://arxiv.org/abs/2404.06654
- **Code:** https://github.com/NVIDIA/RULER
- **Dataset:** see repo
- **Examples:** 13
- **License:** see dataset page
- **Year:** 2024

### Loong
- **Description:** Long-context benchmark, aligning with realistic scenarios through extended multi-document question answering (QA).
- **Paper:** Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA 
https://arxiv.org/abs/2406.17419
- **Code:** https://github.com/MozerWang/Loong
- **Dataset:** (in Chinese) https://modelscope.cn/datasets/iic/Loong
- **Examples:** 1600
- **License:** see dataset page
- **Year:** 2024

### WiCE
- **Description:** Textual entailment dataset built on natural claim and evidence pairs extracted from Wikipedia.
- **Paper:** WiCE: Real-World Entailment for Claims in Wikipedia 
https://arxiv.org/abs/2303.01432
- **Code:** https://github.com/ryokamoi/wice
- **Dataset:** https://huggingface.co/datasets/tasksource/wice
- **Examples:** 5377
- **License:** CC-BY-SA-4.0
- **Year:** 2023

### LFQA-Verification
- **Description:** Tests how retrieval augmentation impacts different LMs. Compares answers generated while using the same evidence documents by different LMs, and how differing quality of retrieval documents impacts the answers generated from the same LM.
- **Paper:** Understanding Retrieval Augmentation for Long-Form Question Answering 
https://arxiv.org/abs/2310.12150
- **Code:** https://github.com/timchen0618/LFQA-Verification/
- **Dataset:** https://github.com/timchen0618/LFQA-Verification/tree/main/data
- **Examples:** 100
- **License:** see dataset page
- **Year:** 2023

### FEVER
- **Description:** A dataset for verification against textual sources, FEVER: Fact Extraction and VERification.
- **Paper:** FEVER: a large-scale dataset for Fact Extraction and VERification 
https://arxiv.org/abs/1803.05355
- **Code:** https://github.com/awslabs/fever
- **Dataset:** https://fever.ai/dataset/fever.html
- **Examples:** 185445
- **License:** see dataset page
- **Year:** 2018

### NeedleInAHaystack (NIAH)
- **Description:** A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.
- **Paper:** nan
- **Code:** https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main
- **Dataset:** https://huggingface.co/datasets/YurtsAI/NIAH_eval_dataset
- **Examples:** 215
- **License:** MIT License
- **Year:** N/A

### CRAG
- **Description:** A factual question answering benchmark of question-answer pairs and mock APIs to simulate web and Knowledge Graph (KG) search.
- **Paper:** CRAG -- Comprehensive RAG Benchmark 
https://arxiv.org/abs/2406.04744
- **Code:** https://github.com/facebookresearch/CRAG
- **Dataset:** https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md
- **Examples:** 4409
- **License:** see dataset page
- **Year:** 2024

### LongGenBench
- **Description:** A synthetic benchmark that allows for flexible configurations of customized generation context lengths.
- **Paper:** LongGenBench: Long-context Generation Benchmark 
https://arxiv.org/abs/2410.04199
- **Code:** https://github.com/mozhu621/LongGenBench
- **Dataset:** https://huggingface.co/datasets/mozhu/LongGenBench
- **Examples:** nan
- **License:** CC-BY-ND-4.0
- **Year:** 2024

### FaithEval
- **Description:** Comprehensive evaluation of how well LLMs can align their responses with the context.
- **Paper:** FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows" 
https://arxiv.org/abs/2410.03727
- **Code:** https://github.com/SalesforceAIResearch/FaithEval
- **Dataset:** https://huggingface.co/collections/Salesforce/faitheval-benchmark-66ff102cda291ca0875212d4
- **Examples:** 4900
- **License:** see dataset page
- **Year:** 2024

### MTRAG
- **Description:** An end-to-end human-generated multi-turn RAG benchmark that reflects several real-world properties across diverse dimensions for evaluating the full RAG pipeline. 
- **Paper:** MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems 

https://arxiv.org/abs/2501.03468
- **Code:** https://github.com/ibm/mt-rag-benchmark
- **Dataset:** https://github.com/ibm/mt-rag-benchmark?tab=readme-ov-file#human-data
- **Examples:** 842
- **License:** see dataset page
- **Year:** 2025

### ContextualBench
- **Description:** A compilation of 7 popular contextual question answering benchmarks to evaluate LLMs in RAG application.
- **Paper:** nan
- **Code:** https://github.com/SalesforceAIResearch/SFR-RAG/blob/main/README_ContextualBench.md
- **Dataset:** https://huggingface.co/datasets/Salesforce/ContextualBench
- **Examples:** 215527
- **License:** see dataset page
- **Year:** N/A

### WixQA
- **Description:** A benchmark suite featuring QA datasets grounded in the released knowledge base corpus, enabling holistic evaluation of retrieval and generation components. 
- **Paper:** https://arxiv.org/abs/2505.08643 
WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/Wix/WixQA
- **Examples:** 12842
- **License:** MIT License
- **Year:** 2025

## 🗂 Information retrieval & rag,language & reasoning,domain-specific

### OmniEval
- **Description:** RAG benchmark in the financial domain, with queries in five task classes and 16 financial topics.
- **Paper:** OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain 
https://arxiv.org/abs/2412.13018
- **Code:** https://github.com/RUC-NLPIR/OmniEval
- **Dataset:** see repo
- **Examples:** nan
- **License:** MIT License
- **Year:** 2024

## 🗂 Information retrieval & rag,language & reasoning,safety

### FRAMES (Factuality, Retrieval, And reasoning MEasurement Set)
- **Description:** Tests the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.
- **Paper:** Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation 
https://arxiv.org/abs/2409.12941
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/google/frames-benchmark
- **Examples:** 824
- **License:** Apache-2.0 license
- **Year:** 2024

## 🗂 Information retrieval & rag,multimodal

### ViDoRe (Visual Document Retrieval Benchmark)
- **Description:** A benchmark to evaluate LLMs on visually rich document retrieval.
- **Paper:** ColPali: Efficient Document Retrieval with Vision Language Models 
https://arxiv.org/abs/2407.01449
- **Code:** https://github.com/illuin-tech/vidore-benchmark
- **Dataset:** https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

## 🗂 Information retrieval & rag,safety

### RAGTruth
- **Description:** A corpus tailored for analyzing word-level hallucinations within the standard RAG frameworks for LLM applications. RAGTruth comprises 18,000 naturally generated responses from diverse LLMs using RAG.
- **Paper:** RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models 
https://arxiv.org/abs/2401.00396
- **Code:** https://github.com/ParticleMedia/RAGTruth
- **Dataset:** https://huggingface.co/datasets/wandb/RAGTruth-processed
- **Examples:** 18000
- **License:** see dataset page
- **Year:** 2023

## 🗂 Instruction-following

### Infobench
- **Description:** Evaluating Large Language Models' (LLMs) ability to follow instructions by breaking complex instructions into simpler criteria, facilitating a detailed analysis of LLMs' compliance with various aspects of tasks.
- **Paper:** INFOBENCH: Evaluating Instruction Following Ability in Large Language Models 
https://arxiv.org/abs/2401.03601
- **Code:** https://github.com/qinyiwei/InfoBench
- **Dataset:** https://huggingface.co/datasets/kqsong/InFoBench
- **Examples:** 500
- **License:** MIT License
- **Year:** 2024

### PandaLM
- **Description:** A judge large language model which is trained to distinguish the superior model given several LLMs. It compares the responses of different LLMs and provide a reason for the decision, along with a reference answer.
- **Paper:** PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization
https://arxiv.org/abs/2306.05087
- **Code:** https://github.com/WeOpenML/PandaLM
- **Dataset:** https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvYy8xZDM3ZWRlNmVhYTk3NGRkL0VkMTBxZXJtN1RjZ2dCMnJBZ0FBQUFBQk5hbTM2YVExNlpjTU1IMjFaVU85ZlE%5FZT1nTjZueFI&cid=1D37EDE6EAA974DD&id=1D37EDE6EAA974DD%21683&parId=1D37EDE6EAA974DD%21682&o=OneUp
- **Examples:** 1000
- **License:** see dataset page
- **Year:** 2023

## 🗂 Instruction-following,conversation & chatbots

### AlpacaEval
- **Description:** An automatic evaluator for instruction-following LLMs.
- **Paper:** Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators
https://arxiv.org/abs/2404.04475
- **Code:** https://github.com/tatsu-lab/alpaca_eval
- **Dataset:** https://huggingface.co/datasets/tatsu-lab/alpaca_eval
- **Examples:** nan
- **License:** CC-BY-NC-4.0
- **Year:** 2024

## 🗂 Knowledge

### ConflictBank
- **Description:** Evaluates knowledge conflicts from three aspects: 1) conflicts in retrieved knowledge, 2) conflicts within the models’ encoded knowledge, and 3) the interplay between these conflict forms. 
- **Paper:** ConflictBank: A Benchmark for Evaluating Knowledge Conflicts in Large Language Models 
https://arxiv.org/html/2408.12076v1
- **Code:** https://github.com/zhaochen0110/conflictbank
- **Dataset:** see repo
- **Examples:** 553000
- **License:** CC-BY-SA-4.0
- **Year:** 2024

### FreshQA
- **Description:** Tests factuality of LLM-generated text in the context of answering questions that test current world knowledge. The dataset is updated weekly.
- **Paper:** FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation
https://arxiv.org/abs/2310.03214
- **Code:** https://github.com/freshllms/freshqa
- **Dataset:** https://github.com/freshllms/freshqa?tab=readme-ov-file#freshqa
- **Examples:** 599
- **License:** Apache-2.0 license
- **Year:** 2023

## 🗂 Knowledge,language & reasoning

### MMLU Pro
- **Description:** An enhanced dataset designed to extend the MMLU benchmark. More challenging questions, the choice set of ten options.
- **Paper:** MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
https://arxiv.org/abs/2406.01574
- **Code:** https://github.com/TIGER-AI-Lab/MMLU-Pro
- **Dataset:** https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- **Examples:** 12100
- **License:** MIT License
- **Year:** 2024

### BigBench Hard
- **Description:** A suite of BigBench tasks for which LLMs did not outperform the average human-rater.
- **Paper:** Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
https://arxiv.org/abs/2210.09261
- **Code:** https://github.com/suzgunmirac/BIG-Bench-Hard
- **Dataset:** https://huggingface.co/datasets/maveriq/bigbenchhard
- **Examples:** 6500
- **License:** MIT License
- **Year:** 2022

### BigBench
- **Description:** Set of questions crowdsourced by domain experts in math, biology, physics, and beyond.
- **Paper:** Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models
https://arxiv.org/abs/2206.04615
- **Code:** https://github.com/google/BIG-bench
- **Dataset:** https://huggingface.co/datasets/google/bigbench
- **Examples:** nan
- **License:** Apache-2.0 license
- **Year:** 2022

### MMLU
- **Description:** Multi-choice tasks across 57 subjects, high school to expert level.
- **Paper:** Measuring Massive Multitask Language Understanding
https://arxiv.org/abs/2009.03300
- **Code:** https://github.com/hendrycks/test/tree/master
- **Dataset:** https://huggingface.co/datasets/cais/mmlu
- **Examples:** 231400
- **License:** MIT License
- **Year:** 2020

### ARC
- **Description:** Grade-school level, multiple-choice science questions.
- **Paper:** Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/abs/1803.05457
- **Code:** https://github.com/allenai/aristo-leaderboard/tree/master/arc
- **Dataset:** https://huggingface.co/datasets/allenai/ai2_arc
- **Examples:** 7787
- **License:** CC-BY-SA-4.0
- **Year:** 2018

### Humanity's Last Exam (HLE)
- **Description:** A multi-modal benchmark at the frontier of human knowledge, consists of 2,500 questions across dozens of subjects, including mathematics, humanities, and the natural sciences.
- **Paper:** Humanity's Last Exam 
https://arxiv.org/abs/2501.14249 
- **Code:** https://github.com/centerforaisafety/hle
- **Dataset:** https://huggingface.co/datasets/cais/hle
- **Examples:** 2500
- **License:** MIT License
- **Year:** 2025

### MegaScience
- **Description:** A large-scale mixture of high-quality open-source datasets totaling 1.25 million instances.
- **Paper:** MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning 
https://arxiv.org/pdf/2507.16812
- **Code:** https://github.com/GAIR-NLP/MegaScience
- **Dataset:** https://huggingface.co/datasets/MegaScience/MegaScience
- **Examples:** 1.25M+
- **License:** CC-BY-NC-SA-4.0
- **Year:** 2025

### SKA-Bench
- **Description:** A Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: Knowledge Graph (KG), Table, KG+Text, and Table+Text.
- **Paper:** SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs 

https://arxiv.org/abs/2507.17178
- **Code:** https://github.com/Lza12a/SKA-Bench
- **Dataset:** https://github.com/Lza12a/SKA-Bench
- **Examples:** 2100
- **License:** see dataset page
- **Year:** 2025

## 🗂 Knowledge,language & reasoning,multimodal

### ScienceQA
- **Description:** Multimodal multiple choice questions with diverse science topics and annotations of their answers with corresponding lectures and explanations.
- **Paper:** Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
https://arxiv.org/abs/2209.09513
- **Code:** https://github.com/lupantech/ScienceQA
- **Dataset:** https://huggingface.co/datasets/derek-thomas/ScienceQA
- **Examples:** 21208
- **License:** CC-BY-SA-4.0
- **Year:** 2022

## 🗂 Knowledge,language & reasoning,safety

### TruthfulQA
- **Description:** Evaluates how well models generate truthful responses.
- **Paper:** TruthfulQA: Measuring How Models Mimic Human Falsehoods
https://arxiv.org/abs/2109.07958v2
- **Code:** https://github.com/sylinrl/TruthfulQA
- **Dataset:** https://huggingface.co/datasets/truthfulqa/truthful_qa
- **Examples:** 1634
- **License:** Apache-2.0 license
- **Year:** 2021

## 🗂 Language & reasoning

### Graphwalks
- **Description:** A dataset for evaluating multi-hop long-context reasoning. In Graphwalks, the model is given a graph represented by its edge list and asked to perform an operation.
- **Paper:** Introducing GPT-4.1 in the API 
https://openai.com/index/gpt-4-1/
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/openai/graphwalks
- **Examples:** 1150
- **License:** MIT License
- **Year:** 2025

### Zebralogic
- **Description:** Evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). 
- **Paper:** ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning 
https://arxiv.org/abs/2502.01100
- **Code:** https://github.com/WildEval/ZeroEval
- **Dataset:** https://huggingface.co/datasets/WildEval/ZebraLogic
- **Examples:** 4259
- **License:** see dataset page
- **Year:** 2024

### NovelQA
- **Description:** Tests deep textual understanding in LLMs with extended texts. Constructed from English novels.
- **Paper:** NovelQA: Benchmarking Question Answering on Documents Exceeding 200K Tokens 
https://arxiv.org/abs/2403.12766
- **Code:** https://github.com/NovelQA/novelqa.github.io
- **Dataset:** https://huggingface.co/datasets/NovelQA/NovelQA
- **Examples:** 2305
- **License:** Apache-2.0 license
- **Year:** 2024

### Reveal 
- **Description:** A dataset to benchmark automatic verifiers of complex Chain-of-Thought reasoning in open-domain question-answering settings
- **Paper:** A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains 
https://arxiv.org/abs/2402.00559
- **Code:** https://reveal-dataset.github.io
- **Dataset:** https://huggingface.co/datasets/google/reveal
- **Examples:** 6102
- **License:** CC-BY-ND-4.0
- **Year:** 2024

### LongBench
- **Description:** Assesses the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. Consists of multiple-choice questions, with contexts ranging from 8k to 2M words.
- **Paper:** LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks 
https://arxiv.org/abs/2412.15204
- **Code:** https://github.com/THUDM/LongBench
- **Dataset:** https://huggingface.co/datasets/THUDM/LongBench-v2
- **Examples:** 503
- **License:** Apache-2.0 license
- **Year:** 2024

### InfiniteBench
- **Description:** Evaluates the capabilities of language models to process, understand, and reason over super long contexts (100k+ tokens).
- **Paper:** ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens 
https://arxiv.org/abs/2402.13718
- **Code:** https://github.com/OpenBMB/InfiniteBench
- **Dataset:** https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

### Chain-of-Thought Hub
- **Description:** Curated complex reasoning tasks including math, science, coding, long-context.
- **Paper:** Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance
https://arxiv.org/abs/2305.17306
- **Code:** https://github.com/FranxYao/chain-of-thought-hub/
- **Dataset:** see repository 
- **Examples:** 1000+
- **License:** MIT License
- **Year:** 2023

### MuSR
- **Description:** Multistep reasoning tasks based on text narratives (e.g., 1000 words murder mysteries).
- **Paper:** https://arxiv.org/abs/2310.16049
- **Code:** https://github.com/Zayne-sprague/MuSR
- **Dataset:** https://github.com/Zayne-sprague/MuSR/tree/main/datasets
- **Examples:** 756
- **License:** MIT License
- **Year:** 2023

### GPQA
- **Description:** A set of multiple-choice questions written by domain experts in biology, physics, and chemistry.
- **Paper:** GPQA: A Graduate-Level Google-Proof Q&A Benchmark
https://arxiv.org/abs/2311.12022
- **Code:** https://github.com/idavidrein/gpqa
- **Dataset:** https://huggingface.co/datasets/Idavidrein/gpqa
- **Examples:** 448
- **License:** CC-BY-4.0
- **Year:** 2023

### AGIEval
- **Description:** A collection of standardized tests, including GRE, GMAT, SAT, LSAT.
- **Paper:** AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models
https://arxiv.org/abs/2304.06364
- **Code:** https://github.com/ruixiangcui/AGIEval/tree/main
- **Dataset:** https://github.com/ruixiangcui/AGIEval/tree/main/data

- **Examples:** nan
- **License:** MIT License
- **Year:** 2023

### SummEdits
- **Description:** Inconsistency detection in summaries
- **Paper:** https://arxiv.org/abs/2305.14540
- **Code:** https://github.com/salesforce/factualNLG
- **Dataset:** https://github.com/salesforce/factualNLG/tree/master/data/summedits
- **Examples:** 6,348
- **License:** Apache-2.0 license
- **Year:** 2023

### NPHardEval
- **Description:** Evaluates the reasoning abilities of LLMs across a broad spectrum of 900 algorithmic questions, extending up to the NP-Hard complexity class.
- **Paper:** NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes 
https://arxiv.org/abs/2312.14890
- **Code:** https://github.com/casmlab/NPHardEval
- **Dataset:** https://github.com/casmlab/NPHardEval
- **Examples:** 900
- **License:** see dataset page
- **Year:** 2023

### e-CARE (explainable CAusal REasoning dataset)
- **Description:** A human-annotated dataset that contains causal reasoning questions.
- **Paper:** e-CARE: a New Dataset for Exploring Explainable Causal Reasoning
https://arxiv.org/abs/2205.05849
- **Code:** https://github.com/Waste-Wood/e-CARE
- **Dataset:** https://github.com/Waste-Wood/e-CARE/tree/main/dataset
- **Examples:** 21000
- **License:** MIT License
- **Year:** 2022

### PlanBench
- **Description:** A benchmark designed to evaluate the ability of LLMs to generate plans of action and reason about change.
- **Paper:** PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change
https://arxiv.org/abs/2206.10498
- **Code:** https://github.com/karthikv792/LLMs-Planning/tree/main/plan-bench
- **Dataset:** https://huggingface.co/datasets/tasksource/planbench
- **Examples:** 11113
- **License:** see dataset page
- **Year:** 2022

### GLUE-X
- **Description:** Includes 13 publicly available datasets for Out-of-distribution testing, and evaluations are conducted on 8 classic NLP tasks over 21 popularly used PLMs, including GPT-3 and GPT-3.5
- **Paper:** GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective 
https://arxiv.org/abs/2211.08073
- **Code:** https://github.com/YangLinyi/GLUE-X
- **Dataset:** https://drive.google.com/drive/folders/1BcwjmVOqq96igfbB2MCXwLzthFX7XEhy
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2022

### FOLIO
- **Description:** A human-annotated, logically complex dataset for reasoning in natural language, equipped with first-order logic (FOL) annotations.
- **Paper:** FOLIO: Natural Language Reasoning with First-Order Logic 
https://arxiv.org/abs/2209.00840
- **Code:** https://github.com/Yale-LILY/FOLIO
- **Dataset:** https://huggingface.co/datasets/yale-nlp/FOLIO
- **Examples:** 1204
- **License:** MIT License
- **Year:** 2022

### SpartQA
- **Description:** A textual question answering benchmark for spatial reasoning on natural language text.
- **Paper:** SpartQA: : A Textual Question Answering Benchmark for Spatial Reasoning
https://arxiv.org/abs/2104.05832
- **Code:** https://github.com/HLR/SpartQA-baselines
- **Dataset:** https://github.com/HLR/SpartQA_generation
- **Examples:** 510
- **License:** MIT License
- **Year:** 2021

### Natural Questions
- **Description:** User questions issued to Google search, and answers found from Wikipedia by annotators.
- **Paper:** Natural Questions: A Benchmark for Question Answering Research
https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/Natural-Questions-A-Benchmark-for-Question
- **Code:** https://github.com/google-research-datasets/natural-questions
- **Dataset:** https://ai.google.com/research/NaturalQuestions
- **Examples:** 300000
- **License:** Apache-2.0 license
- **Year:** 2019

### ANLI
- **Description:** Large-scale NLI benchmark dataset, collected via an iterative, adversarial human-and-model-in-the-loop procedure.
- **Paper:** Adversarial NLI: A New Benchmark for Natural Language Understanding
https://arxiv.org/abs/1910.14599
- **Code:** https://github.com/facebookresearch/anli
- **Dataset:** https://huggingface.co/datasets/facebook/anli
- **Examples:** 169265
- **License:** CC-BY-NC-4.0
- **Year:** 2019

### BoolQ
- **Description:** Yes/No questions from Google searches, paired with Wikipedia passages.
- **Paper:** BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions
https://arxiv.org/abs/1905.10044
- **Code:** https://github.com/google-research-datasets/boolean-questions
- **Dataset:** https://github.com/google-research-datasets/boolean-questions
- **Examples:** 16000
- **License:** CC-BY-SA-3.0
- **Year:** 2019

### SuperGLUE
- **Description:** Improved and more challenging version of GLUE benchmark.
- **Paper:** SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems
https://arxiv.org/abs/1905.00537
- **Code:** https://github.com/nyu-mll/jiant
- **Dataset:** https://huggingface.co/datasets/aps/super_glue
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2019

### DROP (Discrete Reasoning Over Paragraphs)
- **Description:** Tasks to resolve references in a question and perform discrete operations over them (such as addition, counting, or sorting).
- **Paper:** DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
https://arxiv.org/abs/1903.00161
- **Code:** https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/drop/README.md
- **Dataset:** https://huggingface.co/datasets/ucinlp/drop
- **Examples:** 96000
- **License:** CC-BY-SA-4.0
- **Year:** 2019

### HellaSwag
- **Description:** Predict the most likely ending of a sentence, multiple-choice.
- **Paper:** HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/abs/1905.07830
- **Code:** https://github.com/rowanz/hellaswag/tree/master
- **Dataset:** https://github.com/rowanz/hellaswag/tree/master/data
- **Examples:** 59950
- **License:** MIT License
- **Year:** 2019

### Winogrande
- **Description:** Fill-in-a-blank tasks resolving ambiguities in pronoun references with binary options.
- **Paper:** WinoGrande: An Adversarial Winograd Schema Challenge at Scale
https://arxiv.org/abs/1907.10641
- **Code:** https://github.com/allenai/winogrande
- **Dataset:** https://huggingface.co/datasets/allenai/winogrande
- **Examples:** 44000
- **License:** Apache-2.0 license
- **Year:** 2019

### PIQA (Physical Interaction QA)
- **Description:** Naive physics reasoning tasks focusing on how we interact with everyday objects in everyday situations.
- **Paper:** PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/abs/1911.11641
- **Code:** https://github.com/ybisk/ybisk.github.io/tree/master/piqa
- **Dataset:** https://huggingface.co/datasets/ybisk/piqa
- **Examples:** 18000
- **License:** Academic Free License ("AFL") v. 3.1
- **Year:** 2019

### HotpotQA
- **Description:** A set of Wikipedia-based question-answer
pairs with multi-hop questions.
- **Paper:** HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering
https://arxiv.org/abs/1809.09600
- **Code:** https://github.com/hotpotqa/hotpot
- **Dataset:** https://hotpotqa.github.io/
- **Examples:** 113000
- **License:** CC-BY-SA-4.0
- **Year:** 2018

### GLUE (General Language Understanding Evaluation)
- **Description:** Tool for evaluating and analyzing the performance of models on NLU tasks. Was quickly outperformed by LLMs and replaced by SuperGLUE.
- **Paper:** GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
https://arxiv.org/abs/1804.07461
- **Code:** https://github.com/nyu-mll/GLUE-baselines
- **Dataset:** https://huggingface.co/datasets/nyu-mll/glue
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2018

### OpenBookQA
- **Description:** Question answering dataset, modeled after open book exams.
- **Paper:** Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering
https://arxiv.org/abs/1809.02789
- **Code:** https://github.com/allenai/OpenBookQA
- **Dataset:** https://huggingface.co/datasets/allenai/openbookqa
- **Examples:** 12000
- **License:** Apache-2.0 license
- **Year:** 2018

### SQuAD2.0
- **Description:** Combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones.
- **Paper:** Know What You Don't Know: Unanswerable Questions for SQuAD
https://arxiv.org/abs/1806.03822
- **Code:** https://rajpurkar.github.io/SQuAD-explorer/
- **Dataset:** https://huggingface.co/datasets/bayes-group-diffusion/squad-2.0
- **Examples:** 150000
- **License:** CC-BY-SA-4.0
- **Year:** 2018

### SWAG
- **Description:** Multi-choice tasks of grounded commonsense inference with adversarial filtering.
- **Paper:** SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference
https://arxiv.org/abs/1808.05326

- **Code:** https://github.com/rowanz/swagaf
- **Dataset:** https://github.com/rowanz/swagaf/tree/master/data
- **Examples:** 113000
- **License:** MIT License
- **Year:** 2018

### CommonsenseQA
- **Description:** Multiple-choice question answering dataset that requires commonsense knowledge to predict the correct answers.
- **Paper:** CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge
https://arxiv.org/abs/1811.00937
- **Code:** https://github.com/jonathanherzig/commonsenseqa
- **Dataset:** https://github.com/jonathanherzig/commonsenseqa
- **Examples:** 12102
- **License:** see dataset page
- **Year:** 2018

### RACE (ReAding Comprehension Dataset From Examinations)
- **Description:** Reading comprehension tasks collected from the English exams for middle and high school Chinese students.
- **Paper:** RACE: Large-scale ReAding Comprehension Dataset From Examinations
https://arxiv.org/abs/1704.04683
- **Code:** _No repository provided_
- **Dataset:** https://www.cs.cmu.edu/~glai1/data/race/
- **Examples:** 100000
- **License:** see dataset page
- **Year:** 2017

### SciQ
- **Description:** Multiple choice science exam questions.
- **Paper:** Crowdsourcing Multiple Choice Science Questions
https://arxiv.org/abs/1707.06209
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/allenai/sciq
- **Examples:** 13700
- **License:** CC-BY-SA-3.0
- **Year:** 2017

### TriviaQA
- **Description:** A large-scale question-answering dataset.
- **Paper:** TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
https://arxiv.org/abs/1705.03551
- **Code:** https://github.com/mandarjoshi90/triviaqa
- **Dataset:** https://huggingface.co/datasets/mandarjoshi/trivia_qa
- **Examples:** 650000
- **License:** see dataset page
- **Year:** 2017

### MultiNLI (Multi-Genre Natural Language Inference)
- **Description:** A crowdsourced collection of sentence pairs annotated with textual entailment information.
- **Paper:** A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
https://arxiv.org/abs/1704.05426
- **Code:** https://github.com/nyu-mll/multiNLI
- **Dataset:** https://huggingface.co/datasets/nyu-mll/multi_nli
- **Examples:** 433000
- **License:** see dataset page
- **Year:** 2017

### SQuAD (Stanford Question Answering Dataset)
- **Description:** A reading comprehension dataset consisting of 100,000 questions posed by crowdworkers on a set of Wikipedia articles.
- **Paper:** SQuAD: 100,000+ Questions for Machine Comprehension of Text
https://arxiv.org/abs/1606.05250
- **Code:** https://rajpurkar.github.io/SQuAD-explorer/
- **Dataset:** https://huggingface.co/datasets/rajpurkar/squad
- **Examples:** 100000
- **License:** CC-BY-SA-4.0
- **Year:** 2016

### LAMBADA (LAnguage Modelling Broadened to Account for Discourse Aspects)
- **Description:** A set of passages composed of a
context and a target sentence. The task is to guess the last word of the target sentence. 
- **Paper:** The LAMBADA dataset: Word prediction requiring a broad discourse context
https://arxiv.org/abs/1606.06031
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/cimec/lambada
- **Examples:** 12684
- **License:** CC-BY-SA-4.0
- **Year:** 2016

### MS MARCO
- **Description:** Questions sampled from Bing's search query logs and passages from web documents.
- **Paper:** MS MARCO: A Human Generated MAchine Reading COmprehension Dataset
https://arxiv.org/abs/1611.09268
- **Code:** https://microsoft.github.io/msmarco/
- **Dataset:** https://huggingface.co/datasets/microsoft/ms_marco
- **Examples:** 1112939
- **License:** see dataset page
- **Year:** 2016

### RAFT
- **Description:** A benchmark evaluating the ability of LLMs to solve text classification tasks.
- **Paper:** RAFT: A Real-World Few-Shot Text Classification Benchmark
https://arxiv.org/abs/2109.14076
- **Code:** https://github.com/oughtinc/raft-baselines
- **Dataset:** https://huggingface.co/datasets/ought/raft
- **Examples:** 29000
- **License:** see dataset page
- **Year:** 2021

## 🗂 Language & reasoning,agents & tools use,safety,instruction-following

### BiGGen-Bench
- **Description:** Evaluates nine distinct capabilities of LMs, including instruction following, reasoning, tool usage, and safety.
- **Paper:** The BiGGen Bench: A Principled Benchmark for Fine-grained Evaluation of Language Models with Language Models 
https://arxiv.org/abs/2406.05761
- **Code:** https://github.com/prometheus-eval/prometheus-eval/tree/main/BiGGen-Bench
- **Dataset:** https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
- **Examples:** 765
- **License:** CC-BY-SA-4.0
- **Year:** 2024

## 🗂 Language & reasoning,bias & ethics

### Social Chemistry 101
- **Description:** A conceptual formalism to study people’s everyday social norms and moral judgments.
- **Paper:** Social Chemistry 101: Learning to Reason about Social and Moral Norms
https://arxiv.org/abs/2011.00620
- **Code:** https://github.com/mbforbes/social-chemistry-101
- **Dataset:** https://github.com/mbforbes/social-chemistry-101?tab=readme-ov-file#data
- **Examples:** 4500000
- **License:** CC-BY-SA-4.0
- **Year:** 2020

## 🗂 Language & reasoning,coding,math,instruction-following

### Livebench
- **Description:** A new benchmark designed to be resistant to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. It contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets.
- **Paper:** LiveBench: A Challenging, Contamination-Limited LLM Benchmark 
https://arxiv.org/abs/2406.19314
- **Code:** https://github.com/livebench/livebench
- **Dataset:** https://huggingface.co/collections/livebench/livebench-67eaef9bb68b45b17a197a98
- **Examples:** 1000+
- **License:** see dataset page
- **Year:** 2024

## 🗂 Language & reasoning,information retrieval & rag

### BEIR
- **Description:** BEIR is a heterogeneous benchmark for information retrieval (IR) tasks, contains 15+ IR datasets.
- **Paper:** BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models
https://arxiv.org/abs/2104.08663
- **Code:** https://github.com/beir-cellar/beir
- **Dataset:** https://huggingface.co/BeIR
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2021

### NarrativeQA
- **Description:** A dataset that requires reading entire books or movie scripts to answer the questions. It requires an understanding of the underlying narrative rather than relying on pattern matching or salience. 
- **Paper:** The NarrativeQA Reading Comprehension Challenge 
https://arxiv.org/abs/1712.07040 
- **Code:** https://github.com/google-deepmind/narrativeqa
- **Dataset:** https://huggingface.co/datasets/deepmind/narrativeqa_manual
- **Examples:** 1572
- **License:** Apache-2.0 license
- **Year:** 2017

## 🗂 Language & reasoning,information retrieval & rag,multimodal

### LOFT
- **Description:** A benchmark of real-world tasks requiring context up to millions of tokens designed to evaluate LCLMs' performance on in-context retrieval and reasoning. 
- **Paper:** Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? 
https://arxiv.org/abs/2406.13121
- **Code:** https://github.com/google-deepmind/loft
- **Dataset:** https://github.com/google-deepmind/loft?tab=readme-ov-file#datasets
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

## 🗂 Language & reasoning,instruction-following

### IFEval
- **Description:** A set of prompts with verifiable instructions, such as "write in more than 400 words".
- **Paper:** https://arxiv.org/abs/2311.07911
- **Code:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Dataset:** https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **Examples:** 500
- **License:** Apache-2.0 license
- **Year:** 2023

## 🗂 Language & reasoning,knowledge

### ARB
- **Description:** Advanced reasoning problems in math, physics, biology, chemistry, and law.
- **Paper:** ARB: Advanced Reasoning Benchmark for Large Language Models
https://arxiv.org/abs/2307.13692
- **Code:** https://github.com/TheDuckAI/arb?tab=readme-ov-file
- **Dataset:** https://advanced-reasoning-benchmark.netlify.app/documentation
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

### AGIEval
- **Description:** Uses human-centric standardized exams, such as college entrance exams, law school admission tests, math competitions, and lawyer qualification tests.
- **Paper:** AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models https://arxiv.org/pdf/2304.06364
- **Code:** https://github.com/ruixiangcui/AGIEval
- **Dataset:** see repo
- **Examples:** 8000
- **License:** see dataset page
- **Year:** 2023

## 🗂 Language & reasoning,multilingual

### MultiLoKo
- **Description:** A new benchmark for evaluating multilinguality in LLMs covering 31 languages.
- **Paper:** MultiLoKo: a multilingual local knowledge benchmark for LLMs spanning 31 languages 
https://arxiv.org/abs/2504.10356
- **Code:** https://github.com/facebookresearch/multiloko
- **Dataset:** see repo
- **Examples:** 15500
- **License:** MIT License
- **Year:** 2025

### Include
- **Description:** An evaluation suite to measure the capabilities of multilingual LLMs in a variety of regional contexts across 44 written languages.
- **Paper:** INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge

https://arxiv.org/pdf/2411.19799
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/CohereLabs/include-base-44
- **Examples:** 22953
- **License:** Apache-2.0 license
- **Year:** 2024

### MultiNRC
- **Description:** Assesses LLMs on reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. 
- **Paper:** MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs 

https://arxiv.org/abs/2507.17476
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/ScaleAI/MultiNRC
- **Examples:** 1000
- **License:** see dataset page
- **Year:** 2025

## 🗂 Language & reasoning,multimodal,video

### Video-MME
- **Description:** A benchmark for multimodal long context understanding.
- **Paper:** Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis 
https://arxiv.org/abs/2405.21075 
- **Code:** https://github.com/MME-Benchmarks/Video-MME
- **Dataset:** https://github.com/MME-Benchmarks/Video-MME?tab=readme-ov-file#-dataset
- **Examples:** 900
- **License:** see dataset page
- **Year:** 2024

## 🗂 Language & reasoning,safety

### HELM
- **Description:** Reasoning tasks in several domains (reusing other benchmarks) with a focus on multi-metric evaluation (https://crfm.stanford.edu/helm/).
- **Paper:** https://arxiv.org/abs/2211.09110
- **Code:** https://github.com/stanford-crfm/helm
- **Dataset:** see repository 
- **Examples:** unspecified
- **License:** Apache-2.0 license
- **Year:** 2022

## 🗂 Llm judge evaluation

### JudgeBench
- **Description:** A benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding. Evaluated on a collection of prompted judges, fine-tuned judges, multi-agent judges, and reward models.
- **Paper:** JudgeBench: A Benchmark for Evaluating LLM-based Judges 
https://arxiv.org/abs/2410.12784 
- **Code:** https://github.com/ScalerLab/JudgeBench?tab=readme-ov-file
- **Dataset:** https://huggingface.co/datasets/ScalerLab/JudgeBench
- **Examples:** 620
- **License:** MIT License
- **Year:** 2024

## 🗂 Llm-generated text detection

### DetectRL
- **Description:** Human-written datasets from domains where LLMs are particularly prone to misuse.
- **Paper:** DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios 
https://arxiv.org/abs/2410.23746
- **Code:** https://github.com/NLP2CT/DetectRL
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

## 🗂 Math

### AIME
- **Description:** This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2024.
- **Paper:** nan
- **Code:** https://artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination
- **Dataset:** https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
- **Examples:** 30
- **License:** MIT License
- **Year:** 2024

### CHAMP
- **Description:** High-school math problems, annotated with general math facts and problem-specific hints. These annotations allow exploring the effects of additional information, such as relevant hints, misleading concepts, or related problems. 
- **Paper:** CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs' Mathematical Reasoning Capabilities 
https://arxiv.org/abs/2401.06961
- **Code:** https://github.com/YilunZhou/champ-dataset
- **Dataset:** https://yujunmao1.github.io/CHAMP/explorer.html
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

### TemplateGSM
- **Description:** A dataset comprising over 7 million synthetically generated grade school math problems, each accompanied by code-based and natural language solutions.
- **Paper:** Training and Evaluating Language Models with Template-based Data Generation 
https://arxiv.org/abs/2411.18104
- **Code:** https://github.com/iiis-ai/TemplateMath
- **Dataset:** https://huggingface.co/datasets/math-ai/TemplateGSM
- **Examples:** 7000000
- **License:** CC-BY-4.0
- **Year:** 2024

### HARD-Math
- **Description:** Human-Annotated Reasoning Dataset for Math. Consists of short answer problems, based on the AHSME, AMC, & AIME contests.
- **Paper:** HARDMath: A Benchmark Dataset for Challenging Problems in Applied Mathematics 
https://arxiv.org/abs/2410.09988
- **Code:** https://github.com/sarahmart/HARDMath
- **Dataset:** https://github.com/sarahmart/HARDMath/tree/main/data
- **Examples:** 1400
- **License:** see dataset page
- **Year:** 2024

### TheoremQA	
- **Description:** Theorem-driven QA dataset that evaluates LLMs capabilities to apply theorems to solve science problems. Contains 800 questions covering 350 theorems from math, physics, EE&CS, and finance.
- **Paper:** TheoremQA: A Theorem-driven Question Answering dataset 
https://arxiv.org/abs/2305.12524
- **Code:** https://github.com/TIGER-AI-Lab/TheoremQA
- **Dataset:** https://huggingface.co/datasets/TIGER-Lab/TheoremQA
- **Examples:** 800
- **License:** MIT License
- **Year:** 2023

### MGSM (Multilingual Grade School Math)
- **Description:** Grade-school math problems from the GSM8K dataset, translated into 10 languages.
- **Paper:** Language Models are Multilingual Chain-of-Thought Reasoners
https://arxiv.org/abs/2210.03057
- **Code:** https://github.com/google-research/url-nlp
- **Dataset:** https://huggingface.co/datasets/juletxara/mgsm
- **Examples:** 2500
- **License:** CC-BY-SA-4.0
- **Year:** 2022

### GSMHard
- **Description:** The harder version of the GSM8K math reasoning dataset. Numbers in the questions of GSM8K are replaced with larger numbers that are less common.
- **Paper:** PAL: Program-aided Language Models 
https://arxiv.org/abs/2211.10435
- **Code:** https://github.com/reasoning-machines/pal
- **Dataset:** https://huggingface.co/datasets/reasoning-machines/gsm-hard
- **Examples:** 1319
- **License:** MIT License
- **Year:** 2022

### SVAMP
- **Description:** Grade-school-level math word problems that require models to perform single-variable arithmetic operations. Created by applying variations over examples sampled from existing datasets. 
- **Paper:** Are NLP Models really able to Solve Simple Math Word Problems?
https://arxiv.org/abs/2103.07191
- **Code:** https://github.com/arkilpatel/SVAMP
- **Dataset:** https://github.com/arkilpatel/SVAMP/tree/main/data
- **Examples:** 1000
- **License:** MIT License
- **Year:** 2021

### MATH
- **Description:** Tasks from US mathematics competitions that cover algebra, calculus, geometry, and statistics.
- **Paper:** Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/abs/2103.03874
- **Code:** https://github.com/hendrycks/math/?tab=readme-ov-file
- **Dataset:** https://github.com/hendrycks/math/?tab=readme-ov-file
- **Examples:** 12500
- **License:** MIT License
- **Year:** 2021

### GSM8K
- **Description:** Grade school math word problems.
- **Paper:** Training Verifiers to Solve Math Word Problems
https://arxiv.org/abs/2110.14168
- **Code:** https://github.com/openai/grade-school-math
- **Dataset:** https://github.com/openai/grade-school-math/tree/master/grade_school_math/data
- **Examples:** 8500
- **License:** MIT License
- **Year:** 2021

### AQUA-RAT
- **Description:** An algebraic word problem dataset, with multiple choice questions annotated with rationales.
- **Paper:** Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems 
https://arxiv.org/abs/1705.04146
- **Code:** https://github.com/google-deepmind/AQuA
- **Dataset:** https://huggingface.co/datasets/deepmind/aqua_rat
- **Examples:** 100000
- **License:** Apache-2.0 license
- **Year:** 2017

### We-Math
- **Description:** A benchmark that evaluates the problem-solving principles in knowledge acquisition and generalization for math tasks.
- **Paper:** We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning? 
https://arxiv.org/abs/2407.01284
- **Code:** https://github.com/We-Math/We-Math
- **Dataset:** https://huggingface.co/datasets/We-Math/We-Math
- **Examples:** 1740
- **License:** CC-BY-NC-4.0
- **Year:** 2024

### MathArena
- **Description:** A benchmark for evaluating LLMs on newly-released math competition problems.
- **Paper:** MathArena: Evaluating LLMs on Uncontaminated Math Competitions
https://arxiv.org/abs/2505.23281 
- **Code:** https://github.com/eth-sri/matharena
- **Dataset:** https://github.com/eth-sri/matharena
- **Examples:** 149
- **License:** see dataset page
- **Year:** 2025

## 🗂 Math,domain-specific

### TeleMath
- **Description:** Designed to evaluate LLM performance in solving mathematical problems with numerical solutions in the telecommunications domain.
- **Paper:** TeleMath: A Benchmark for Large Language Models in Telecom Mathematical Problem Solving

https://arxiv.org/abs/2506.10674
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/netop/TeleMath
- **Examples:** 500
- **License:** MIT License
- **Year:** 2025

## 🗂 Math,language & reasoning

### DocMath-Eval
- **Description:** Designed to evaluate the numerical reasoning capabilities of LLMs in the context of understanding and analyzing specialized documents containing both text and tables. 
- **Paper:** DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents 
https://arxiv.org/abs/2311.09805
- **Code:** https://github.com/yale-nlp/DocMath-Eval
- **Dataset:** https://huggingface.co/datasets/yale-nlp/DocMath-Eval
- **Examples:** 4000
- **License:** MIT License
- **Year:** 2023

### FinQA
- **Description:** Large-scale dataset with Question-Answering pairs over Financial reports, written by financial experts.
- **Paper:** FinQA: A Dataset of Numerical Reasoning over Financial Data 
https://arxiv.org/abs/2109.00122 
- **Code:** https://github.com/czyssrs/FinQA
- **Dataset:** https://huggingface.co/datasets/ibm-research/finqa
- **Examples:** 8000
- **License:** CC-BY-4.0
- **Year:** 2021

## 🗂 Multimodal

### SEED-Bench
- **Description:** A benchmark for evaluating Multimodal LLMs using multiple-choice questions.
- **Paper:** SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension
https://arxiv.org/abs/2307.16125
- **Code:** https://github.com/AILab-CVC/SEED-Bench
- **Dataset:** https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2
- **Examples:** 24000
- **License:** CC-BY-NC-4.0
- **Year:** 2023

### Q-bench
- **Description:** Evaluates MLLMs on three dimensions: low-level visual perception, low-level visual description, and overall visual quality assessment.
- **Paper:** Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision
https://arxiv.org/abs/2309.14181
- **Code:** https://github.com/Q-Future/Q-Bench
- **Dataset:** https://huggingface.co/datasets/q-future/Q-Bench-HF
- **Examples:** 2990
- **License:** S-Lab License 1.0
- **Year:** 2023

### M3Exam
- **Description:** A set of human exam questions in 9 diverse languages with three educational levels, where about 23% of the questions require processing images for successful solving.
- **Paper:** M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models
https://arxiv.org/abs/2306.05179
- **Code:** https://github.com/DAMO-NLP-SG/M3Exam
- **Dataset:** https://github.com/DAMO-NLP-SG/M3Exam?tab=readme-ov-file#data
- **Examples:** 12317
- **License:** see dataset page
- **Year:** 2023

### A Multitask, Multilingual, Multimodal Evaluation of ChatGPT
- **Description:** A framework for quantitatively evaluating interactive LLMs such as ChatGPT using 23 data sets covering 8 common NLP tasks.
- **Paper:** A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity
https://arxiv.org/abs/2302.04023
- **Code:** https://github.com/HLTCHKUST/chatgpt-evaluation
- **Dataset:** https://github.com/HLTCHKUST/chatgpt-evaluation/tree/main/src
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

### MME
- **Description:** Measures both perception and cognition abilities on a total of 14 subtasks.
- **Paper:** MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models
https://arxiv.org/abs/2306.13394
- **Code:** https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation
- **Dataset:** https://huggingface.co/datasets/lmms-lab/MME
- **Examples:** 2374
- **License:** see dataset page
- **Year:** 2023

### LVLM-eHub
- **Description:** Multi-Modality Arena helps benchmark vision-language models side-by-side while providing images as inputs. 
- **Paper:** LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models 
https://arxiv.org/abs/2306.09265
- **Code:** https://github.com/OpenGVLab/Multi-Modality-Arena
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

### MMBench
- **Description:** A bilingual benchmark for assessing the multi-modal capabilities of vision-language models. Contains 2974 multiple-choice questions, covering 20 ability dimensions.
- **Paper:** MMBench: Is Your Multi-modal Model an All-around Player? 
https://arxiv.org/abs/2307.06281
- **Code:** https://github.com/open-compass/MMBench
- **Dataset:** see repo
- **Examples:** 2974
- **License:** see dataset page
- **Year:** 2023

### HQHBench
- **Description:** Evaluates the performance of LVLMs across different types of hallucination. It consists of 4000 free-form VQA image-instruction pairs, with 500 pairs for each hallucination type.
- **Paper:** Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations
https://arxiv.org/pdf/1602.07332v1
- **Code:** https://github.com/HQHBench/HQHBench
- **Dataset:** see repo
- **Examples:** 4000
- **License:** see dataset page
- **Year:** 2016

### MM-Vet
- **Description:** An evaluation benchmark that examines LMMs on complicated multimodal tasks. 
- **Paper:** MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities
 
https://arxiv.org/abs/2308.02490
- **Code:** https://github.com/yuweihao/MM-Vet
- **Dataset:** https://huggingface.co/datasets/whyu/mm-vet
- **Examples:** 218
- **License:** CC-BY-NC-4.0
- **Year:** 2023

## 🗂 Multimodal,information retrieval & rag

### MMNeedle
- **Description:** MultiModal Needle-in-a-haystack (MMNeedle) benchmark is designed to assess the long-context capabilities of MLLMs.
- **Paper:** Multimodal Needle in a Haystack: Benchmarking Long-Context Capability of Multimodal Large Language Models 
https://arxiv.org/abs/2406.11230
- **Code:** https://github.com/Wang-ML-Lab/multimodal-needle-in-a-haystack
- **Dataset:** https://drive.google.com/drive/folders/1D2XHmj466e7WA4aY7zLkbdTmp3it2ZPy
- **Examples:** 880000
- **License:** see dataset page
- **Year:** 2024

## 🗂 Multimodal,language & reasoning

### MMMU
- **Description:** Evaluates multimodal models on massive multi-discipline tasks demanding college-level subject knowledge. Includes 11.5K questions from college exams, quizzes, and textbooks (https://mmmu-benchmark.github.io/).
- **Paper:** MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
https://arxiv.org/abs/2311.16502
- **Code:** https://github.com/MMMU-Benchmark/MMMU
- **Dataset:** https://huggingface.co/datasets/MMMU/MMMU
- **Examples:** 11500
- **License:** Apache-2.0 license
- **Year:** 2023

### WebQA
- **Description:** Visual Question Answering (VQA) benchmark that evaluates models' language groundable visual representations for novel objects and the ability to reason.
- **Paper:** WebQA: Multihop and Multimodal QA
https://arxiv.org/abs/2109.00590
- **Code:** https://github.com/WebQnA/WebQA
- **Dataset:** https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ
- **Examples:** 41732
- **License:** see dataset page
- **Year:** 2021

## 🗂 Safety

### FACTS Grounding
- **Description:** A measure of how accurately LLMs ground their responses in provided source material and avoid hallucinations.
- **Paper:** FACTS Grounding: A new benchmark for evaluating the factuality of large language models
Published 
https://arxiv.org/abs/2501.03200
- **Code:** https://www.kaggle.com/code/andrewmingwang/facts-grounding-benchmark-starter-code
- **Dataset:** https://www.kaggle.com/datasets/deepmind/facts-grounding-examples
- **Examples:** 1719
- **License:** CC-BY-4.0
- **Year:** 2025

### HarmBench
- **Description:** Adversarial behaviors including cybercrime, copyright violations, and generating misinformation (https://www.harmbench.org). 
- **Paper:** https://arxiv.org/abs/2402.04249
- **Code:** https://github.com/centerforaisafety/HarmBench/tree/main
- **Dataset:** https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets
- **Examples:** 510
- **License:** MIT License
- **Year:** 2024

### SimpleQA
- **Description:** Measures the ability for language models to answer short, fact-seeking questions to reduce hallucinations.
- **Paper:** Measuring short-form factuality in large language models
https://arxiv.org/abs/2411.04368
- **Code:** https://github.com/openai/simple-evals
- **Dataset:** https://huggingface.co/datasets/basicv8vc/SimpleQA
- **Examples:** 4326
- **License:** MIT License
- **Year:** 2024

### AgentHarm
- **Description:** Explicitly malicious agent tasks, including fraud, cybercrime, and harassment.
- **Paper:** AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents
https://arxiv.org/abs/2410.09024
- **Code:** https://github.com/UKGovernmentBEIS/inspect_evals
- **Dataset:** https://huggingface.co/datasets/ai-safety-institute/AgentHarm
- **Examples:** 110
- **License:** MIT License
- **Year:** 2024

### StrongReject
- **Description:** Tests a model’s resistance against common attacks from the literature.
- **Paper:** A StrongREJECT for Empty Jailbreaks
https://arxiv.org/abs/2402.10260
- **Code:** https://github.com/dsbowen/strong_reject
- **Dataset:** https://github.com/dsbowen/strong_reject/tree/main/docs/api
- **Examples:** nan
- **License:** MIT License
- **Year:** 2024

### AIR-Bench
- **Description:** AI safety benchmark aligned with emerging regulations. Considers operational, content safety, legal and societal risks (https://crfm.stanford.edu/helm/air-bench/latest/).
- **Paper:** https://arxiv.org/abs/2407.17436
- **Code:** https://github.com/stanford-crfm/air-bench-2024
- **Dataset:** https://huggingface.co/datasets/stanford-crfm/air-bench-2024
- **Examples:** 5,694
- **License:** Apache-2.0 license
- **Year:** 2024

### OR-Bench
- **Description:** 80,000 benign prompts likely rejected by LLMs across 10 common rejection categories.
- **Paper:** An Over-Refusal Benchmark for Large
Language Models
https://arxiv.org/abs/2405.20947 
- **Code:** https://github.com/justincui03/or-bench
- **Dataset:** https://huggingface.co/datasets/bench-llm/or-bench
- **Examples:** 80000
- **License:** CC-BY-4.0
- **Year:** 2024

### TOFUEVAL
- **Description:** Evaluation benchmark on topic-focused dialogue summarization. Contains binary sentence-level human annotations of the factual consistency of these summaries along with detailed explanations of factually inconsistent sentences.
- **Paper:** TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue Summarization
https://arxiv.org/abs/2402.13249
- **Code:** https://github.com/amazon-science/tofueval
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2024

### BackdoorLLM
- **Description:** A benchmark for backdoor attacks in text generation.
- **Paper:** BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models 
https://arxiv.org/abs/2408.12798
- **Code:** https://github.com/bboylyg/BackdoorLLM
- **Dataset:** https://huggingface.co/datasets/BackdoorLLM/Backdoored_Dataset
- **Examples:** 4200
- **License:** see dataset page
- **Year:** 2024

### Hal-eval
- **Description:** Assesses LVLMs' ability to tackle a broad spectrum of hallucinations.
- **Paper:** Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models 
https://arxiv.org/abs/2402.15721
- **Code:** https://github.com/WisdomShell/hal-eval
- **Dataset:**  https://github.com/WisdomShell/hal-eval/tree/main/evaluation_dataset
- **Examples:** 2000000
- **License:** see dataset page
- **Year:** 2024

### LLM-AggreFact
- **Description:** Fact verification benchmark that aggregates 11 publicly available datasets on factual consistency evaluation across both closed-book and grounded generation settings.
- **Paper:** MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents 
https://arxiv.org/abs/2404.10774 
- **Code:** https://github.com/Liyan06/MiniCheck
- **Dataset:** https://huggingface.co/datasets/lytang/LLM-AggreFact
- **Examples:** 59740
- **License:** CC-BY-ND-4.0
- **Year:** 2024

### ForbiddenQuestions
- **Description:** A set of questions targetting 13 behavior scenarios disallowed by OpenAI.
- **Paper:** "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models
https://arxiv.org/abs/2308.03825
- **Code:** https://github.com/verazuo/jailbreak_llms
- **Dataset:** https://github.com/verazuo/jailbreak_llms
- **Examples:** 15140
- **License:** MIT License
- **Year:** 2023

### MaliciousInstruct
- **Description:** Covers ten 'malicious intentions', including psychological manipulation, theft, cyberbullying, and fraud.
- **Paper:** Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation
https://arxiv.org/abs/2310.06987
- **Code:** https://github.com/Princeton-SysML/Jailbreak_LLM/tree/main
- **Dataset:** https://github.com/Princeton-SysML/Jailbreak_LLM/tree/main/data
- **Examples:** 100
- **License:** see dataset page
- **Year:** 2023

### SycophancyEval
- **Description:** Tests if human feedback encourages model responses to match user beliefs over truthful ones, a behavior known as sycophancy.
- **Paper:** Towards Understanding Sycophancy in Language Models
https://arxiv.org/abs/2310.13548
- **Code:** https://github.com/meg-tong/sycophancy-eval
- **Dataset:** https://huggingface.co/datasets/meg-tong/sycophancy-eval
- **Examples:** nan
- **License:** MIT License
- **Year:** 2023

### DecodingTrust
- **Description:** Evaluate trustworthiness of LLMs across 8 perspectives: toxicity, stereotypes, adversarial and robustness, privacy, ethics and fairness.
- **Paper:** DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models
https://arxiv.org/abs/2306.11698
- **Code:** https://github.com/AI-secure/DecodingTrust
- **Dataset:** https://huggingface.co/datasets/AI-Secure/DecodingTrust
- **Examples:** 243,877
- **License:** CC-BY-SA-4.0
- **Year:** 2023

### AdvBench
- **Description:** A set of 500 harmful strings that the model should not reproduce and 500 harmful instructions.
- **Paper:** Universal and Transferable Adversarial Attacks on Aligned Language Models
https://arxiv.org/abs/2307.15043
- **Code:** https://github.com/llm-attacks/llm-attacks
- **Dataset:** https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench
- **Examples:** 1000
- **License:** MIT License
- **Year:** 2023

### XSTest
- **Description:** A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.
- **Paper:** XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models
https://arxiv.org/abs/2308.01263
- **Code:** https://github.com/paul-rottger/exaggerated-safety
- **Dataset:** https://github.com/paul-rottger/exaggerated-safety/blob/main/xstest_v2_prompts.csv
- **Examples:** 450
- **License:** CC-BY-4.0
- **Year:** 2023

### OpinionQA
- **Description:** A dataset for evaluating the alignment of LM opinions with those of 60 US demographic groups.
- **Paper:** Whose Opinions Do Language Models Reflect?
https://arxiv.org/abs/2303.17548
- **Code:** https://github.com/tatsu-lab/opinions_qa
- **Dataset:** https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69
- **Examples:** 1498
- **License:** see dataset page
- **Year:** 2023

### SafetyBench
- **Description:** Multiple-choice questions concerning offensive content, bias, illegal activities, and mental health.
- **Paper:** SafetyBench: Evaluating the Safety of Large Language Models
https://arxiv.org/abs/2309.07045
- **Code:** https://github.com/thu-coai/SafetyBench
- **Dataset:** https://huggingface.co/datasets/thu-coai/SafetyBench
- **Examples:** 11435
- **License:** MIT License
- **Year:** 2023

### HarmfulQA
- **Description:** Harmful questions covering 10 topics and ~10 subtopics each.
- **Paper:** Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment
https://arxiv.org/abs/2308.09662
- **Code:** https://github.com/declare-lab/red-instruct
- **Dataset:** https://huggingface.co/datasets/declare-lab/HarmfulQA
- **Examples:** 1960
- **License:** Apache-2.0 license
- **Year:** 2023

### QHarm
- **Description:** Dataset consists of human-written entries sampled randomly from AnthropicHarmlessBase.
- **Paper:** Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions
https://arxiv.org/abs/2309.07875
- **Code:** https://github.com/vinid/safety-tuned-llamas
- **Dataset:** https://github.com/vinid/safety-tuned-llamas
- **Examples:** 100
- **License:** CC-BY-SA-4.0
- **Year:** 2023

### BeaverTails
- **Description:** A set of prompts sampled from AnthropicRedTeam that cover 14 harm categories.
- **Paper:** BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset
https://arxiv.org/abs/2307.04657
- **Code:** https://github.com/PKU-Alignment/beavertails
- **Dataset:** https://huggingface.co/datasets/PKU-Alignment/BeaverTails
- **Examples:** 334000
- **License:** CC-BY-SA-4.0
- **Year:** 2023

### DoNotAnswer
- **Description:** The dataset consists of prompts across 12 harm types to which responsible LLMs do not answer. 
- **Paper:** Do-Not-Answer: Evaluating Safeguards in LLMs
https://arxiv.org/abs/2308.13387
- **Code:** https://github.com/Libr-AI/do-not-answer
- **Dataset:** https://huggingface.co/datasets/LibrAI/do-not-answer
- **Examples:** 939
- **License:** Apache-2.0 license
- **Year:** 2023

### ExpertQA
- **Description:** Long-form QA dataset with 2177 questions spanning 32 fields for evaluating attribution and factuality of LLM outputs in domain-specific scenarios.
- **Paper:** ExpertQA: Expert-Curated Questions and Attributed Answers 
https://arxiv.org/abs/2309.07852
- **Code:** https://github.com/chaitanyamalaviya/ExpertQA
- **Dataset:** see repo
- **Examples:** 2177
- **License:** MIT License
- **Year:** 2023

### HaluEval
- **Description:** A collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination.
- **Paper:** HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models 
https://arxiv.org/abs/2305.11747
- **Code:** https://github.com/RUCAIBox/HaluEval
- **Dataset:** https://github.com/RUCAIBox/HaluEval?tab=readme-ov-file#data-release
- **Examples:** 35000
- **License:** Apache-2.0 license
- **Year:** 2023

### RED-EVAL
- **Description:** Safety evaluation benchmark that carries out red-teaming.
- **Paper:** Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment
https://arxiv.org/abs/2308.09662
- **Code:** https://github.com/declare-lab/red-instruct
- **Dataset:** see repo
- **Examples:** 1960
- **License:** see dataset page
- **Year:** 2023

### ToxiGen
- **Description:** A set of toxic and benign statements about minority groups.
- **Paper:** ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection
https://arxiv.org/abs/2203.09509
- **Code:** https://github.com/microsoft/TOXIGEN/tree/main
- **Dataset:** https://huggingface.co/datasets/toxigen/toxigen-data
- **Examples:** 274000
- **License:** MIT License
- **Year:** 2022

### HHH (Helpfulness, Honesty, Harmlessness)
- **Description:** Human preference data about helpfulness and harmlessness.
- **Paper:** Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
https://arxiv.org/abs/2204.05862
- **Code:** https://github.com/anthropics/hh-rlhf
- **Dataset:** https://github.com/anthropics/hh-rlhf
- **Examples:** 44849
- **License:** MIT License
- **Year:** 2022

### PersonalInfoLeak
- **Description:** Evaluates whether LLMs are prone to leaking PII, contains name-email pairs.
- **Paper:** Are Large Pre-Trained Language Models Leaking Your Personal Information?
https://arxiv.org/abs/2205.12628
- **Code:** https://github.com/jeffhj/LM_PersonalInfoLeak
- **Dataset:** https://github.com/jeffhj/LM_PersonalInfoLeak/tree/main/data
- **Examples:** 3238
- **License:** Apache-2.0 license
- **Year:** 2022

### AnthropicRedTeam
- **Description:** Human-generated and annotated red teaming dialogues.
- **Paper:** Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned
https://arxiv.org/abs/2209.07858
- **Code:** https://github.com/anthropics/hh-rlhf
- **Dataset:** https://huggingface.co/datasets/Anthropic/hh-rlhf
- **Examples:** 38961
- **License:** MIT License
- **Year:** 2022

### TURINGBENCH
- **Description:** The TuringBench Dataset will assist researchers in building models that can effectively distinguish machine-generated texts from human-written texts.
- **Paper:** TURINGBENCH: A Benchmark Environment for Turing Test in the Age of Neural Text Generation 
https://arxiv.org/abs/2109.13296
- **Code:** https://github.com/AdaUchendu/TuringBench
- **Dataset:** https://turingbench.ist.psu.edu/
- **Examples:** 200000
- **License:** see dataset page
- **Year:** 2021

### HHH alignment
- **Description:** Evaluates language models on alignment, broken down into the categories of helpfulness, honesty/accuracy, harmlessness, and other.
- **Paper:** A General Language Assistant as a Laboratory for Alignment 
https://arxiv.org/abs/2112.00861
- **Code:** _No repository provided_
- **Dataset:** https://huggingface.co/datasets/HuggingFaceH4/hhh_alignment
- **Examples:** 221
- **License:** Apache-2.0 license
- **Year:** 2021

### RealToxicityPrompt
- **Description:** A dataset of 100K naturally occurring, sentence-level prompts derived from a large corpus of English web text, paired with toxicity scores from a widely-used toxicity classifier.
- **Paper:** RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models
https://arxiv.org/abs/2009.11462
- **Code:** https://github.com/allenai/real-toxicity-prompts?tab=readme-ov-file
- **Dataset:** https://huggingface.co/datasets/allenai/real-toxicity-prompts
- **Examples:** 99442
- **License:** Apache-2.0 license
- **Year:** 2020

### RobustBench
- **Description:** Adversarial robustness benchmark.
- **Paper:** RobustBench: a standardized adversarial robustness benchmark 
https://arxiv.org/abs/2010.09670
- **Code:** https://github.com/RobustBench/robustbench
- **Dataset:** see repo
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2020

### CYBERSECEVAL 2
- **Description:** A novel benchmark to quantify LLM security risks, including prompt injection and code interpreter abuse.
- **Paper:** CyberSecEval 2: A Wide-Ranging Cybersecurity Evaluation Suite for Large Language Models 
https://arxiv.org/abs/2404.13161
- **Code:** https://github.com/meta-llama/PurpleLlama
- **Dataset:** https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks/benchmark
- **Examples:** nan
- **License:** MIT License
- **Year:** 2024

### ConfAIde
- **Description:** A benchmark designed to identify critical weaknesses in the privacy reasoning capabilities of instruction-tuned LLMs.
- **Paper:** Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory 
https://arxiv.org/abs/2310.17884
- **Code:** https://github.com/skywalker023/confAIde
- **Dataset:** https://github.com/skywalker023/confAIde/tree/main/benchmark
- **Examples:** nan
- **License:** see dataset page
- **Year:** 2023

## 🗂 Safety,agents & tools use

### ToolEmu
- **Description:** A framework that uses an LM to emulate tool execution and enables the testing of LM agents against a diverse range of tools and scenarios, without manual instantiation.
- **Paper:** Identifying the Risks of LM Agents with an LM-Emulated Sandbox 
https://arxiv.org/abs/2309.15817
- **Code:** https://github.com/ryoungj/ToolEmu
- **Dataset:** https://github.com/ryoungj/ToolEmu/blob/main/assets/all_cases.json
- **Examples:** 144
- **License:** see dataset page
- **Year:** 2023

## 🗂 Safety,bias & ethics

### TrustLLM
- **Description:** A benchmark across six dimensions including truthfulness, safety, fairness, robustness, privacy, and machine ethics. Consists of over 30 datasets.
- **Paper:** TrustLLM: Trustworthiness in Large Language Models
https://arxiv.org/abs/2401.05561
- **Code:** https://github.com/HowieHwong/TrustLLM
- **Dataset:** https://github.com/HowieHwong/TrustLLM?tab=readme-ov-file#dataset-download
- **Examples:** nan
- **License:** MIT License
- **Year:** 2024

### TRUSTGPT
- **Description:** Evaluates large language models on toxicity, bias, and value-alignment to ensure ethical and moral compliance.
- **Paper:** TRUSTGPT: A Benchmark for Trustworthy and
Responsible Large Language Models
https://arxiv.org/pdf/2306.11507
- **Code:** https://github.com/HowieHwong/TrustGPT
- **Dataset:** https://github.com/mbforbes/social-chemistry-101
- **Examples:** 292000
- **License:** CC-BY-SA-4.0
- **Year:** 2023

### BOLD
- **Description:** A set of unfinished sentences from Wikipedia designed to assess bias in text generation.
- **Paper:** BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation
https://arxiv.org/abs/2101.11718
- **Code:** https://github.com/amazon-science/bold
- **Dataset:** https://github.com/amazon-science/bold/tree/main/prompts
- **Examples:** 23679
- **License:** CC-BY-SA-4.0
- **Year:** 2021

### BBQ
- **Description:** Evaluate social biases of LLMs in question answering.
- **Paper:** BBQ: A Hand-Built Bias Benchmark for Question Answering
https://arxiv.org/abs/2110.08193
- **Code:** https://github.com/nyu-mll/BBQ
- **Dataset:** https://github.com/nyu-mll/BBQ/tree/main/data
- **Examples:** 58492
- **License:** CC-BY-SA-4.0
- **Year:** 2021

### StereoSet
- **Description:** A large-scale natural dataset in English to measure stereotypical biases in four domains: gender, profession, race, and religion.
- **Paper:** StereoSet: Measuring stereotypical bias in pretrained language models
https://arxiv.org/abs/2004.09456
- **Code:** https://github.com/moinnadeem/StereoSet
- **Dataset:** https://huggingface.co/datasets/McGill-NLP/stereoset
- **Examples:** 4229
- **License:** CC-BY-SA-4.0
- **Year:** 2020

### ETHICS
- **Description:** A set of binary-choice questions on ethics with two actions to choose from.
- **Paper:** Aligning AI With Shared Human Values
https://arxiv.org/abs/2008.02275
- **Code:** https://github.com/hendrycks/ethics
- **Dataset:** https://huggingface.co/datasets/hendrycks/ethics
- **Examples:** 134400
- **License:** MIT License
- **Year:** 2020

### CrowS-Pairs (Crowdsourced Stereotype Pairs)
- **Description:** Covers stereotypes dealing with nine types of bias, like race, religion, and age.
- **Paper:** CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models
https://arxiv.org/abs/2010.00133
- **Code:** https://github.com/nyu-mll/crows-pairs
- **Dataset:** https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv
- **Examples:** 1508
- **License:** CC-BY-SA-4.0
- **Year:** 2020

### SEAT (Sentence Encoder Association Test)
- **Description:** Measures bias in sentence encoders.
- **Paper:** On Measuring Social Biases in Sentence Encoders
https://arxiv.org/abs/1903.10561
- **Code:** https://github.com/W4ngatang/sent-bias
- **Dataset:** https://github.com/W4ngatang/sent-bias/tree/master/tests
- **Examples:** nan
- **License:** CC-BY-NC-4.0
- **Year:** 2019

### WinoGender
- **Description:** Pairs of sentences that differ only by the gender of one pronoun in the sentence, designed to test for the presence of gender bias in automated coreference resolution systems.
- **Paper:** Gender Bias in Coreference Resolution
https://arxiv.org/abs/1804.09301
- **Code:** https://github.com/rudinger/winogender-schemas
- **Dataset:** https://huggingface.co/datasets/oskarvanderwal/winogender
- **Examples:** 720
- **License:** MIT License
- **Year:** 2018

## 🗂 Safety,domain-specific

### HealthBench
- **Description:** Realistic healthcare scenarios: emergency referrals, global health, health data tasks, context-seeking, expertise-tailored communication, response depth, and responding under uncertainty.
- **Paper:** HealthBench: Evaluating Large Language Models
Towards Improved Human Health

https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf
- **Code:** https://github.com/openai/simple-evals
- **Dataset:** https://github.com/openai/simple-evals
- **Examples:** 5000
- **License:** MIT License
- **Year:** 2025

## 🗂 Summarization

### TLDR 9+
- **Description:** A large-scale summarization dataset that contains over 9 million training instances extracted from Reddit discussion forum.
- **Paper:** TLDR9+: A Large Scale Resource for Extreme Summarization of Social Media Posts 
https://arxiv.org/abs/2110.01159
- **Code:** https://github.com/sajastu/reddit_collector
- **Dataset:** https://github.com/sajastu/reddit_collector?tab=readme-ov-file#dataset-links
- **Examples:** 9M+
- **License:** see dataset page
- **Year:** 2021

## 🗂 Summarization,language & reasoning

### MLSUM
- **Description:** Multilingual summarization dataset crawled from different news websites.
- **Paper:** MLSUM: The Multilingual Summarization Corpus
https://arxiv.org/abs/2004.14900
- **Code:** https://github.com/ThomasScialom/MLSUM
- **Dataset:** https://huggingface.co/datasets/GEM/mlsum
- **Examples:** 535062
- **License:** see dataset page
- **Year:** 2020

## 🗂 Video,multimodal

### EvalCrafter
- **Description:** A framework and pipeline for evaluating the performance of the generated videos, such as visual qualities, content qualities, motion qualities, and text-video alignment.
- **Paper:** EvalCrafter: Benchmarking and Evaluating Large Video Generation Models
https://arxiv.org/abs/2310.11440
- **Code:** https://github.com/EvalCrafter/EvalCrafter
- **Dataset:** https://huggingface.co/datasets/RaphaelLiu/EvalCrafter_T2V_Dataset
- **Examples:** 700
- **License:** Apache-2.0 license
- **Year:** 2023

---
## 📜 License
This list is under the **MIT License**.  
Each dataset or benchmark has its **own license** — please check before use.
