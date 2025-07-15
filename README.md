# wikiBOT: Wikipedia-Based Question Answering App

**wikiBOT** is an end-to-end question answering system that:
- Collects and samples Wikipedia content
- Trains a machine learning model to answer questions
- Uses a retrieval + reader model for inference
- Provides a CLI or web-based app for interaction

---

## 🚀 Features
- Automated Wikipedia content collection and sampling
- Preprocessing pipeline for clean training data
- Generative QA model training (e.g., using T5 or similar)
- CLI-based interface (web version optional/coming soon)

## 🧠 How to Train

1. **Crawl Wikipedia Topics**  
   ```bash
   python -m ml.scripts.wikipedia_crawler
Creates a text file with a list of fetched Wikipedia topics.
(You can write your own script to fetch topics smartly)

2. **Sample Wikipedia Content**  
   ```bash
   python -m ml.scripts.data_sampler
Creates a text file with a list of fetched Wikipedia topics.

3. **Preprocess Dataset**  
   ```bash
   python -m ml.scripts.preprocess_data
Creates a text file with a list of fetched Wikipedia topics.

4. **Train the Model**  
   ```bash
   python -m ml.scripts.train_gen_model
Creates a text file with a list of fetched Wikipedia topics.



## How to Run

 ```bash
python -m wikiBOT_cli.py