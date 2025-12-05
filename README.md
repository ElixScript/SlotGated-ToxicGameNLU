# **SlotGated-ToxicGameNLU**

**Joint Intent Classification & Slot Filling for Toxic Game Chat Understanding**

Proyek ini merupakan implementasi model **Slot-Gated Joint Natural Language Understanding (Goo et al., 2018)** untuk menangani dua task utama pada percakapan game online yang bersifat toxic:

1. **Intent Classification** — menentukan kategori toksisitas utama dalam satu utterance.
2. **Slot Filling** — menandai token yang mengandung toksisitas spesifik seperti penghinaan, hinaan implisit, sarkasme, flame, dan toxic-category lainnya.

Dataset yang digunakan merupakan bagian dari **CONDA Shared Task** yang berisi percakapan game dalam konteks multiplayer online match.

---

# 🌐 **1. Dataset Overview**

Dataset berisi chat player dari pertandingan game online dengan struktur:

* **Tokenized text**
* **Intent label (E, I, A, O)**
* **Slot labels per token (T, C, D, S, P, O)**
* Metadata: matchId, conversationId, timestamps.

Distribusi label menunjukkan:

* Intent **O (Other)** mendominasi (imbalance tinggi).
* Slot label **S (slang)/T (toxic)** memiliki peran kuat dalam mendeteksi toxicity.

---

# 🧠 **2. Model: Slot-Gated Joint NLU**

Model yang digunakan adalah **Slot-Gated Intent Attention**, sesuai arsitektur resmi pada paper Goo et al. (2018).
Fitur utama:

### **🔹 BiLSTM Encoder**

Menghasilkan representasi kontekstual token-level (h₁ … hₙ).

### **🔹 Intent Attention**

Membangun context vector global **cᴵ** melalui additive attention.

### **🔹 Slot Gate (Intent Gating)**

Menggabungkan informasi lokal (hᵢ) dan global (cᴵ) untuk memperkuat prediksi slot:

[
g_i = v^\top \tanh(h_i + Wc^I)
]

Slot prediction:
[
y_i^S = \text{softmax}(W(h_i + g_i \cdot h_i))
]

Model ini **TIDAK menggunakan full slot attention**, hanya intent attention—sesuai implementasi baseline & optimized.

---

# ⚙️ **3. Training Setup**

### **Hyperparameters Baseline**

* Epoch: 10
* Batch size: 32
* Embedding dim: 200
* Hidden dim: 128
* Dropout: 0.4
* Learning rate: 1e-3

### **Hyperparameters Optimized**

* Batch size: 16
* LR: 1e-4
* Weight decay: 1e-3
* Class weighting untuk intent & slot
* Word2Vec pretrained embeddings
* Synonym-based data augmentation

---

# 🔀 **4. Workflow Summary**

1. **Preprocessing**

   * Token alignment
   * Label mapping
   * Padding & masking

2. **Model Forward Pass**

   * Encoding → Attention → Gating → Predictions

3. **Joint Loss Computation**
   [
   \mathcal{L} = \mathcal{L}*{intent} + \mathcal{L}*{slot}
   ]

4. **Evaluation**

   * JSA (Joint Semantic Accuracy)
   * F1 Intent
   * F1 Slot

5. **Submission Generation**

   * Model → Test inference → answer_test.txt

---

# 📊 **5. Experimental Results**

## **Baseline — Best Performance**

* **JSA:** **0.8726 (best)**
* **F1 Intent E:** 0.831
* **F1 Intent I:** 0.719
* **F1 Slot T:** 0.966
* **F1 Slot D:** 0.933
* **F1 Slot S:** 0.988

## **Optimized — Best Performance**

* **JSA:** 0.8685
* Slot-F1 meningkat
* Intent-F1 mengalami penurunan, terutama kelas I (Implicit Attack)

---

# 🧩 **6. Why Baseline Outperforms Optimized**

Walaupun optimized menggunakan embedding pretrained, augmentation, dan regularization, **baseline tetap unggul dalam JSA** karena:

### ✔ JSA sangat sensitif terhadap kesalahan intent

Optimized meningkatkan slot-F1, tetapi menurunkan F1 Intent → JSA turun.

### ✔ Augmentasi menambah noise pada sinyal intent

Synonym replacement sering mengubah makna pragmatik (toxic slang).

### ✔ Word2Vec tidak sepenuhnya cocok dengan domain game toxicity

Representasi intent menjadi kurang stabil.

### ✔ Learning rate kecil + weight decay

Model terlalu “conservative”, sulit belajar representasi intent secara efektif.

### ✔ Batch kecil → gradient intent kurang stabil

---
