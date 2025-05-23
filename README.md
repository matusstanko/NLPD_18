## 🧪 Methods Summary

- **NER Models**: RoBERTa and SpaCy  
- **Classification Models**: Logistic Regression, SVM, KNN, Decision Tree  
- **Tagging Comparison**: With and without XML-tagged entity markup  
- **Evaluation**: Accuracy, F1, Precision, Recall  

---

## 📈 Results Highlights

- **SpaCy’s richer label set** improved entity count-based classifiers over RoBERTa  
- **RoBERTa** showed strong contextual NER but coarser tags like `MISC`  
- **XML-tagging** did **not improve performance**, and slightly reduced F1 scores  
- **BERT without XML** tags performed best overall  

---

## 📚 Dataset

We use the [LIAR2](https://github.com/XuYinjun/LIAR-PLUS) dataset:  
A collection of short political statements labeled by a 6-point truthfulness scale, which we binarized into **true (1)** and **false (0)**.

---

## 👨‍💻 Authors

- **Matus Stanko** – RoBERTa NER pipeline, BERT training, HPC usage  
- **Lucie Navratilova** – SpaCy training, feature extraction, data visualization  
- **David Poda** – Error analysis, dataset inspection, XML-based evaluation  

---

## 🤖 AI & Tool Usage

We used ChatGPT and DeepL to refine writing for clarity and style.  
AI assistance was also used for debugging and learning library syntax, but all modeling and experimentation logic was our own.

---

## 📄 License

For educational purposes only. Please contact us before reusing or distributing.