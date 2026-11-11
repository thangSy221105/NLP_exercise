from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
def M(text): return nbf.v4.new_markdown_cell(text.strip())
def C(text): return nbf.v4.new_code_cell(text.strip())

token_cells = [
M("""
# LAB 08 — Pretrained Transformers and BERT Fine-Tuning
Notebook 1/2: masked language modeling calculation, subword tokenization, special tokens and attention mask.
"""),
C("""
import pandas as pd
from IPython.display import display
"""),
M("""
## 1. Vai trò của LAB 08
LAB 07 xây Transformer từ đầu; LAB 08 dùng một Transformer pretrained rồi thích nghi cho sentiment classification. Mạch chính: pretraining → MLM → tokenizer/contextual representation → fine-tuning → classification.
"""),
M("""
## 2. Learning Outcomes
Sau lab, có thể giải thích pretraining/fine-tuning, MLM, tokenizer, subword, special tokens, attention mask, [CLS], frozen encoder; tính MLM loss; chạy và phân tích pretrained text classifier.
"""),
M("""
## 3. Kiến thức nền
Dùng lại self-attention, multi-head attention, positional information, residual connection và LayerNorm từ LAB 06–07. Trọng tâm ở đây là cách dùng trọng số và tokenizer pretrained.
"""),
M("""
## 4. Pretraining vs. Fine-tuning
Pretraining học biểu diễn từ corpus lớn với mục tiêu như MLM. Fine-tuning tiếp tục cập nhật trọng số pretrained trên nhãn downstream. Scratch khởi tạo ngẫu nhiên và phải học biểu diễn từ dữ liệu task.
"""),
M("""
## 5. Subword Tokenization
Một từ có thể thành nhiều mảnh WordPiece. Điều này giúp biểu diễn từ hiếm bằng các mảnh đã có trong vocabulary; số token vì vậy có thể khác số từ.
"""),
M("""
## 6. Special Tokens
| Token | Vai trò |
|---|---|
| [CLS] | Đại diện chuỗi trong nhiều tác vụ classification |
| [SEP] | Đánh dấu kết thúc/tách sequence |
| [PAD] | Đệm để các chuỗi có cùng độ dài |
| [MASK] | Vị trí bị che trong MLM |
| [UNK] | Từ không có mục từ tương ứng, tùy tokenizer |
"""),
M("""
## 7. Attention Mask
Mask bằng 1 chỉ token thật, bằng 0 cho padding. Khi truyền mask, padding không được tính như nội dung của review.
"""),
M("""
## 8. Masked Language Modeling
Model ước lượng token bị che từ ngữ cảnh hai phía. Với tập vị trí mask M, loss là tổng cross-entropy trên các vị trí đó.
"""),
M("""
## 9. Calculation Exercise 1 — MLM Loss
Đúng token là eats; model gán P(eats)=0.70. Tính loss và xem loss thay đổi thế nào khi xác suất đúng bằng 0.95 hoặc 0.01.
"""),
C("""
import math
import pandas as pd
values = [0.70, 0.95, 0.01]
pd.DataFrame({"P(token đúng)": values,
              "MLM loss -ln(p)": [-math.log(p) for p in values]})
"""),
M("""
Loss nhỏ hơn khi model gán xác suất cao hơn cho token đúng; nếu xác suất đúng rất thấp thì cross-entropy phạt mạnh.
"""),
M("""
## 10. Calculation Exercise 2 — Tokenization
Tokenize chính xác câu “The cats are playing.” bằng tokenizer pretrained được cung cấp. Không đoán tokenization trước khi gọi tokenizer.
"""),
C("""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from pretrained_classification import get_tokenizer
tokenizer = get_tokenizer()
text = "The cats are playing."
encoded = tokenizer(text, padding="max_length", truncation=True,
                    max_length=12, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
token_ids = encoded["input_ids"][0].tolist()
mask = encoded["attention_mask"][0].tolist()
pd.DataFrame({"position": range(len(tokens)), "token": tokens,
              "token_id": token_ids, "attention_mask": mask})
"""),
C("""
lexical = [t for t in tokens if t not in tokenizer.all_special_tokens]
summary = pd.DataFrame([{
    "words in raw text": len(text.rstrip(".").split()),
    "tokens excluding special/padding": len(lexical),
    "tokens including [CLS]/[SEP]/padding": len(tokens),
    "special tokens": ", ".join(t for t in tokens if t in tokenizer.all_special_tokens)
}])
summary
"""),
M("""
Cột attention_mask có 1 cho token nội dung/special token của chuỗi và 0 cho padding. So sánh số word với token thực tế; WordPiece có thể giữ một từ thành một token hoặc tách nó ra.
""")
]

fine_cells = [
M("""
# LAB 08 — Pretrained Transformers and BERT Fine-Tuning
Notebook 2/2: dự đoán, thực nghiệm, đánh giá và phân tích. Các mục được tách theo đúng thứ tự trong W8.pdf.
"""),
C("""
import json
import pandas as pd
from IPython.display import Image, display
from pretrained_classification import get_tokenizer
tokenizer = get_tokenizer()
results = pd.read_csv("result/results.csv")
"""),
M("""
## 11. Prediction Before Experiment
### Prediction 1 — Pretrained vs. from scratch
Dự đoán BERT-Tiny fine-tuned có F1 cao hơn Transformer scratch vì đã có biểu diễn pretrained. Bằng chứng cần xem: accuracy, F1, thời gian và số tham số trên cùng split IMDb.

### Prediction 2 — Frozen vs. fine-tuning
Dự đoán frozen encoder có thể phân loại nhờ representation sẵn có; fine-tuning có thể tăng F1 vì encoder điều chỉnh với sentiment.

### Prediction 3 — 10% training data
Dự đoán pretrained giảm ít hơn scratch do scratch phải học biểu diễn từ ít ví dụ. Kiểm tra bằng điểm test 10% và 100%; so sánh với curve.
"""),
M("""
## 12. Core Implementation — Tokenization
Tokenizer chuyển raw text thành input_ids, attention_mask và (nếu tokenizer/model cần) token_type_ids. Xem bảng token thực tế trong notebook tokenization.ipynb.
"""),
C("""
sample = "The cats are playing."
encoded_sample = tokenizer(sample, padding=True, truncation=True,
                            max_length=12, return_tensors="pt")
{key: value.shape for key, value in encoded_sample.items()}
"""),
M("""
## 13. Core Implementation — Classification Head
Pipeline: raw text → tokenizer → input IDs + attention mask → pretrained encoder → biểu diễn [CLS] → linear classification head → logits → softmax → nhãn. Classification head là phần task-specific; fine-tuning cập nhật nó và encoder.
"""),
M("""
## 14. Experiment 1 — Pretrained vs. From Scratch
Dùng cùng IMDb train/validation/test và max length 64. Scratch và pretrained đều huấn luyện 2 epoch. Cách token hóa khác nhau vì scratch tự học vocabulary còn BERT dùng WordPiece pretrained.
"""),
C("""
results = pd.read_csv("result/results.csv")
results.query("experiment == 'pretrained_vs_scratch'")[
    ["condition","training_examples","max_length","epochs","accuracy","f1",
     "training_seconds","parameters","trainable_parameters"]
].round(4)
"""),
M("""
Đọc kết quả theo cả F1, thời gian và tham số. Chênh lệch ở đây chỉ áp dụng cho cấu hình và split đã chạy; không chứng minh model nào luôn thắng.
"""),
M("""
## 15. Experiment 2 — Feature Extraction vs. Fine-Tuning
Frozen: encoder không cập nhật, chỉ head được train. Fine-tuning: cập nhật encoder và head. So sánh cùng data, max length 128 và 2 epoch.
"""),
C("""
results.query("experiment == 'feature_extraction_vs_fine_tuning'")[
    ["condition","training_examples","max_length","epochs","accuracy","f1",
     "training_seconds","parameters","trainable_parameters"]
].round(4)
"""),
M("""
Frozen vẫn phân loại được vì encoder trả về contextual representations, classifier học ranh giới giữa hai nhãn. Fine-tuning có thể cải thiện vì representations thích nghi trực tiếp với IMDb sentiment.
"""),
M("""
## 16. Experiment 3 — Amount of Training Data
Validation/test cố định; BERT-Tiny dùng 10%, 25%, 50%, 100% của 6.000 train, một epoch mỗi cấu hình. Điểm scratch 10%/100% được thêm làm đối chiếu.
"""),
C("""
results.query("experiment in ['data_efficiency','data_efficiency_scratch']")[
    ["experiment","condition","training_examples","accuracy","f1","training_seconds"]
].sort_values(["experiment","training_examples"]).round(4)
"""),
C("""
from IPython.display import Image, display
display(Image(filename="data_efficiency.png"))
"""),
M("""
Đánh giá data efficiency bằng toàn đường cong: xem mức thay đổi accuracy/F1 khi tăng số review train, không chỉ so hai đầu mút.
"""),
M("""
## 17. Experiment 4 — Maximum Sequence Length
So sánh max length 64, 128, 256 với train/validation/test giữ nguyên. Self-attention có chi phí theo O(T²). Bộ nhớ bên dưới là ước lượng attention-score tensor cho một layer (batch 32, 2 heads, FP32), không phải tổng RAM tiến trình.
"""),
C("""
seq = results.query("experiment == 'sequence_length'").copy()
seq["estimated_attention_score_MiB_per_layer"] = (
    32 * 2 * seq["max_length"]**2 * 4 / (1024**2)
)
seq[["max_length","accuracy","f1","training_seconds",
     "estimated_attention_score_MiB_per_layer"]].round(4)
"""),
M("""
Khi length tăng, model nhận thêm ngữ cảnh nhưng attention memory tăng bình phương. Chỉ nên tăng nếu nội dung giữ lại đem lại cải thiện đủ lớn so với chi phí.
"""),
M("""
## 18. Experiment 5 — Domain Shift
Train chỉ trên IMDb; kiểm tra trên IMDb test (in-domain) và 1.000 câu đánh giá sản phẩm Amazon của UCI (out-of-domain). Amazon không tham gia train/validation.
"""),
C("""
domain = results.query("experiment == 'domain_shift'")
in_domain = results.query(
    "experiment == 'pretrained_vs_scratch' and condition == 'BERT-Tiny fine-tuned'"
)
pd.concat([in_domain.assign(domain="IMDb in-domain"),
           domain.assign(domain="Amazon product reviews out-of-domain")],
          ignore_index=True)[["domain","training_examples","accuracy","precision",
                              "recall","f1","training_seconds"]].round(4)
"""),
M("""
Pretrained không đồng nghĩa model đã thích nghi với mọi domain. Khác biệt về nội dung, từ vựng và độ dài có thể làm hiệu năng thay đổi; chỉ kết luận trên hai tập đã dùng.
"""),
M("""
## 19. Application — Sentiment Classification
Bài toán phân loại review thành negative/positive. Báo cáo accuracy, precision, recall, F1 và confusion matrix.
"""),
C("""
from sklearn.metrics import precision_recall_fscore_support
main = results.query(
    "experiment == 'pretrained_vs_scratch' and condition == 'BERT-Tiny fine-tuned'"
).iloc[0]
main[["accuracy","precision","recall","f1"]].to_frame("BERT-Tiny IMDb test")
"""),
C("""
display(Image(filename="confusion_matrix.png"))
"""),
M("""
Accuracy không cho biết loại lỗi nào phổ biến; confusion matrix cho thấy false positives và false negatives riêng biệt.
"""),
M("""
## 20. Error Analysis
Chọn tối thiểu 3 false positives, 3 false negatives và 2 dự đoán sai confidence cao. CSV lưu văn bản, nhãn đúng/sai và xác suất.
"""),
C("""
errors = pd.read_csv("result/error_examples.csv")
errors.assign(text=errors.text.str.slice(0, 260))[
    ["error_type","text","true_label","predicted_label",
     "positive_probability","confidence","confidence_bin"]
]
"""),
M("""
Phân loại từng ví dụ theo negation, sarcasm, long-range dependency, ambiguous sentiment, domain-specific expression, rare word, tokenization/context, truncation hoặc label ambiguity. Hypothesis phải dựa vào câu thực tế.
"""),
M("""
## 21. Confidence Analysis
Confidence là xác suất lớn nhất của hai lớp. Chia thành Low ≤0.60, Medium >0.60–0.75, High >0.75 rồi so sánh confidence trung bình với accuracy thực tế.
"""),
C("""
pd.read_csv("result/confidence_analysis.csv").round(4)
"""),
C("""
display(Image(filename="confidence_analysis.png"))
"""),
M("""
Confidence cao không bảo đảm prediction đúng. Đồ thị so sánh xác suất model tự tin với tỷ lệ dự đoán đúng trong từng nhóm.
"""),
M("""
## 22. Ablation / Analysis
Ablation được chọn: giảm lượng train từ 100% xuống 10%. Validation/test giữ nguyên; so sánh BERT-Tiny với scratch trên các hàng có trong results.csv. Đây là kiểm tra có kiểm soát trên một seed.
"""),
C("""
ablation = results.query(
    "experiment in ['data_efficiency','data_efficiency_scratch'] and "
    "condition in ['10%','100%','Scratch 10%','Scratch 100%']"
)
ablation[["experiment","condition","training_examples","accuracy","f1"]].round(4)
"""),
M("""
## 23. Reproducibility
Ghi lại model/tokenizer, dataset và split, sequence length, batch size, learning rate, epoch, optimizer, seed, hardware và kết quả. Thời gian trong bảng là thời gian train; không tính tải checkpoint/tokenization/evaluation.
"""),
C("""
import json
with open("result/reproducibility.json", encoding="utf-8") as f:
    repro = json.load(f)
pd.Series({k:v for k,v in repro.items() if k != "audit"}, name="value").to_frame()
"""),
M("""
## 24. AI Usage
AI hỗ trợ boilerplate, API, debugging và visualization. Người học cần tự đọc kết quả, kiểm tra error examples và viết lại phần nhận xét theo hiểu biết của mình. Dataset, split và kết quả chạy được lưu để có thể kiểm tra.
"""),
M("""
## 25. Deliverables
- README.md
- calculations.md
- prediction.md
- tokenization.ipynb
- pretrained_classification.py
- fine_tuning.ipynb
- result/results.csv
- data_efficiency.png, confusion_matrix.png, confidence_analysis.png
- error_analysis.md, reflection.md
"""),
M("""
## 26. Rubric
| Thành phần | Điểm |
|---|---:|
| Theory & calculation | 10 |
| Prediction before experiment | 10 |
| Tokenization & attention mask | 10 |
| Pretrained model implementation | 15 |
| From-scratch vs pretrained | 10 |
| Frozen vs fine-tuning | 10 |
| Data-efficiency experiment | 10 |
| Quantitative evaluation | 10 |
| Error & confidence analysis | 10 |
| Reflection | 5 |
| **Tổng** | **100** |
"""),
M("""
## 27. Individual Learning Check
1. Pretraining học biểu diễn trên corpus lớn; fine-tuning cập nhật cho task cụ thể.
2. MLM che token và dự đoán token gốc từ ngữ cảnh.
3. Tokenizer ánh xạ text thành token IDs model có thể xử lý.
4. Subword giúp biểu diễn từ hiếm/từ mới từ các mảnh phổ biến.
5. [CLS] thường dùng làm biểu diễn toàn chuỗi trong classification.
6. Attention mask đánh dấu token thật và padding.
7. Frozen encoder giữ nguyên trọng số encoder khi train.
8. Fine-tuning trong bài này cập nhật encoder và classification head.
9. Pretrained model đã mang biểu diễn ngôn ngữ ban đầu, hữu ích khi downstream data nhỏ.
10. Pretraining không bảo đảm hết domain shift.
11. Attention so sánh các cặp vị trí nên chi phí tăng theo O(T²).
12. Confidence là mức tin model, không đồng nghĩa đúng.
"""),
M("""
## 28. Error Analysis — Câu hỏi bắt buộc
1. Model hưởng lợi hơn từ ít dữ liệu: so sánh mức giảm F1 100%→10% của hai đường trong data-efficiency plot.
2. Pretrained tốt hơn scratch là bằng chứng pretraining hữu ích trong task/split này, không phải bằng chứng hiểu ngôn ngữ tổng quát.
3. Frozen gần fine-tuned cho thấy encoder pretrained đã có biểu diễn hữu ích và cập nhật toàn bộ có thể đem lại ít lợi ích thêm.
4. Fine-tuning cao hơn nhiều gợi ý encoder cần thích nghi với domain/task sentiment.
5. Domain khác giảm hiệu năng có thể do vocabulary, phong cách, phân bố và label conventions thay đổi.
6. Nếu length tăng mà điểm gần như giữ nguyên nhưng time tăng mạnh thì chọn length ngắn hơn, trừ khi dữ liệu bổ sung cần thiết cho một nhóm trường hợp cụ thể.
"""),
M("""
## 29. Reflection
Sau LAB 07, chúng ta đã biết cách xây dựng Transformer. Sau LAB 08, tại sao thực tế thường không bắt đầu bằng train từ đầu?

Transformer train from scratch phải học biểu diễn từ task data. Với downstream dataset nhỏ, dữ liệu có thể không đủ để học đồng thời vocabulary, ngữ cảnh và tín hiệu sentiment; pretrained model có khởi tạo đã học trên corpus lớn. Trong thí nghiệm này, kết quả pretrained vs scratch là [điền observation từ results.csv], cho thấy [giới hạn kết luận].

Pretraining tạo biểu diễn tổng quát qua mục tiêu như MLM. Tokenizer chuyển văn bản thành subword IDs, attention mask bỏ padding và [CLS] được đưa qua classification head. Fine-tuning cập nhật encoder và head theo nhãn IMDb; frozen chỉ train head. Bảng frozen/fine-tuning cho thấy [điền observation], cho thấy representation pretrained [đã đủ hữu ích/cần thích nghi thêm] trong cấu hình này.

Pretrained model giảm nhu cầu học từ đầu nhưng không miễn phí về tính toán. Self-attention tăng xấp xỉ O(T²) theo sequence length. Kết quả sequence length cho thấy [điền trade-off], vì vậy độ dài nên được chọn dựa trên cả nội dung bị cắt và thời gian/bộ nhớ.

Model pretrained cũng không tự hiểu mọi domain. Khi train IMDb và test Amazon product reviews, kết quả [điền observation] thể hiện khả năng chuyển domain có giới hạn. Tokenization, rare words, truncation, nhãn mơ hồ và confidence cao nhưng sai tiếp tục là rủi ro. Fine-tuning có thể thích nghi model nhưng cũng có khả năng làm giảm một số kiến thức trước đó (catastrophic forgetting). Vì thế nên chọn pretrained model theo downstream task, kiểm tra domain và tính toán có sẵn, đồng thời dùng error analysis thay vì đánh giá chỉ bằng accuracy.
"""),
M("""
## 30. Bridge to LAB 09
LAB 08 dự đoán một nhãn cho toàn chuỗi. LAB 09 chuyển sang token classification/NER: xác định span thực thể như Barack Obama → PERSON và Paris → LOCATION bằng pretrained Transformer.
""")
]

for name, cells in [("tokenization.ipynb", token_cells), ("fine_tuning.ipynb", fine_cells)]:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name":"Python 3","language":"python","name":"python3"}
    nb.metadata.language_info = {"name":"python","version":"3"}
    nbf.write(nb, ROOT / name)
    print("wrote", name, len(cells), "cells")
