# LAB 04 — Text Classification

- Họ tên: [Tự điền].
- MSSV: [Tự điền].
- Hạn nộp theo W4: 23:59 ngày 14/10/2026.
- Task: sentiment classification; negative=0, positive=1.
- Thư mục làm việc: D:/NLP_exercise/lab4.

## Dataset và split

Nguồn chính thức: [IMDb — Stanford](https://ai.stanford.edu/~amaas/data/sentiment/).
Kiểm tra 50.000 labeled reviews, loại exact/normalized duplicates và nhóm normalized text có nhãn xung đột trước sampling.
Subset 10.000 review: 6.000 train, 2.000 validation từ official train; 2.000 test từ official test.
Mỗi split có 50% positive và 50% negative; seed=2026.
C4 của lab trước không có sentiment labels nên không dùng làm dữ liệu supervised của lab này.

data/imdb_10k.jsonl.gz chứa subset cố định và split/ID/hash; data/dataset_audit.json ghi audit, nguồn và SHA256.
Archive 80 MiB và .venv/.jupyter được Git ignore; subset có thể commit để người khác chạy lại.
Không đọc phần unsupervised hay fit embeddings/IDF trên validation/test.

## Triển khai theo thứ tự đề

| Mục | Nội dung |
|---|---|
| 6–9 | Calculations.pdf: bài tính tay |
| 10 | Prediction.pdf: phần prediction |
| 11 | inspection nguồn/subset, missing, duplicates, class/length distributions |
| 12 | chống leakage, kiểm tra normalized hash overlap |
| 13–14 | core TF-IDF + LR, baseline và đủ metrics |
| 15–16 | TF-IDF/LR vs Avg Word2Vec/LR, thêm TF-IDF/SVM |
| 17 | negation và hoán vị cùng bag-of-words |
| 18 | unigram vs uni+bigram; thêm vocabulary và stopwords |
| 19–20 | test evaluation, classification reports, confusion matrices |
| 21 | ba đúng/ba sai, full text và feature contributions |
| 22 | top positive/negative LR weights |
| 23 | application dự đoán review mới |
| 24–25 | AI assistance, individual check |
| 26–29 | deliverables, peer checklist, reflection/mạch nối các lab |

## Files

- classification.ipynb: notebook thực nghiệm, output bảng và plot inline.
- preprocessing.py: normalize, tokenizer, loader, dedup và reproducible split.
- classification.py: fit/evaluate TF-IDF/LR/SVM, Word2Vec và contributions.
- results.csv: toàn bộ metrics, error examples, threshold, model weights và test predictions; phân biệt bằng record_type.
- data/confusion_matrix.png: hình bắt buộc theo đề.
- Error_analysis.pdf: phần phân tích lỗi viết tay.
- run_summary.json: seed, model được chọn và tổng số records.
- Calculations.pdf, Prediction.pdf, Reflection.pdf: bài làm PDF thay thế các Markdown cũ.
- run_notebook.py, requirements.txt: chạy lại.

## Chạy trong PowerShell

    cd D:/NLP_exercise/lab4
    python -m venv .venv
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt
    ./.venv/Scripts/python.exe run_notebook.py

Có thể mở notebook bằng VS Code/Jupyter và chọn .venv.
Tất cả output được lưu trong lab4. Plot nhúng vào notebook; chỉ data/confusion_matrix.png lưu riêng.
Không lưu file model; chạy lại notebook sẽ train lại và cập nhật metrics.

## Cấu hình và quy trình đánh giá

TF-IDF: unigram hoặc uni+bigram, min_df=2, sublinear TF, L2 normalization.
LR: C=1, liblinear, max_iter=1000, threshold=0.5; SVM: C=1, LinearSVC.
Word2Vec: CBOW, dimension 100, window 5, min_count 2, epochs 10, negative 5, workers 1, seed 2026, SHA256 hash, fixed window; average known token vectors rồi L2 normalize.
Tách câu cho Word2Vec; không loại stopwords trong baseline. Chuẩn hóa contraction để giữ not.
Training time gồm fit representation/transform train + fit classifier, không gồm transform validation/test.

So sánh tám cấu hình; chọn theo validation macro-F1, khóa trước test, không refit train+validation.
Precision/Recall/F1 là positive-class, macro-F1 bổ sung; zero_division=0.
N-gram comparison không cap vocabulary ở cả hai cấu hình, chỉ thay ngram_range.
Vocabulary/stopword ablations chỉ thay đúng tham số tương ứng so với baseline.
Threshold trade-off chỉ minh họa trên validation; không tune từ test.

## AI assistance statement — mục 24

Tool: Codex.
Purpose: code, preprocessing, debugging, visualization, thực nghiệm và bản nháp tài liệu.
Generated: implementation, notebook, output chạy và các đoạn tham khảo.
Modified: [Sinh viên ghi phần thực tế đã sửa].
Verified: output thực nghiệm được lưu; sinh viên tự kiểm tra nhận xét và phép tính.

W4 yêu cầu prediction, calculation, interpretation, lựa chọn nguyên nhân error,
reflection và individual check do người học tự làm. Bản AI được ghi rõ là tham khảo để kiểm tra.

## Giới hạn

Subset balanced không đại diện tỷ lệ nhãn thực tế; một seed chưa đủ khẳng định khác biệt nhỏ.
Dedup exact/normalized chưa xử lý near-duplicate. Error examples được chọn để đọc, không dùng ước lượng tỷ lệ nguyên nhân.
Bốn câu application là minh họa, không có benchmark mới. Word2Vec chỉ học từ train IMDb, không dùng pretrained model lớn.

## Tài liệu

- [Stanford IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [scikit-learn — data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

## Kết quả lần chạy seed 2026

Notebook đã chạy đủ 14 code cells, 7 plot inline; 8 cấu hình học + dummy baseline, 18 dòng metrics.
Sampling và split được tạo lại khi cache có seed khác; LR, SVM, Word2Vec và PYTHONHASHSEED dùng SEED=2026.

| Model | Test accuracy | Test F1 positive |
|---|---:|---:|
| TF-IDF unigram 10k + LR | 0.8740 | 0.8751 |
| Avg Word2Vec + LR | 0.7970 | 0.7952 |
| TF-IDF unigram 10k + SVM | 0.8665 | 0.8675 |
| TF-IDF uni+bigram uncapped + LR | 0.8755 | 0.8768 |

Model chọn từ validation: TF-IDF uni+bigram uncapped + LR; TN=865, FP=135, FN=114, TP=886.

## 12. Vì sao fit TF-IDF trước split là data leakage?

Vocabulary biết từ của test và IDF sử dụng số tài liệu/tần suất xuất hiện ở cả train lẫn test. Thông tin test vì vậy ảnh hưởng features dùng để train dù không dùng nhãn test. Test không còn độc lập; kết quả đánh giá có thể thiên lệch. Cần split trước, fit chỉ trên train, rồi transform validation/test.

## Đọc results.csv

Chỉ xuất một file CSV tổng hợp, gồm 2052 records:

| record_type | Nội dung | Số dòng |
|---|---|---:|
| evaluation | Metrics validation/test và dummy baseline | 18 |
| error_analysis | Sáu ví dụ đúng/sai | 6 |
| threshold | Precision/recall/F1 theo threshold | 4 |
| model_weight | Top positive/negative feature weights | 24 |
| test_prediction | 2.000 review, nhãn thật và dự đoán của tám model | 2000 |

Các nhóm giữ nguyên cột và giá trị dữ liệu gốc; cột không áp dụng cho nhóm nào sẽ để trống.
Đọc bảng metrics: pandas.read_csv("results.csv").query("record_type == 'evaluation'").
Notebook chạy lại chỉ ghi results.csv, không tạo các CSV phụ.
