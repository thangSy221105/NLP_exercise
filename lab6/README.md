# LAB 06 — Attention Mechanism

- Họ tên: [Tự điền]; MSSV: [Tự điền].
- Hạn nộp theo W6: 23:59 ngày 28/10/2026.
- Thư mục: D:/NLP_exercise/lab6.
- Seed=2026; task IMDb sentiment: negative=0, positive=1.

## Dataset và evaluation

Giữ đúng 10.000 reviews/split Lab 5: 6.000 train, 2.000 validation, 2.000 test.
Dataset/audit copy vào data và kiểm SHA256. Vocabulary chỉ fit train, cap=10.000 IDs, min_count=2,
PAD=0, UNK=1; tokenizer giữ negation, giống Lab 4/5.
Cả LSTM và attention nhìn cùng prefix 100 tokens; không dùng test fit vocabulary hoặc chọn checkpoint.

## Các phần theo W6

| Mục | Nội dung |
|---|---|
| 5–11 | scores, softmax, context và calculations.md |
| 12 | prediction.md, ghi rõ là prediction mẫu |
| 13–15 | scaling, temperature và entropy |
| 16–17 | NumPy core, stable softmax và [1000,1001,1002] |
| 18 | toy attention với query/vectors giả lập |
| 19 | đổi sentiment/topic query trên cùng keys/values |
| 20–22 | self-attention, ma trận I/like/cats và heatmap |
| 23 | learned attention của hai câu trong đề, cat/adopted/sleeping |
| 24 | LSTM final state vs self-attention classifier, metrics/cost/parameters |
| 25, 29 | T² elements, memory estimates và timing tới T=1000 |
| 26 | ba câu đề bài, prediction/probability/heatmap, ít nhất một đúng và một sai |
| 27 | ba đúng/ba sai, attention patterns, value contributions và occlusion |
| 28 | so sánh recurrent processing và attention, giới hạn order/cost |
| 30–35 | AI statement, individual check, deliverables/rubric và reflection |

## Giả định cần biết

Mục 21 của W6 chưa có giá trị cụ thể cho X/W_Q/W_K/W_V. Notebook/calculations dùng ví dụ tự chọn,
ghi rõ không phải dữ kiện được cho: X=[[1,0],[0,1],[1,1]], cả ba W=I_2.
Toy sentiment/topic queries dùng vectors gán tay; không gọi đây là learned semantic embeddings.
Heatmaps mục 23/26/27 dùng model thật học từ train IMDb, không phải parser hoặc pretrained language model.

## Models và training

LSTM 1 layer unidirectional, embedding 64, hidden 64, final real timestep.
Attention: embedding 64 → learned Q/K/V 64 → single-head scaled dot-product → masked mean context → linear.
Không built-in attention/MultiheadAttention, multi-head, positional encoding, Transformer block, residual hay FFN.
Vì chưa có positional signal, attention outputs permutation-equivariant; mean pooling làm classifier invariant với token permutation.
Điểm này giới hạn cách hiểu bảng sequence order tổng quát trong đề.

Cùng batch=128, Adam lr=0.002, dropout=0.2 sau pooled context, clip_grad_norm=1, epochs=8, seed=2026, CPU 4 threads.
Cùng initialization seed và embedding shape. Số parameters khác nhau vì architecture khác.
Chạy đủ tám epochs; checkpoint mỗi model chọn min validation loss. Mở test sau khi khóa checkpoints.
Cached checkpoints chỉ reuse khi dataset/vocabulary/config/source/preprocessing fingerprints trùng.

Training seconds gồm validation trong loop; fit_seconds tách phần update. Time phụ thuộc CPU/warm-up.
Metrics: binary positive=1, threshold=0.5, zero_division=0; có macro-F1 và confusion matrices.

## Files và cách chạy

- self_attention.ipynb: toàn bộ experiment, bảng và plot inline.
- attention_numpy.py: NumPy scores/softmax/context, mask và temperature.
- attention_models.py: classifier và training/evaluation.
- preprocessing.py: tokenizer, vocabulary, data loading/retention.
- analysis_utils.py: attention trace và diagnostics.
- results.csv: toàn bộ bảng kết quả, lọc bằng record_type.
- data/: dataset/audit/vocabulary/run summary và tất cả ảnh.
- data/attention_heatmap.png và data/confusion_matrix.png: hai hình theo deliverables.
- calculations.md, prediction.md, error_analysis.md, reflection.md: bản tham khảo để tự hoàn thiện và thay bằng PDF.
- run_notebook.py, requirements.txt: chạy lại; temporary checkpoints Git ignored.

PowerShell:

    cd D:/NLP_exercise/lab6
    python -m venv .venv
    ./.venv/Scripts/python.exe -m pip install -r requirements.txt
    ./.venv/Scripts/python.exe run_notebook.py

Mọi output ở lab6; không sửa các lab cũ. Chỉ có một file CSV kết quả.
Đọc: pandas.read_csv("results.csv", low_memory=False), rồi lọc record_type.

## Diễn giải attention

Heatmap hàng=query, cột=key/value. Với masked mean pooling, beta_j là mean attention received từ valid queries.
Ở eval, signed value contribution_j=beta_j*(w^T v_j), sum contributions+bias≈logit.
Beta không bằng contribution; cả hai không tự là causal importance.
Occlusion thay đổi nhiều attention scores và có thể kéo token mới vào prefix, nên chỉ hỗ trợ hypothesis.
Không suy ra model hiểu grammar/reasoning từ heatmap hoặc luôn chọn đúng từ con người coi quan trọng.

## AI assistance — mục 30

Tool: Codex.
Purpose: API, preprocessing, implementation, debugging, training loop, plots và documentation.
Generated: code, notebook, output thực nghiệm và các bản nháp.
Modified: [Sinh viên ghi phần thực tế đã sửa].
Verified: output chạy được lưu; người học cần tự kiểm tra suy luận và phép tính.

W6 yêu cầu calculation, prediction cá nhân, Q/K/V explanation, attention interpretation, error analysis,
reflection và individual check do người học tự thực hiện.

## Tài liệu

- [PyTorch Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Attention is not Explanation](https://aclanthology.org/N19-1357/)

## Kết quả đã chạy — seed 2026

Hai models được train đủ 8 epochs; checkpoint tốt nhất của cả hai ở epoch 4 theo validation loss.

| Model | Test accuracy | Precision | Recall | F1 positive | Parameters |
|---|---:|---:|---:|---:|---:|
| LSTM L=100 | 0.6810 | 0.6607 | 0.7440 | 0.6999 | 673345 |
| Attention L=100 | 0.7585 | 0.7707 | 0.7360 | 0.7529 | 652545 |

Attention giảm false positive nhưng tăng nhẹ false negative so với LSTM. Kết quả một seed chưa là kết luận tổng quát.
Ba application sentences có một đúng và hai sai, được phân tích bằng heatmaps/contributions.
Error analysis có ba đúng/ba sai và occlusion; full text/cue/truncation/OOV được lưu.
Phép hoán vị cùng multiset cho cùng attention prediction trong sai số số học, xác nhận giới hạn chưa có positional encoding.

## Trạng thái hoàn thiện

Notebook đã chạy đủ 17 code cells, không có error output; 12 plot nhúng inline và 12 PNG trong data.
results.csv có 4.548 records, gồm metrics, learning curves, attention distributions, cost, applications,
3 đúng/3 sai và 4.000 test predictions. Không tạo CSV phụ.
Phân rã logit từ signed contributions khớp đến sai số float32 (max quan sát khoảng 7.75e-7).
Môi trường riêng lab6/.venv đã cài xong và được dùng chạy notebook cuối.
