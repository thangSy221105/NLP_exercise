# LAB 02 — Language Models

**Môn học:** Xử lý ngôn ngữ tự nhiên và ứng dụng  
**Hạn nộp theo đề:** 23:59, 30/09/2026  
**Sinh viên:** [Điền họ tên]  
**Mã sinh viên:** [Điền mã số]  
**Nhóm:** [Điền nhóm, nếu có]

## Mục tiêu

Tự cài đặt unigram, bigram và trigram language model; so sánh MLE với Laplace smoothing; đánh giá bằng perplexity; thử dự đoán từ tiếp theo và xếp hạng câu.

## Cấu trúc thư mục

- `calculations.md` — bài tính tay
- `prediction.md` — dự đoán trước experiment
- `ngram_lm.py` — implementation n-gram language model
- `experiments.ipynb` — thống kê corpus, thí nghiệm và ứng dụng
- `results.csv` — perplexity và kết quả prediction
- `error_analysis.md` — phân tích prediction đúng/sai
- `reflection.md` — câu hỏi tổng kết

## Dữ liệu và preprocessing

- **Corpus / nguồn dữ liệu:** [Điền tên, nguồn hoặc đường dẫn]
- **Số documents dùng:** [Điền]
- **Tokenizer / preprocessing:** [Điền]
- **Quy tắc tách câu và token biên:** [Điền]
- **Xử lý từ OOV:** [Điền]
- **Train / validation / test split:** [Điền]
- **Random seed (nếu có):** [Điền]

## Thứ tự thực hiện

1. Hoàn thành bài tính tay và ghi prediction trước khi chạy experiment.
2. Chốt preprocessing và chia dữ liệu.
3. Hoàn thiện implementation trong `ngram_lm.py`.
4. Chạy các thí nghiệm trong `experiments.ipynb` và ghi kết quả vào `results.csv`.
5. Hoàn thành error analysis và reflection.

## Cách chạy

[Điền hướng dẫn chạy notebook và implementation sau khi hoàn thiện.]

## AI assistance statement

Cập nhật phần này theo đúng hỗ trợ đã sử dụng và quy định của môn học.

- **Tool:** ChatGPT / Codex
- **Purpose:** Tạo bộ khung file theo yêu cầu Lab 2.
- **What was generated:** Cấu trúc thư mục, tiêu đề và câu hỏi gợi ý, khung notebook, function/class stubs.
- **What was modified:** Chưa tạo implementation logic, kết quả thí nghiệm hoặc đáp án bài lab.
- **How the result was verified:** Đối chiếu bộ file với danh sách deliverables; chưa chạy code hoặc notebook.
