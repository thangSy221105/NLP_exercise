# Lab 2 — N-gram Language Models

Thư mục này chứa Lab 2 chạy trên 10.000 documents đầu tiên của corpus C4 tại D:\NLP_exercise\lab01\data\c4-train.00000-of-01024-30K.json.gz.

## Nội dung

- experiments.ipynb: các mục 13–23 theo W2, bảng kết quả, biểu đồ inline và thảo luận tiếng Việt.
- ngram_lm.py: tự cài đặt đếm n-gram, MLE, Laplace, xác suất câu, log probability, perplexity và dự đoán từ.
- results.csv: 50 dòng kết quả perplexity, prediction, sentence ranking, error analysis và n-gram coverage.
- calculations.pdf, prediction.pdf, refliction.pdf: bài tính tay, dự đoán và reflection.
- error_analysis.pdf: phân tích bốn ví dụ đúng/sai rút từ lần chạy 10k.

## Chạy notebook

Mở experiments.ipynb trong Jupyter và chạy từ trên xuống. Notebook cần NumPy, pandas, Matplotlib và IPython. Dữ liệu được chia theo document với seed 42 và tỉ lệ 70/15/15. Các bảng và biểu đồ được lưu trong output của chính notebook; không có file ảnh plot riêng.
