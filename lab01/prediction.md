# 6. Part C — Prediction Before Experiment

Trước khi mở corpus 30K documents, ghi ít nhất ba prediction.

## Prediction 1 — Vocabulary

Nếu corpus có 30K documents, vocabulary sẽ có khoảng bao nhiêu unique terms?

### Student prediction

> Em dự đoán vocabulary có khoảng 100.000 unique terms vì corpus gồm nhiều văn bản với nội dung đa dạng.

## Prediction 2 — Sparsity

TF-IDF matrix sẽ dense hay sparse? Tỷ lệ zero entries có thể lớn đến mức nào?

### Student prediction

> Em dự đoán TF-IDF matrix sẽ sparse, với hơn 99% giá trị bằng 0, vì mỗi document chỉ chứa một phần nhỏ vocabulary.

## Prediction 3 — Search

Với một query bất kỳ, các documents đứng đầu kết quả tìm kiếm có nhất thiết là documents gần nghĩa nhất không?

### Student prediction

> Em dự đoán document đứng đầu không phải lúc nào cũng gần nghĩa nhất, vì TF-IDF chủ yếu dựa trên mức độ trùng khớp từ.

