# 16. Reflection


1. Prediction nào của em sai?

### Student answer

> Dự đoán vocabulary của em chưa chính xác: em đoán khoảng 100.000 terms, nhưng Pipeline A có 193.540 terms. Dự đoán matrix sparse là đúng, với khoảng 99,91% giá trị bằng 0. Kết quả cũng cho thấy document đứng đầu không phải lúc nào cũng gần nghĩa nhất.

2. Kết quả nào bất ngờ nhất?

### Student answer

> Pipeline C có vocabulary nhỏ hơn (41.668 terms) và search performance cao nhất (0,3458), so với Pipeline A (0,2990) và Pipeline B (0,2997). Tuy nhiên, chỉ số này là similarity trung bình, không trực tiếp đo độ liên quan thực tế.

3. Experiment nào cung cấp evidence mạnh nhất?

### Student answer

> So sánh Pipeline A, B và C cung cấp evidence định lượng rõ nhất vì các pipeline được thử trên cùng corpus. Kết quả cho thấy cách preprocessing ảnh hưởng đến vocabulary, sparsity và điểm tìm kiếm.

4. Failure case quan trọng nhất là gì?

### Student answer

> Với query “medical image classification”, kết quả đầu là document 18971 về phân loại công trình theo tiêu chí môi trường. Từ “classification” trùng với query nên tài liệu được xếp cao, dù không nói về ảnh y tế. Nhãn relevant hiện tại cũng cần được kiểm tra lại vì không khớp nội dung tài liệu.

5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?

### Student answer

> Em sẽ thử sentence embeddings để tìm tài liệu gần nghĩa hơn, đồng thời giữ TF-IDF làm baseline. Em cũng sẽ kiểm tra lại relevance labels và đánh giá bằng các nhãn được đọc, xác nhận thủ công.

6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?

### Student answer

> Em dùng AI để giải thích công thức, code, hỗ trợ kiểm tra notebook và lưu kết quả vào CSV, đồng thời gợi ý cách phân tích các truy vấn. Em vẫn cần đối chiếu kết quả với văn bản gốc; việc kiểm tra cho thấy một số nhãn relevant ban đầu chưa chính xác.

