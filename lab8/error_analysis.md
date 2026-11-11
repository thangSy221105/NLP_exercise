# Error Analysis — LAB 08

## Thiết lập và confusion matrix

Phân tích trên 2.000 IMDb test của BERT-Tiny fine-tuned, seed 2026, max length 64. Confusion matrix (true rows, predicted columns) là [[648, 352], [326, 674]]: 648 true negatives, 352 false positives, 326 false negatives, 674 true positives.

Bảng dưới lấy ba false positives và ba false negatives có confidence cao nhất trong mỗi nhóm. Hai lỗi tự tin nhất cũng được đánh dấu riêng trong result/error_examples.csv. Confidence được chia thành Low ≤0.60, Medium >0.60 đến 0.75, High >0.75. Sáu ví dụ đều dài hơn 64 WordPiece và bị truncation.

| Loại | Confidence | WordPiece | Trích đoạn | Nhận xét |
|---|---:|---:|---|---|
| False positive, high | 0.7688 | 429 | “This really is a film of two halves. The first … works extremely well.” | Phần đầu khen tình bạn/diễn xuất, phần sau phê bình nhân vật và cốt truyện. Cắt ở 64 token có thể giữ nhiều tín hiệu tích cực đầu review nhưng bỏ phần phê bình quan trọng. |
| False positive, high | 0.7549 | 225 | “Beautiful images … good, interesting but with many shadows.” | Có nhiều từ tích cực ở đầu và lời khen lịch sử/nhân vật, nhưng toàn review kết luận thất vọng. Đây là sentiment pha trộn; model có thể dựa vào từ tích cực bề mặt. |
| False positive | 0.7490 | 459 | “...and god bless him for it. Unfortunately, The Diamonds … is a truly awful movie.” | Cụm “god bless him” và từ ngữ cường điệu dễ bị hiểu tích cực, dù câu sau chê phim rõ ràng; truncation có thể làm mất thêm bằng chứng phía sau. |
| False negative | 0.7072 | 154 | “Good, boring or bad? It's good. Worth your money? … sure.” | Review có khen trực tiếp, nhưng cũng nêu “poorly-executed”, “ripoff” và nhiều phê bình cốt truyện. Model có thể bị kéo về phía từ tiêu cực; nhãn tổng thể thiên positive. |
| False negative | 0.7067 | 207 | “The mystery and its solution was a great noir conceit. I do have some questions…” | Review nêu nhiều thiếu sót và câu hỏi, nhưng kết bằng mong muốn xem lại. Tín hiệu positive yếu và nằm muộn có thể không được thể hiện đầy đủ sau truncation. |
| False negative | 0.7042 | 186 | “No real plot, no character development … The only thing that matters is that this flick is absolutely hilarious.” | Mở đầu bằng phê bình mạnh rồi chuyển sang khen hài hước; cần xử lý tương phản/ngữ cảnh thay vì đếm từ positive/negative. |

Truncation là một giả thuyết có bằng chứng độ dài, không phải nguyên nhân duy nhất đã được chứng minh. Các câu dài cũng chứa sentiment pha trộn, phủ định, mỉa mai hoặc kết luận ở phần sau. Error analysis CSV lưu toàn bộ text, nhãn, xác suất và confidence để kiểm tra.

## Confidence analysis

| Nhóm | Số dự đoán | Confidence trung bình | Accuracy trong nhóm |
|---|---:|---:|---:|
| Low | 897 | 0.5505 | 0.5674 |
| Medium | 1.073 | 0.6617 | 0.7316 |
| High | 30 | 0.7618 | 0.9333 |

Nhóm High ít mẫu; trong đó vẫn có lỗi. Hai false positives tự tin nhất lần lượt có confidence 0.7688 và 0.7549. Confidence cao không đồng nghĩa prediction chắc chắn đúng.

## Câu hỏi bắt buộc

1. **Model nào hưởng lợi hơn khi ít dữ liệu?** Theo accuracy ở 10% và 100%, BERT-Tiny lần lượt đạt 0.5805 và 0.6630, giảm 0.0825; scratch lần lượt 0.5000 và 0.6495, giảm 0.1495. Điều này ủng hộ pretrained ít suy giảm hơn theo accuracy. Tuy nhiên scratch 10% có precision 0.5, recall 1.0; F1 của scratch từ 10% đến 100% giảm từ 0.6667 xuống 0.6324, còn F1 của BERT giảm từ 0.4484 xuống 0.6493. F1 phản ánh sự lệch precision/recall của scratch 10%, nên kết luận cần dựa trên cả metric và prediction behavior.
2. **Nếu pretrained tốt hơn scratch thì chứng minh điều gì?** Nó là bằng chứng pretrained initialization có ích cho task, split và cấu hình đang xét; không chứng minh model hiểu ngôn ngữ tổng quát hoặc luôn thắng. Trong kết quả hiện tại, scratch lại cao hơn ở Experiment 1.
3. **Frozen gần bằng fine-tuned có ý nghĩa gì?** Khi hai điểm gần nhau, encoder pretrained đã cung cấp biểu diễn đủ hữu ích và chỉ train head có thể tiết kiệm chi phí. Ở đây chúng không gần: F1 frozen 0.6471, fine-tuned 0.7229.
4. **Fine-tuning cao hơn frozen nhiều gợi ý gì?** Encoder cần điều chỉnh để phù hợp với tín hiệu sentiment/domain. Ở đây F1 tăng 0.0759, đổi lại số tham số trainable tăng từ 258 lên 4.386 triệu.
5. **Domain khác giảm mạnh có thể do gì?** Vocabulary, phong cách, độ dài, chủ đề và cách gán nhãn khác nhau. Ở đây accuracy IMDb/Amazon là 0.6610/0.6660, nhưng F1 là 0.6654/0.6213; UCI Amazon ngắn và được chọn câu sentiment rõ nên chưa thể xem đây là đánh giá domain shift tổng quát.
6. **Length tăng nhưng điểm gần như không đổi, thời gian tăng mạnh thì sao?** Khi lợi ích không bù chi phí, không nên tăng length. Trong lần chạy này điểm cũng tăng: F1 0.5785→0.6295→0.6642; thời gian 26.6→55.4→130.2 giây. Vì vậy 256 có ích ở cấu hình này nhưng tốn gần 4.9 lần thời gian so với 64; cần cân nhắc theo giới hạn tài nguyên và xác nhận thêm seed.
