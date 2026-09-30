# 22. Error analysis

Bốn trường hợp dưới đây được chọn từ test khi context trigram đã xuất hiện trong train. Xác suất là MLE của từ được dự đoán. Nhận xét nguyên nhân dựa trên ví dụ cụ thể và cần được sinh viên xem lại trước khi nộp.

## Prediction đúng 1

- Context: alliance is
- Model prediction: a
- Expected / actual: a
- Probability: 0.5000
- Nhận xét: Context đã có trong train; từ a được model xếp đầu và trùng từ quan sát trong test.

## Prediction đúng 2

- Context: nonprofit organization
- Model prediction: dedicated
- Expected / actual: dedicated
- Probability: 0.1667
- Nhận xét: Model chọn dedicated với xác suất 1/6; từ này cũng xuất hiện sau context trong ví dụ test.

## Prediction sai 1

- Context: the downtown
- Model prediction: area
- Expected / actual: alliance
- Probability: 0.5000
- Nhận xét: Hai từ context có thể dẫn tới nhiều cách tiếp nối. Model ưu tiên area theo train, còn test có alliance.

## Prediction sai 2

- Context: is a
- Model prediction: great
- Expected / actual: 501c6
- Probability: 0.0357
- Nhận xét: Model ưu tiên một từ thông dụng là great, còn test có token 501c6. Đây là ví dụ về nội dung web nhiều từ hiếm và context ngắn; cần kiểm tra thêm tần suất trước khi kết luận nguyên nhân chính.

## Giới hạn của phép kiểm tra

Một context có thể có nhiều từ tiếp theo hợp lý. Bốn ví dụ chỉ minh họa hành vi của model trên các câu được chọn, không phải độ chính xác tổng thể. Tỉ lệ n-gram chưa thấy trên test được trình bày ở mục 23 của notebook.
