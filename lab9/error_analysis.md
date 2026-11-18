# Error Analysis — LAB 09

Phân tích dùng dự đoán BERT-Tiny trên test split CoNLL-2003. Exact entity match yêu cầu đúng toàn bộ boundary và loại entity. Một câu có thể xuất hiện ở nhiều nhóm vì lỗi tách span đồng thời tạo false positive và false negative.

## False Positive — dự đoán thừa/sai entity (5 ví dụ)

| Câu | Gold entity | Predicted entity | Gold type | Predicted type |
|---|---|---|---|---|
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | JAPAN; CHINA | CHINA | LOC;PER | LOC |
| Nadim Ladki | Nadim Ladki | Nadim | PER | PER |
| Nadim Ladki | Nadim Ladki | Ladki | PER | PER |
| AL-AIN , United Arab Emirates 1996-12-06 | AL-AIN; United Arab Emirates | AL-AIN | LOC;LOC | PER |
| AL-AIN , United Arab Emirates 1996-12-06 | AL-AIN; United Arab Emirates | United | LOC;LOC | LOC |

## False Negative — bỏ sót/sai entity (5 ví dụ)

| Câu | Gold entity | Predicted entity | Gold type | Predicted type |
|---|---|---|---|---|
| SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT . | CHINA | JAPAN; CHINA | PER | LOC;LOC |
| Nadim Ladki | Nadim Ladki | Nadim; Ladki | PER | PER;PER |
| AL-AIN , United Arab Emirates 1996-12-06 | AL-AIN | AL-AIN; United; Arab; Emirates | LOC | PER;LOC;LOC;LOC |
| AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates | AL-AIN; United; Arab; Emirates | LOC | PER;LOC;LOC;LOC |
| Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday . | Asian Cup | Japan; Asian; Cup; Syria | MISC | LOC;MISC;MISC;LOC |

## Boundary Error — sai ranh giới (5 ví dụ)

| Câu | Gold entity | Predicted entity | Gold type | Predicted type |
|---|---|---|---|---|
| Nadim Ladki | Nadim Ladki | Nadim | PER | PER |
| Nadim Ladki | Nadim Ladki | Ladki | PER | PER |
| AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates | United | LOC | LOC |
| AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates | Arab | LOC | LOC |
| AL-AIN , United Arab Emirates 1996-12-06 | United Arab Emirates | Emirates | LOC | LOC |

## Nhận xét từ các ví dụ

1. **Sai ranh giới tên nhiều từ:** `Nadim Ladki` bị tách thành hai span PER; `United Arab Emirates` bị chia thành các từ riêng. Exact match tính span gold là false negative và các span con không khớp là false positive.
2. **Nhầm loại entity:** `AL-AIN` được gán PER trong khi gold là LOC. Viết hoa không đủ để xác định loại; context địa lý cần được học tốt hơn.
3. **Tên nhiều từ và MISC:** `Asian Cup` là một span MISC trong gold nhưng model phân đoạn/đánh nhãn sai. Lỗi này kết hợp nhầm loại với sai boundary.
4. **Quan hệ với metric:** token accuracy có thể ít giảm hơn entity F1 khi phần lớn từ đúng nhưng span chưa khớp. Confusion matrix token-level bổ trợ nhưng không thay thế exact-span metric.
5. **Giới hạn:** đây là ví dụ định tính trên test, không chứng minh nguyên nhân duy nhất. Muốn kết luận entity hiếm khó hơn cần nối lỗi với tần suất trong train. Alignment dùng -100 cho subword phụ giúp loss đúng ở word level nhưng không bảo đảm span/type chính xác.

## Tóm tắt

Trong các ví dụ, lỗi nổi bật là tách sai tên nhiều từ và nhầm loại một số tên riêng. Các nhóm entity dài có F1 thấp hơn ở BERT-Tiny trong cấu hình này. Kết quả phụ thuộc seed 2026, CPU, checkpoint nhỏ và hai epoch.
