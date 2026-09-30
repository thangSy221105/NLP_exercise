# Bài tính tay — mục 7, 9, 11 và 18

## 7. Bài tập tính toán

> Hoàn thành phần này trước khi chạy code. Tự trình bày các bước tính và kết quả.

### 7.1 — Bài 1: Unigram

Corpus:

```text
the cat eats fish
the cat likes fish
the dog eats meat
```

1. Vocabulary:
   - [Tự điền]
2. Tổng số token:
   - [Tự điền]
3. Tính P(w) cho `the`, `cat`, `fish`, `dog`:
   - [Tự điền]
4. Kiểm tra tổng xác suất:
   - [Tự điền]

### 7.2 — Bài 2: Bigram

Dùng corpus ở Bài 1.

- P(cat | the): [Tự điền]
- P(dog | the): [Tự điền]
- P(eats | cat): [Tự điền]
- P(likes | cat): [Tự điền]
- Vì sao tổng xác suất các từ đứng sau `the` bằng 1 khi vocabulary/context được xử lý đầy đủ?
  - [Tự giải thích]

### 7.3 — Bài 3: Xác suất câu

Câu: `the cat eats fish`

Theo bigram: P(the) P(cat | the) P(eats | cat) P(fish | eats).

- Các xác suất thành phần và xác suất câu: [Tự điền]
- Nếu thêm một từ vào câu, xác suất cả câu có thể tăng không? Giải thích:
  - [Tự giải thích]

### 7.4 — Bài 4: Sentence ranking

- S1: `the cat eats fish`
- S2: `the dog eats fish`
- Dự đoán câu có xác suất cao hơn trước khi chạy code: [Tự điền]
- Lý do: [Tự giải thích]

## 9. Bài tập suy luận trước smoothing

Corpus:

```text
I like NLP
I like AI
I study NLP
```

Xét P(AI | study):

1. Count(study, AI): [Tự điền]
2. Xác suất MLE: [Tự điền]
3. Ảnh hưởng lên xác suất câu có bigram này: [Tự điền]
4. Việc không quan sát thấy bigram có nghĩa câu không thể xảy ra không? [Tự giải thích]

## 11. Bài tập tính smoothing

Cho C(cat) = 10, V = 5.

1. Trường hợp C(cat, eats) = 0: [Tự tính P_Laplace(eats | cat)]
2. Trường hợp C(cat, eats) = 3: [Tự tính P_Laplace(eats | cat)]
3. Các bigram khác bị thay đổi xác suất thế nào? [Tự giải thích]

## 18. Bài tập tính Perplexity

Cho P(w1) = 0.5, P(w2 | w1) = 0.25, P(w3 | w2) = 0.5.

1. P(W): [Tự điền]
2. PP(W): [Tự điền]
3. Tính lại khi P(w2 | w1) = 0.1: [Tự điền]
4. Vì sao một xác suất nhỏ có thể làm perplexity thay đổi đáng kể? [Tự giải thích]
