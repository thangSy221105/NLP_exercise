# Các bài tính tay

## 6. Bài 1 - Co-occurrence

Corpus:

    the cat eats fish
    the dog eats fish
    the cat likes milk
    the dog likes meat

Context window k = 1. Không sử dụng Python.

- Vocabulary và thứ tự chiều: [Tự ghi]
- Quy ước context và biên câu: [Tự ghi]
- Vector cat: [Tự tính]
- Vector dog: [Tự tính]
- Vector eats: [Tự tính]
- Vector likes: [Tự tính]

## 6. Bài 2 - Cosine similarity

x = [1, 2, 1]; y = [2, 4, 2].

- Tích vô hướng: [Tự tính]
- Độ dài hai vector: [Tự tính]
- Cosine similarity: [Tự tính]
- Khác độ lớn nhưng cosine bằng 1 có ý nghĩa gì? [Tự giải thích]

## 7. Bài 3 - Semantic similarity

doctor = [0.8, 0.1, 0.7]
physician = [0.7, 0.2, 0.8]
banana = [-0.2, 0.9, -0.1]

- Dự đoán từ gần doctor hơn trước khi tính: [Tự điền]
- cos(doctor, physician): [Trình bày phép tính]
- cos(doctor, banana): [Trình bày phép tính]
- Nhận xét sau khi tính: [Tự viết]

## 8. Bài 4 - Sparse vs dense

Vocabulary 10.000 từ. Word-context vector 10.000 chiều có 30 phần tử khác 0.
Embedding 300 chiều có hầu hết phần tử khác 0.

1. Biểu diễn nào sparse? [Tự trả lời]
2. Biểu diễn nào dense? [Tự trả lời]
3. Vì sao dense có thể thuận lợi cho semantic similarity? [Tự trả lời]
4. Dense có chắc tốt hơn trong mọi bài toán không? [Tự trả lời]

## 16. Training examples - CBOW vs Skip-gram

Câu: the cat eats fish. Window = 1. Không chạy code.

- Quy ước tại biên câu: [Tự ghi]
- Các training examples CBOW: [Tự liệt kê input và target]
- Các training pairs Skip-gram: [Tự liệt kê input và target]
- Khác nhau về input/target: [Tự giải thích]

## 23. Bài tập tính analogy

king = [8, 2, 7]; man = [5, 1, 5]; woman = [5, 3, 5].

- king - man + woman: [Tự tính]
- Vector mới có thể đại diện cho quan hệ nào? [Tự giải thích]
