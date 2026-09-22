# Part B — Calculation Exercises

> Hoàn thành phần này bằng tay trước khi dùng code để kiểm chứng.

## Exercise 1 — Count Vector

Corpus:

- D1 = `cat eats fish`
- D2 = `dog eats fish`
- D3 = `cat likes fish`

Vocabulary: `[cat, dog, eats, fish, likes]`

### D1

Đáp án: TODO

### D2

Đáp án: TODO

### D3

Đáp án: TODO

## Exercise 2 — TF

Với D1 = `cat eats fish`:

- `tf(cat, D1)` = TODO
- `tf(eats, D1)` = TODO
- `tf(fish, D1)` = TODO
- Tổng TF = TODO

## Exercise 3 — IDF

Sử dụng `idf(t) = log(N / df(t))`, với `N = 3`.

| Term | df | IDF |
|---|---:|---:|
| cat | 2 | TODO |
| dog | 1 | TODO |
| eats | 2 | TODO |
| fish | 3 | TODO |
| likes | 1 | TODO |

Term có IDF thấp nhất: TODO

## Exercise 4 — TF-IDF

Với D1 = `cat eats fish`:

- `tfidf(cat, D1)` = TODO
- `tfidf(eats, D1)` = TODO
- `tfidf(fish, D1)` = TODO

Giải thích: TODO

## Exercise 5 — Cosine Similarity

Với `x = [1, 1, 1]`, `y = [1, 1, 0]`:

`cos(x, y)` = TODO

Giải thích vì sao không bằng `2/3`: TODO

## Exercise 6 — Prediction

Với query `medical image classification`:

1. Similarity cao nhất: TODO
2. Similarity thấp nhất: TODO
3. Term có thể có IDF thấp: TODO
4. Ranking khi bỏ IDF: TODO

