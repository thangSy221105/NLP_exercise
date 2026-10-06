# LAB 03 - Word Representations and Embeddings

Corpus: 10.000 documents đầu tiên của C4 tại ../lab01/data/c4-train.00000-of-01024-30K.json.gz.
Hạn nộp theo W3: 23:59 ngày 07/10/2026.

- Họ tên: [Tự điền]
- Mã sinh viên: [Tự điền]

## Mã và kết quả

- cooccurrence.py: tự cài đặt vocabulary, ma trận sparse, cosine và most_similar.
- embedding_utils.py: loader, cấu hình Word2Vec, dẫn chứng corpus, TF-IDF và search.
- word_embedding.ipynb: đầy đủ mục 10-28, bảng kết quả, plot inline và thảo luận tiếng Việt.
- results.csv: các kết quả định lượng xuất từ notebook.
- run_notebook.py: chạy toàn bộ notebook và lưu output vào cùng file.
- requirements.txt: thư viện chạy thực nghiệm.
- calculations.md, prediction.md, error_analysis.md, reflection.md: các phần cá nhân theo đề.

## Cấu hình và giới hạn

Tokenizer giống Lab 2: lowercase, tách câu theo dấu kết câu/newline, giữ token chữ/số và dấu nháy nội bộ.
Co-occurrence: window 1/2/5; min_count 5; top 5.000 từ cộng từ khảo sát đủ count; giữ nguyên vị trí OOV và biên câu.
Word2Vec: baseline CBOW 100 chiều, window 5, min_count 2, epochs 10, negative 5, sample 0.001, workers 1, seed 42, hash SHA256 và fixed window.
Window: 2/5/10; dimension: 50/100/300; thêm Skip-gram 100 chiều/window 5. Tổng cộng sáu cấu hình độc lập.
Weights MiB là bộ nhớ hai ma trận trọng số, không bao gồm metadata Python hay file model.
Downstream dimension: 12 tài liệu nhân tạo / 4 query có nhãn chủ đề định trước; không phải benchmark có nhãn độc lập của C4.
Semantic search trên corpus thật là in-sample retrieval, kèm TF-IDF để đối chiếu và snippet cho người đọc đánh giá.
Các bảng in thành text trực tiếp để trình xem không che bảng HTML. PNG chỉ nhúng trong output notebook, không ghi file ảnh/model riêng.

## Chạy

Từ thư mục lab3 trong PowerShell:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe run_notebook.py

Môi trường .venv hiện được tạo cục bộ và bỏ qua bởi Git. Có thể mở notebook trong VS Code/Jupyter và chọn Python của .venv.
Các nhận xét gắn với cấu hình và output của lần chạy đã lưu; nếu thay dữ liệu/tham số, đọc lại bảng và cập nhật nhận xét Markdown.

## AI assistance statement - mục 28

- Tool: Codex.
- Purpose: hỗ trợ code, chạy thực nghiệm, visualization và bản nháp nhận xét.
- Generated content: implementation, notebook, kết quả chạy và đoạn thảo luận.
- Modified content: sinh viên tự ghi phần thực tế đã chỉnh.
- Verification: kiểm tra count/cosine/window, chạy toàn bộ notebook, đối chiếu output và results.csv.
- W3 yêu cầu người học tự làm phần tính tay, prediction, interpretation, error analysis và reflection; các nhận xét AI là bản nháp để kiểm tra.

## Individual learning check - mục 29

Tự chuẩn bị giải thích distributional hypothesis, CBOW/Skip-gram, window, polysemy và khác biệt TF-IDF với word embedding.

## Kết quả đã thực thi

10.000 documents, 207.954 câu, 3.634.369 tokens; vocabulary co-occurrence 5.003 từ và Word2Vec 55.084 từ.
Notebook đã chạy đủ 14 code cells, có 5 biểu đồ inline và 206 dòng kết quả trong results.csv.
Sáu cấu hình Word2Vec đã được huấn luyện. Các nhận xét tiếng Việt là bản nháp cần người học kiểm tra.

Tài liệu API: [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html), [SciPy CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
