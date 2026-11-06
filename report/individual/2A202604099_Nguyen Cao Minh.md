# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV**: 2A202604099
- **Họ và tên**: Nguyễn Cao Minh
- **SEED** (in ở ô đầu notebook): 7141

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | V |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | V |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | V |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.771      0.871        0.857   
2. BoW + MLP (deep learning)         1.000   0.905      0.950        0.947   
3a. Encoder (MiniLM) + kNN           0.896   0.981      0.936        1.000   
3b. Encoder (MiniLM) + LogReg        0.961   0.943      0.952        1.000   
3c. Encoder (MiniLM) + MLP head      0.990   0.914      0.950        0.974   
4a. LLM nhỏ zero-shot                1.000   0.950      0.974        0.974   
4b. LLM lớn zero-shot                1.000   0.850      0.919        0.919   
4c. LLM nhỏ few-shot                 1.000   0.900      0.947        0.947   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.169      40707.130       0.045   
2. BoW + MLP (deep learning)         5.097      38412.828       0.181   
3a. Encoder (MiniLM) + kNN          70.703         24.983      44.810   
3b. Encoder (MiniLM) + LogReg       71.190         34.358      22.540   
3c. Encoder (MiniLM) + MLP head     71.663         34.297      23.947   
4a. LLM nhỏ zero-shot                0.000          0.963     874.683   
4b. LLM lớn zero-shot                0.000          0.811    1162.669   
4c. LLM nhỏ few-shot                 0.000          0.882     946.470   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.070         0.000  
2. BoW + MLP (deep learning)          0.252         0.000  
3a. Encoder (MiniLM) + kNN           60.900         0.000  
3b. Encoder (MiniLM) + LogReg        32.967         0.000  
3c. Encoder (MiniLM) + MLP head      34.687         0.000  
4a. LLM nhỏ zero-shot              1418.490         0.008  
4b. LLM lớn zero-shot              1521.822         0.079  
4c. LLM nhỏ few-shot               1449.556         0.020  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

**1. Thí nghiệm 1.B — Learning rate**

- **Dự đoán:** Với `lr=50`, loss sẽ dao động mạnh hoặc tăng do learning rate quá lớn.
- **Thực tế:** Loss ban đầu tăng vọt lên khoảng 2.6 nhưng sau đó giảm nhanh xuống 0.006, thấp hơn cả `lr=0.5` (0.084).
- **Giải thích:** Dù learning rate lớn gây overshooting ban đầu, gradient descent vẫn hội tụ trên dữ liệu này. Learning rate lớn không phải lúc nào cũng khiến mô hình phân kỳ.

**2. Thí nghiệm 2.A — MLP**

- **Dự đoán:** MLP có F1 tương đương hoặc chỉ cao hơn một chút so với Logistic Regression (F1 = 0.871).
- **Thực tế:** MLP đạt F1 = 0.950, tăng 0.079 (7.9 điểm phần trăm). Validation loss bắt đầu tăng sau epoch 8.
- **Giải thích:** Hidden layer và ReLU giúp MLP học các quan hệ phi tuyến giữa các từ mà Logistic Regression không biểu diễn trực tiếp được, nên cải thiện F1 rõ rệt hơn dự kiến.

**3. Thí nghiệm 5 — Pareto**

- **Dự đoán:** BoW + MLP và Encoder + LogReg sẽ nằm trên đường Pareto.
- **Thực tế:** Cả ba mô hình đều nằm trên đường Pareto, bao gồm BoW + LogReg.
- **Giải thích:** Pareto xét đồng thời F1 và latency, không chỉ chất lượng dự đoán. BoW + LogReg có F1 thấp hơn nhưng tốc độ nhanh nhất nên không có mô hình nào vừa nhanh hơn vừa chính xác hơn để lấn át nó.

## 4. Phân tích 50 case khó


## 5. Thử thách cá nhân


## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày và p95 < 50 ms/tin, tôi chọn BoW + MLP làm phương án triển khai chính.

Các mô hình đạt yêu cầu độ trễ gồm BoW + LogReg (0.070 ms), BoW + MLP (0.252 ms), Encoder + LogReg (32.967 ms) và Encoder + MLP head (34.687 ms). Trong đó, BoW + MLP và Encoder + LogReg đạt cả chất lượng cao (F1 test lần lượt 0.950 và 0.952) lẫn độ trễ thấp.

BoW + MLP phù hợp hơn cho nhà mạng vì tốc độ xử lý khoảng 38,413 tin/giây khi chạy cả lô, chi phí inference thấp và không cần encoder lớn. Encoder + LogReg có F1 nhỉnh hơn nhưng tốc độ chỉ khoảng 34 tin/giây trong phép đo, nên cần chạy nhiều worker song song để đáp ứng lưu lượng thực tế.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 — TODO 4.2 | Yêu cầu implement `build_classification_prompt()` theo docstring. | Viết zero-shot prompt yêu cầu đúng một từ; hỗ trợ chèn các ví dụ few-shot; kết thúc bằng `Message: """..."""\nLabel:`. | Rà soát từng yêu cầu của docstring, giữ format prompt cho zero-shot/few-shot. |
| Phần 4 — TODO 4.3 | Yêu cầu implement `parse_label()` chống output không hợp lệ. | Viết hàm chuẩn hóa chữ hoa, khoảng trắng, dấu câu và prefix `Label:`; mapping `spam→1`, `ham/not spam→0`, không rõ→`None`. | Test thử các trường hợp label có format khác nhau bao gồm các edge case. |
| Phần 4 — TODO 4.4 | Yêu cầu implement `llm_classify_batch()` đúng mô tả theo comment. | Viết vòng lặp tạo prompt, gọi LLM với retry, parse nhãn, đếm invalid, ước tính số token và USD, lưu latency. | Check output xem đã đúng requirements chưa. |

Mình đã học được cách logistic regression, binary classification, embedding hoạt động, và biết cách sử dụng LLM API qua các provider để ứng dụng vào code của mình.


