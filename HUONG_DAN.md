# Project 2 — Tìm kiếm đa tác nhân

Hoàn thiện trong multiAgents.py: ReflexAgent (q1), MinimaxAgent (q2), AlphaBetaAgent (q3), ExpectimaxAgent (q4), betterEvaluationFunction (q5).
Giữ nguyên mã nền, bản đồ và bộ kiểm thử của UC Berkeley, http://ai.berkeley.edu.

## Chạy
Mở terminal trong thư mục multiagent sau khi giải nén. Dùng Python 3; trên Windows có thể thay python bằng py.

```bash
python pacman.py -p ReflexAgent -l testClassic
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=2
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3
python pacman.py -p ExpectimaxAgent -l smallClassic -a depth=2,evalFn=better
python autograder.py --no-graphics
```

Chế độ đồ họa cần Tkinter. Thêm -q để chạy không có cửa sổ; -n 10 -f để thử 10 ván với hạt ngẫu nhiên cố định.

## Cài đặt
- ReflexAgent xét thức ăn, ma, viên năng lượng, phạt đứng yên và đi lại vị trí đã thăm. Bộ đếm được đặt lại mỗi ván.
- Minimax hỗ trợ nhiều ma, tăng độ sâu sau khi tất cả tác nhân đã đi.
- Alpha–Beta giữ thứ tự hành động và chỉ cắt khi bất đẳng thức nghiêm ngặt để đáp ứng kiểm tra số trạng thái.
- Expectimax dùng trung bình đều ở lượt ma.
- Hàm better kết hợp điểm game, số thức ăn/viên năng lượng, nghịch đảo khoảng cách đường đi tới thức ăn và khoảng cách tới ma. Khoảng cách đường đi được lưu đệm; không sinh thêm trạng thái trong lượng giá. Giá trị thắng/thua hữu hạn tránh phép trung bình vô cực không xác định.

## Kết quả kiểm tra bản cuối
Autograder đi kèm: q1 4/4, q2 5/5, q3 5/5, q4 5/5, q5 6/6; tổng 25/25.
q1: openClassic thắng 10/10, trung bình 1248.5.
q5: smallClassic thắng 10/10, trung bình 1127.1.
ReflexAgent, mỗi cấu hình 10 ván với -f: testClassic thắng 10/10 (562.2); mediumClassic 1 ma thắng 10/10 (1610.9); mediumClassic 2 ma thắng 9/10 (1649.8).
Kết quả có thể khác khi đổi hạt ngẫu nhiên hoặc bộ chấm. Điểm trong đề của lớp có thể dùng thang khác với bộ chấm đi kèm ZIP.
Xem autograder_results.txt và các file *_results.txt để biết kết quả chi tiết.
