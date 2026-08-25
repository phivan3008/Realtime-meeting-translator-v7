# Danh sách từ — chặn và giữ

Ba file văn bản, sửa được mà không cần đụng vào code. Server đọc chúng lúc
khởi động, nên **sửa xong phải khởi động lại uvicorn**.

| File | Vai trò |
| --- | --- |
| `hallucinations.txt` | Chặn theo **nguyên câu** |
| `hallucination_patterns.txt` | Chặn theo **mẫu** (biểu thức chính quy) |
| `keep.txt` | **Giữ** — thắng cả hai danh sách trên |

Đổi chỗ đặt bằng biến môi trường `MEETING_DATA_DIR`.

## Vấn đề ba file này giải quyết

Whisper không trả về chuỗi rỗng khi nghe gần-im-lặng. Nó trả về một câu trôi
chảy chưa ai nói — thường là câu kết thúc video, vì phần lớn dữ liệu huấn
luyện của nó là video.

Điều làm chuyện này khó chặn: **Whisper tự tin hơn khi bịa so với khi phiên
âm thật.**

| | câu bịa | ngưỡng chặn |
| --- | --- | --- |
| `no_speech_prob` | 0.02 | < 0.6 lọt |
| `avg_logprob` | −0.15 | > −1.0 lọt |
| `compression_ratio` | 1.5 | < 2.4 lọt |

Cả ba lớp chắn thống kê đều cho qua. Nên phải chặn theo nội dung.

## Điểm số của bộ lọc nhiễu không dùng được ở đây

Đã thử và đã bác bỏ. Bộ lọc nhiễu chấm điểm tiếng nói cho mọi utterance trước
khi Whisper nhìn thấy, và một câu bịa từng đến với điểm 0.03 trong khi câu
thật đạt 0.66 trở lên — trông như một quy tắc không cần danh sách nào.

Hai lần chạy sau đó bác bỏ:

| câu | điểm | thực tế |
| --- | --- | --- |
| `これ` | 0.01 | **tiếng nói thật** |
| `おいっ` | 0.04 | **tiếng nói thật** |
| `Hãy subscribe cho kênh La La School…` | **0.77** | bịa |

Câu bịa nằm giữa dải câu thật. Không có ngưỡng nào tách được. Điểm số vẫn
được in ra trong bài test vì đáng đọc, nhưng **không quyết định gì**.

## Cách thêm một câu

1. Chạy real test end-to-end.
2. Đọc phần `Committed sentences` và `Running text`.
3. **Nghe lại đoạn ghi âm** tại thời điểm đó.
4. Chỉ thêm câu bạn đã xác nhận **không ai nói**.

Bước 3 không bỏ qua được. Đoán ở đây nghĩa là âm thầm xoá tiếng nói thật, và
người dùng sẽ không bao giờ biết câu của họ đã biến mất.

## Chọn giữa nguyên câu và mẫu

Dùng **nguyên câu** khi câu bịa cố định.

Dùng **mẫu** khi cùng một câu xuất hiện với một chỗ thay đổi được. Đây không
phải giả định — danh sách nguyên câu đã để lọt:

```
đã chặn:  Hãy subscribe cho kênh Ghiền Mì Gõ  Để không bỏ lỡ…
lọt qua:  Hãy subscribe cho kênh La La School Để không bỏ lỡ…
```

Chỉ khác tên kênh, và tên kênh thì vô hạn.

## Viết mẫu cho an toàn

- **Neo cả câu.** Mẫu được khớp bằng `fullmatch`; đừng thêm `.*` ở hai đầu.
- **Giới hạn độ dài khe:** `.{1,40}` chứ không phải `.*`.
- **Giữ phần đuôi đặc trưng.** `hãy đăng ký kênh .*` một mình sẽ chặn cả
  `Hãy đăng ký kênh Teams cho dự án này`.
- **Viết cả biến thể không dấu:** `[ãa]`, `[ểe]`, `[ủu]` — Whisper hay bỏ dấu
  khi nghe không rõ.

Mẫu sai cú pháp bị bỏ qua kèm dòng log `Ignoring bad pattern`, không làm chết
server.

## Thử một mục trước khi thêm

```bash
python3.11 - <<'EOF'
import sys; sys.path.insert(0, '.')
from server.wordlists import Hallucinations
h = Hallucinations()
for line in [
    "CÂU BẠN MUỐN CHẶN",
    "Một câu họp thật gần giống nó",
]:
    print("CHẶN " if h.is_invented(line) else "giữ  ", line)
EOF
```

Luôn thử cả câu thật gần giống. Một mẫu chặn đúng câu bịa mà cũng chặn câu
họp thật thì tệ hơn không có mẫu.

## Khi nào dùng `keep.txt`

Khi một mẫu chặn nhầm câu thật trong cuộc họp của bạn, nhưng bạn không muốn
sửa mẫu vì nó vẫn đang chặn đúng những câu khác. Ví dụ: một cuộc họp
marketing thật sự bàn về việc đăng ký kênh.

`keep.txt` thắng cả hai danh sách chặn.
