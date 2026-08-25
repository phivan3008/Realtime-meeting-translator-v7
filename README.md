# Realtime Meeting Translator (VI ↔ JA)

Phiên dịch hai chiều Việt–Nhật theo thời gian thực cho họp online. Âm thanh
cuộc họp được thu ở máy Windows, xử lý trên GPU server, và trả về phụ đề kèm
bản dịch.

Thiết kế đầy đủ: [`DESIGN.md`](DESIGN.md).

## Ba máy

| Máy | Vai trò |
| --- | --- |
| **Dev PC** | Viết code, chạy unit test, commit/push |
| **Windows Client PC** | Thu âm thanh cuộc họp qua WASAPI loopback, gửi lên server |
| **GPU Server** (pod H100) | Chạy toàn bộ 8 tầng xử lý |

Máy test **pull code từ GitHub**, nên mọi thay đổi phải được push trước khi
chạy thử.

## Đường đi của một câu nói

```
Client                    Server (8 tầng)
──────                    ───────────────
WASAPI loopback           1. VAD (Silero)           cắt thành đoạn có tiếng nói
16 kHz mono 16-bit        2. Buffer Manager         gom thành câu, chốt khi ngắt
chunk 200 ms ─────────►   3. Noise Filter (AST)     bỏ tiếng gõ phím, ho
                          4. Overlap Resolver       hạ giọng chồng lấn
                          5. Diarization (ECAPA)    ai đang nói
                          6. Language ID            tiếng Việt hay tiếng Nhật
                          7. ASR (Whisper large-v3) chuyển thành chữ
      ◄──── final ────    8. Translation (Qwen/vLLM)  chạy ngoài luồng audio
      ◄─ translation ──
```

Câu và bản dịch là **hai message riêng**: câu gửi đi ngay khi Whisper chốt,
bản dịch theo sau và được ghép bằng `sentence_id`. Một lần gọi LLM chậm không
được phép giữ đường vào của âm thanh.

## Cài đặt

Python **3.11** ở cả ba máy. Không phải 3.12.

### GPU Server

```bash
git clone <repo> && cd Realtime-meeting-translator-v7
python3.11 -m venv .venv && source .venv/bin/activate
python --version                      # phải in 3.11.x
python3.11 -m pip install -r server/requirements.lock.txt
```

> Đừng cài vào Python hệ thống. Nó hạ cấp numpy và protobuf bên dưới mọi thứ
> khác đang dùng chung — dự án này đã mất một buổi chiều vì chuyện đó.

Xem trước phép phân giải trước khi tải:

```bash
python3.11 -m pip install --dry-run -r server/requirements.lock.txt
```

### Windows Client

```powershell
git clone <repo>; cd Realtime-meeting-translator-v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                      # phải in 3.11.x
py -3.11 -m pip install -r client\requirements.lock.txt
```

Client **không chạy ML**. Silero VAD nằm ở server — xem `DESIGN.md` mục 3b.

## Chạy

Server cần **hai tiến trình**. vLLM chạy riêng: nó giữ trước một phần GPU lúc
nạp, và tách ra thì LLM restart được mà không rớt cuộc họp.

**Terminal 1 — vLLM:**
```bash
source .venv/bin/activate
python3.11 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.55
```

**Terminal 2 — server âm thanh:**
```bash
source .venv/bin/activate
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Chờ `Application startup complete`, rồi kiểm tra:

```bash
curl -s http://127.0.0.1:8000/health
```

**Cả sáu cờ `*_loaded` phải là `true`.** Cái nào `false` thì dòng `*_error`
bên cạnh nói lý do. Server vẫn khởi động khi thiếu một tầng — thiếu một tầng
còn hơn từ chối cuộc họp — nhưng lúc đó phụ đề không đáng tin.

**Client:**
```powershell
# Nếu pod không truy cập trực tiếp được, mở tunnel ở cửa sổ riêng và để đó:
ssh -N -L 8000:127.0.0.1:8000 <user>@<pod-ssh-host>
```
Cửa sổ phiên dịch:

```powershell
py -3.11 -m client.ui.main --url ws://127.0.0.1:8000
```

Bấm **Bắt đầu**, rồi phát âm thanh cuộc họp. Chữ mờ nghiêng là dự đoán đang
chạy; chữ đậm là câu đã chốt, bản dịch màu xanh ngay dưới. Câu nào không dịch
được sẽ nói rõ lý do và nguyên văn model đã trả lời, thay vì để trống — một ô
trống trông giống lỗi client hơn là một câu trả lời từ máy chủ.

Đóng cửa sổ sẽ gửi `bye` và **chờ máy chủ dịch xong câu cuối**, nên đừng tắt
bằng cách kill tiến trình.

Hoặc chạy bằng script kiểm thử, không giao diện:

```powershell
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 120
```

Server nhận **một cuộc họp mỗi lần**; kết nối thứ hai bị từ chối với mã 1013.

## Kiểm thử

```powershell
# Dev PC: toàn bộ unit test, không cần GPU, không cần sound card
.venv\Scripts\python.exe -m pytest server/tests client/tests common -q
```

Test thật cần phần cứng thật và nằm riêng:

| Thư mục | Chạy ở đâu |
| --- | --- |
| `server/tests_real/` | GPU pod — xem [README](server/tests_real/README.md) |
| `client/tests_real/` | Windows Client PC — xem [README](client/tests_real/README.md) |

## Tinh chỉnh

| Muốn đổi gì | Đọc file nào |
| --- | --- |
| Ngưỡng VAD, cắt câu, nhận dạng người nói, dịch… | [`docs/TUNING.md`](docs/TUNING.md) |
| Câu Whisper bịa cần chặn / giữ | [`server/data/README.md`](server/data/README.md) |

Mỗi thông số trong `TUNING.md` đi kèm **phép đo đã chọn ra nó** và điều gì
hỏng nếu chỉnh sai hướng. Đọc trước khi đổi — phần lớn các số đó nằm giữa hai
vùng đo được rất hẹp.

## Những điều đã học được và đừng làm lại

- **Whisper bịa văn bản với chỉ số tự tin hơn khi phiên âm thật.** Ba lớp
  chắn thống kê không bắt được. Xem `server/data/README.md`.
- **Ép sai ngôn ngữ không báo lỗi.** Whisper trả về văn bản trôi chảy, tự tin
  và sai, rồi tầng dịch dịch trung thành cái vô nghĩa đó.
- **Điểm số của bộ lọc nhiễu không phân biệt được câu bịa với câu thật.** Đã
  thử và đã bác bỏ bằng số liệu.
- **Lịch sử hội thoại có bản dịch trong đó sẽ lái model dịch sai ngôn ngữ.**
  Nó đọc lên như một chuỗi ví dụ mẫu.
- **Đừng bật `condition_on_previous_text`.** Đó là cách một câu bịa thành cả
  đoạn bịa.
- **Voiceprint trên cửa sổ 1 giây không phân biệt được giọng.** Cùng một
  đoạn cho 0.12 theo cửa sổ và 0.77 theo cả câu. Đã thử cắt câu theo đó và đã
  tắt lại — xem `docs/TUNING.md` mục 5b.

## Cấu trúc thư mục

```
client/          thu âm, gửi WebSocket, UI
server/          8 tầng pipeline + FastAPI
  data/          danh sách chặn/giữ, sửa được không cần code
  pipeline/      một file cho mỗi tầng
common/          hợp đồng audio và protocol, dùng chung hai phía
docs/            hướng dẫn tinh chỉnh
```

`common/protocol.py` là nguồn duy nhất cho định dạng audio và các message.
Hai phía import từ đó nên không thể lệch nhau — lệch định dạng audio làm hỏng
tiếng nói một cách âm thầm chứ không ném lỗi.
