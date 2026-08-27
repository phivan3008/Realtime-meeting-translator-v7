# Thiết kế: Phiên dịch cuộc họp VI ↔ JA thời gian thực

Phiên dịch hai chiều Việt–Nhật cho họp online. Âm thanh thu ở máy Windows, xử
lý trên GPU server, trả về phụ đề kèm bản dịch.

Tài liệu này mô tả **các module và ranh giới giữa chúng**. Con số cụ thể và
phép đo đã chọn ra chúng nằm ở [`docs/TUNING.md`](docs/TUNING.md); cách chạy
nằm ở [`README.md`](README.md).

## 1. Ba máy

| Máy | Vai trò |
| --- | --- |
| Dev PC | Viết code, chạy unit test, push lên GitHub |
| Windows Client PC | Thu WASAPI loopback, gửi lên server, hiển thị |
| GPU Server (pod H100) | Toàn bộ pipeline xử lý |

Python **3.11** ở cả ba. Máy test pull code từ GitHub.

## 2. Client

Không chạy ML. Ba module:

| Module | Việc |
| --- | --- |
| `client/audio/` | WASAPI loopback qua `pyaudiowpatch`, 16 kHz mono 16-bit, chunk 200 ms (6400 byte) |
| `client/net/` | WebSocket: binary là PCM thô, text là JSON điều khiển |
| `client/ui/` | PySide6 |

Client gửi **toàn bộ** audio kể cả khoảng lặng (~256 kbps). Không lọc gì.

`client/ui/` tách làm bốn để phần lớn chạy được trên máy không màn hình, không
sound card:

- `transcript.py` — nội dung màn hình, thuần Python. Hàng khoá theo
  `sentence_id`, cập nhật tại chỗ.
- `session.py` — thu âm và socket trên luồng riêng với asyncio loop riêng,
  chạm giao diện chỉ qua Qt signal.
- `window.py` — widget. `sentence_html`/`partial_html` là hàm thuần.
- `record.py` — ghi cuộc họp ra hai file, thuần Python.
- `main.py` — điểm vào.

Chữ mờ nghiêng là dự đoán đang chạy, nằm ở nhãn riêng dưới transcript. Chữ đậm
là câu đã chốt, bản dịch xanh ngay dưới. Câu không dịch được hiện lý do kèm
nguyên văn model — ô trống trông giống lỗi client hơn là câu trả lời.

Transcript được **escape**, không render: Whisper sẵn sàng cho ra `<b>`.

Mỗi cuộc họp ghi ra **hai file**: biên bản (câu đã chốt kèm bản dịch, cho người
đọc lại) và nhật ký gỡ lỗi (mọi message theo thứ tự tới, kể cả chữ mờ). Bản
dịch tới muộn và nhãn người nói được sửa đều làm biên bản **viết lại**, nên nó
không bao giờ lệch với màn hình.

## 3. Server — tám tầng

Mỗi tầng một file trong `server/pipeline/`. Mọi tầng chạy trên **luồng đọc
socket**, trừ tầng dịch.

Một tầng ném lỗi thì **chỉ tầng đó** mất, câu vẫn đi tiếp qua các tầng còn lại
— `Analysis` giữ giá trị mặc định, mà mặc định chính là hành vi khi thiếu tầng
đó. Hỏng ba lần liên tiếp thì tắt tầng; lỗi **thiết bị** (`cuda`, `cudnn`,
`cublas`, `out of memory`) thì tắt ngay lần đầu, vì vào lại tầng đã hỏng là thứ
giết tiến trình. Client được báo bằng `error` không fatal.

| # | Tầng | Model / kỹ thuật | Việc |
| --- | --- | --- | --- |
| 1 | VAD | Silero (CPU) | Cắt stream thành đoạn có tiếng nói, giữ pre-roll để không cụt phụ âm đầu |
| 2 | Buffer Manager | — | Gom thành câu, chốt khi ngắt hoặc quá dài |
| 3 | Noise Filter | AST (AudioSet, CPU) | Bỏ tiếng gõ phím, tiếng ho |
| 4 | Overlap Resolver | `pedalboard` gate + compressor | Hạ giọng nhỏ khi chồng lấn |
| 5 | Diarization | ECAPA-TDNN (SpeechBrain) | Ai đang nói |
| 5b | Reclustering | agglomerative, cosine | Gom cụm lại cả cuộc họp, **sửa nhãn đã gửi** |
| 6 | Language ID | VoxLingua107 ECAPA | `vi`, `ja`, hoặc rỗng |
| 7 | ASR | faster-whisper large-v3 | Chuyển thành chữ |
| 8 | Translation | Qwen3.5-9B qua vLLM | Dịch, **ngoài luồng audio** |

### Ghi chú từng tầng

**1. VAD ở server, không ở client.** `silero-vad` kéo theo `torch` lên Windows
và `torchaudio` fail ABI ở đó. Ngoài ra Buffer Manager cần biết các khoảng
ngắt, mà đặt VAD cạnh nó thì protocol phụ báo ngắt biến mất.

**2. Buffer Manager.** Ranh giới câu: **ngắt** (VAD đóng đoạn) hoặc **quá dài**
(>7 s). Cắt vì quá dài thì lùi về khung 32 ms yên tĩnh nhất gần đó — Whisper
biến nửa từ thành từ khác. Trong lúc câu chưa chốt, cứ 600 ms xuất một cửa sổ
partial; cửa sổ này bị **giới hạn 4 giây cuối**, vì giải mã lại cả câu mỗi
600 ms tốn hơn toàn bộ các câu đã chốt cộng lại.

Có tầng cắt câu theo **đổi giọng** (`speaker_change.py`) nhưng **đang tắt**:
voiceprint cửa sổ 1 giây không phân biệt được giọng. Xem `TUNING.md` 5b.

**3. Noise Filter chạy CPU** (`NOISE_DEVICE`) — nhường VRAM cho Whisper và
vLLM, và trên pod thật AST trên GPU đổ ở cuDNN rồi segfault tiến trình ở lần
gọi sau.

**Rụt rè có chủ đích.** Chỉ bỏ khi speech score thấp **và**
có lớp non-speech đạt ngưỡng. Hai điểm gần 0 không phải bằng chứng, đó là model
đang không biết. Bỏ nhầm câu thật thì mất luôn; để lọt tiếng ho chỉ tốn một lần
gọi Whisper. Câu bị bỏ vẫn báo về client kèm nhãn (`kept: false`).

**4. Overlap Resolver không tách nguồn.** Gate không tách được hai giọng, nó
chỉ hạ những gì thấp hơn hẳn giọng đang át. Ngưỡng đặt theo **đỉnh** chứ không
theo RMS — detector của pedalboard so theo đỉnh.

**5. Diarization.** Gọi `speechbrain` trực tiếp, không qua `pyannote.audio`:
wrapper của pyannote truyền tham số mà speechbrain không nhận, lỗi trước cả khi
nạp model. Voiceprint lấy từ **audio thô**, chưa qua tầng 4 — gate cắt cả âm
tiết nhỏ, mà âm tiết nhỏ vẫn mang chất giọng.

**5b. Gom cụm lại.** Tầng 5 phải trả lời **ngay**, nên câu trả lời phụ thuộc
thứ tự cuộc họp và không sửa được. Định kỳ gom cụm **cả cuộc họp** từ đầu và
gửi về những nhãn đổi qua message `speakers`. Hàng khoá theo `sentence_id` ở cả
hai đầu nên đây là cập nhật hàng đã có trên màn hình.

**6. Language ID chỉ đọc điểm của đúng hai ngôn ngữ** rồi chuẩn hoá lại giữa
chúng. Model biết 107 thứ tiếng và sẽ trả lời tiếng Hàn cho tiếng Nhật nếu được
tự do. Hai điểm quá gần nhau thì trả rỗng: **ép sai ngôn ngữ không báo lỗi**,
Whisper vẫn trả về văn bản trôi chảy, tự tin và sai.

**7. ASR.** Partial giải mã tham lam (beam 1) vì bị thay ngay; final beam 5,
trên audio đã qua tầng 4. **Whisper bịa, và bịa tự tin hơn khi phiên âm thật** —
nên `no_speech_prob` **không được phép** tự nó loại đoạn nào, phải kèm điều kiện
`avg_logprob`, đúng luật của chính Whisper. Câu bịa là việc của
[`server/data/`](server/data/README.md), danh sách sửa được không cần code.

Hai mặc định của faster-whisper bị tắt: `vad_filter` (Silero đã chạy rồi, chạy
lần hai cắt mất pre-roll) và `condition_on_previous_text` (cách một câu bịa
thành cả đoạn bịa).

**8. Translation chạy ngoài luồng audio** — hàng đợi có giới hạn + luồng worker.
Một lần vLLM trả lời chậm từng làm mọi sự kiện VAD tới trễ 12 giây. Câu và bản
dịch là **hai message riêng**, ghép theo `sentence_id`.

vLLM chạy **tiến trình riêng** sau API OpenAI-compatible. Nó giữ trước một phần
GPU lúc nạp; tách ra thì mỗi bên nhìn thấy một GPU lập luận được, và LLM restart
được mà không rớt cuộc họp.

Model được yêu cầu dịch sẽ **cố trò chuyện** — "Sure! Here is the translation:",
ghi chú, ngoặc kép. Câu trả lời bị bóc sạch rồi kiểm tra; trả lời dài gấp nhiều
lần câu gốc không phải bản dịch mà là giải thích, và bị từ chối. Chế độ thinking
tắt: lần chạy đầu tiêu trọn token vào khối `<think>` và không trả về gì.

## 4. Protocol

Một WebSocket mỗi phiên, endpoint `/ws/stream`. Định nghĩa duy nhất ở
`common/protocol.py`, cả hai phía import từ đó nên **không thể lệch nhau** —
lệch định dạng audio làm hỏng tiếng nói âm thầm chứ không ném lỗi.

- **Binary frame:** PCM thô, chỉ client → server. Không header, đúng 6400 byte.
- **Text frame:** JSON, hai chiều.

| Chiều | Message | Nội dung |
| --- | --- | --- |
| → | `hello` | Định dạng audio. Lệch là server từ chối và đóng |
| → | `bye` | Kết thúc phiên |
| ← | `ready` | Bắt tay xong |
| ← | `vad` | `speech_start` / `speech_end` kèm `at_ms` |
| ← | `utterance` | Ranh giới câu, gửi **trước** transcript. Kèm `kept`, `label`, `reason` |
| ← | `partial` | Chữ mờ, thay liên tục |
| ← | `final` | Câu đã chốt, có `sentence_id` |
| ← | `translation` | Bản dịch, ghép theo `sentence_id`. **Luôn tới**, kể cả khi từ chối |
| ← | `speakers` | `{sentence_id: speaker_id}` — nhãn đã sửa |
| ← | `error` | Kèm `fatal` |

Server phục vụ **một phiên tại một thời điểm**: Silero là mô hình hồi quy,
trạng thái ẩn thuộc về đúng một luồng audio. Kết nối thứ hai bị từ chối (1013).

## 5. Kiểm thử

| Nơi | Chạy ở đâu | Cần gì |
| --- | --- | --- |
| `server/tests/`, `client/tests/` | Dev PC | Không GPU, không sound card |
| `server/tests_real/` | GPU pod | Model thật |
| `client/tests_real/` | Windows Client PC | Sound card thật |

Mỗi tầng tách làm **lớp policy thuần Python** (test được ở Dev PC) và **lớp bọc
model mỏng**. Đó là lý do hơn một nghìn test chạy được không cần phần cứng.

## 6. Cấu trúc thư mục

```
client/          thu âm, WebSocket, UI
server/          8 tầng + FastAPI
  data/          danh sách chặn/giữ, sửa được không cần code
  pipeline/      một file mỗi tầng
common/          hợp đồng audio và protocol, dùng chung
docs/            hướng dẫn tinh chỉnh
```
