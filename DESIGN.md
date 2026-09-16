# Thiết kế: Phiên dịch cuộc họp VI ↔ JA thời gian thực

Phiên dịch hai chiều Việt–Nhật cho họp online. Âm thanh thu ở máy Windows, xử
lý trên GPU server, trả về phụ đề kèm bản dịch.

Tài liệu này mô tả **các module và ranh giới giữa chúng**. Con số cụ thể và
phép đo đã chọn ra chúng nằm ở [`docs/TUNING.md`](docs/TUNING.md); cách chạy
nằm ở [`README.md`](README.md); sơ đồ và tham số theo code hiện tại nằm ở
[`ARCHITECTURE.md`](ARCHITECTURE.md).

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
- `main.py` — điểm vào.

`client/record.py` ghi cuộc họp ra hai file, thuần Python.

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
| 3 | Noise Filter | AST (AudioSet, CPU) | Bỏ tiếng gõ phím, tiếng ho — **mặc định tắt** |
| 4 | Overlap Resolver | `pedalboard` gate + compressor | Hạ giọng nhỏ khi chồng lấn |
| 5 | Diarization | ECAPA-TDNN (SpeechBrain) | Ai đang nói |
| 5b | Reclustering | agglomerative, cosine | Gom cụm lại cả cuộc họp, **sửa nhãn đã gửi** |
| 6 | Language ID | VoxLingua107 ECAPA | `vi`, `ja`, hoặc rỗng |
| 6b | Language split | LID trên hai đầu câu | Cắt câu chứa **hai** ngôn ngữ |
| 7 | ASR | faster-whisper large-v3, streaming | Chuyển thành chữ, chốt dần từng từ |
| 8 | Translation | gemma-4-12b-it qua vLLM | Dịch, **ngoài luồng audio** |

### Ghi chú từng tầng

**1. VAD ở server, không ở client.** `silero-vad` kéo theo `torch` lên Windows
và `torchaudio` fail ABI ở đó. Ngoài ra Buffer Manager cần biết các khoảng
ngắt, mà đặt VAD cạnh nó thì protocol phụ báo ngắt biến mất.

**2. Buffer Manager.** Ranh giới câu: **ngắt** (VAD đóng đoạn) hoặc **quá dài**
(>7 s). Cắt vì quá dài thì lùi về khung 32 ms yên tĩnh nhất gần đó — Whisper
biến nửa từ thành từ khác. Trong lúc câu chưa chốt, cứ 600 ms xuất một cửa sổ
partial; cửa sổ này bị **giới hạn 4 giây cuối**, vì giải mã lại cả câu mỗi
600 ms tốn hơn toàn bộ các câu đã chốt cộng lại.

Mỗi câu chốt vì ngắt mang theo **độ dài hangover** — khoảng lặng VAD đã chuyển
tiếp trong lúc chờ chắc chắn người nói đã dừng (~480 ms). ASR dùng nó để không
giải mã khoảng lặng đó: Whisper trả lời im lặng bằng chữ.

Không cắt câu theo **đổi giọng**: đã làm và đo ở nhánh khác, voiceprint cửa sổ
1 giây không phân biệt được giọng. Xem `TUNING.md` 5b.

**3. Noise Filter mặc định tắt** (`ENABLE_NOISE_FILTER=1` để bật). Hai cuộc
họp thật: **237 utterance, bỏ được 0 câu**, tốn 1.31 giây mỗi utterance tức
22% luồng đọc socket, và `slowest sentence` 2.1 s thay vì 0.4 s. Lý do tắt là
nó không bỏ được gì, không phải chi phí.

Khi bật, nó **rụt rè có chủ đích.** Chỉ bỏ khi speech score thấp **và**
có lớp non-speech đạt ngưỡng. Hai điểm gần 0 không phải bằng chứng, đó là model
đang không biết. Bỏ nhầm câu thật thì mất luôn; để lọt tiếng ho chỉ tốn một lần
gọi Whisper. Câu bị bỏ vẫn báo về client kèm nhãn (`kept: false`).

**4. Overlap Resolver không tách nguồn.** Gate không tách được hai giọng, nó
chỉ hạ những gì thấp hơn hẳn giọng đang át. Ngưỡng đặt theo **đỉnh** chứ không
theo RMS — detector của pedalboard so theo đỉnh.

Khách hàng duy nhất của tầng này là ASR, và **chưa ai đo nó giúp hay hại**.
`DISABLE_OVERLAP=1` cho Whisper ăn audio thô để so. Xem `TUNING.md` mục 4.

**5. Diarization.** Gọi `speechbrain` trực tiếp, không qua `pyannote.audio`:
wrapper của pyannote truyền tham số mà speechbrain không nhận, lỗi trước cả khi
nạp model. Voiceprint lấy từ **audio thô**, chưa qua tầng 4 — gate cắt cả âm
tiết nhỏ, mà âm tiết nhỏ vẫn mang chất giọng.

**5b. Gom cụm lại.** Tầng 5 phải trả lời **ngay**, nên câu trả lời phụ thuộc
thứ tự cuộc họp và không sửa được. Định kỳ gom cụm **cả cuộc họp** từ đầu và
gửi về những nhãn đổi qua message `speakers`. Hàng khoá theo `sentence_id` ở cả
hai đầu nên đây là cập nhật hàng đã có trên màn hình.

Gom cụm **không vượt** `SPEAKER_MAX_SPEAKERS`, và một nhãn chỉ đổi khi **hai lần
gom liên tiếp** cùng đề xuất — một lần chạy 30 phút không có hai luật này ra 22
người và sửa 329 nhãn trên 313 câu.

**6. Language ID chỉ đọc điểm của đúng hai ngôn ngữ** rồi chuẩn hoá lại giữa
chúng. Model biết 107 thứ tiếng và sẽ trả lời tiếng Hàn cho tiếng Nhật nếu được
tự do. Hai điểm quá gần nhau thì trả rỗng: **ép sai ngôn ngữ không báo lỗi**,
Whisper vẫn trả về văn bản trôi chảy, tự tin và sai.

**6b. Một câu chứa hai ngôn ngữ thì mất một.** LID phải chọn một cho cả câu,
Whisper bị ép theo, và nửa còn lại **không** ra bản dịch sai — nó không ra gì.
Đo được 4 lượt nói mất mỗi 10 phút họp.

Thăm dò **gần** đầu và cuối câu (lùi vào 300 ms, tránh pre-roll và hangover).
Hai đầu cùng ngôn ngữ hoặc chưa chắc thì không cắt. Khác nhau chắc chắn thì tìm
nhị phân ranh giới, bắt vào khung im nhất, rồi **thăm dò lại chính hai nửa sẽ
gửi đi** — hai nửa không khác nhau thì thôi cắt. Không nửa nào ngắn hơn
1.2 giây tiếng nói, vì Whisper lấp mảnh vụn bằng câu bịa. Hai lần thăm dò cho
câu bình thường, khoảng bảy khi phải cắt. Mọi lần từ chối đều được đếm theo lý
do.

**7. ASR streaming.** Mỗi 600 ms giải mã lại 4 giây cuối của câu đang mở
(tham lam, beam 1). Từ nào **hai lần giải mã liên tiếp** cùng thấy ở cùng chỗ,
và cách mép cửa sổ hơn 1 giây, thì **chốt** — không bao giờ bị viết lại. Chữ mờ
là phần đã chốt cộng phần chưa ổn định. Khi câu kết thúc, chỉ **phần sau chữ đã
chốt** được giải mã lại (beam 5, kèm từ vựng mồi), dừng ở chỗ tiếng nói thật sự
hết cộng 200 ms.

- **Văn bản dựng từ chuỗi từ của chính Whisper**, không tự chèn khoảng trắng:
  chèn khoảng trắng giữa các từ làm 88% câu tiếng Nhật của một lần chạy bị cách
  từng chữ.
- Một từ thuộc phần đã chốt hay chưa được xét theo **điểm giữa** của nó, và bản
  sao của từ chốt cuối ở chỗ nối bị bỏ — không thì 45% câu mang một từ lặp đôi.
- Ngôn ngữ của chữ mờ được chốt **một lần mỗi câu**, ở câu trả lời chắc chắn đầu
  tiên của LID: hai lần giải mã bị ép hai ngôn ngữ khác nhau không bao giờ khớp.
  Câu đã có chữ chốt thì giữ ngôn ngữ đó dù LID của cả câu nói khác — câu lệch
  ngôn ngữ với chữ mờ đo được sai gấp 3–4 lần.
- Hai nửa của một câu bị cắt theo ngôn ngữ được giải mã **nguyên nửa**, bằng
  ngôn ngữ phép thăm dò đã tìm cho nửa đó.

**Whisper bịa, và bịa tự tin hơn khi phiên âm thật** — nên `no_speech_prob`
**không được phép** tự nó loại đoạn nào, phải kèm điều kiện `avg_logprob`, đúng
luật của chính Whisper; trừ khi đoạn audio ngắn hơn 600 ms, nơi mọi câu bịa đã
xác nhận xuất hiện. Câu bịa là việc của [`server/data/`](server/data/README.md),
danh sách sửa được không cần code. `vocabulary.txt` ở cùng chỗ là từ vựng mồi
cho Whisper.

Hai mặc định của faster-whisper bị tắt: `vad_filter` (Silero đã chạy rồi, chạy
lần hai cắt mất pre-roll) và `condition_on_previous_text` (cách một câu bịa
thành cả đoạn bịa).

**8. Translation chạy ngoài luồng audio** — hàng đợi có giới hạn + luồng worker.
Một lần vLLM trả lời chậm từng làm mọi sự kiện VAD tới trễ 12 giây. Câu và bản
dịch là **hai message riêng**, ghép theo `sentence_id`.

Mỗi câu mang theo **bản chụp lịch sử lúc nó được chốt**. Session là nơi duy
nhất ghi lịch sử. Đọc lịch sử lúc worker dịch thì nó đã chứa chính câu đang dịch
(và có khi cả câu sau), và mỗi câu nằm trong đó hai lần. Model thấy câu cần dịch
trong phần "các câu trước" thì trả lại nguyên văn. Câu bị trả lại nguyên văn
được **hỏi lại một lần không kèm lịch sử**.

Gemma không có role `system`: toàn bộ chỉ dẫn nằm trong một message `user`.

vLLM chạy **tiến trình riêng** sau API OpenAI-compatible. Nó giữ trước một phần
GPU lúc nạp; tách ra thì mỗi bên nhìn thấy một GPU lập luận được, và LLM restart
được mà không rớt cuộc họp.

Model được yêu cầu dịch sẽ **cố trò chuyện** — "Sure! Here is the translation:",
ghi chú, ngoặc kép. Câu trả lời bị bóc sạch rồi kiểm tra; trả lời dài gấp nhiều
lần câu gốc không phải bản dịch mà là giải thích, và bị từ chối.

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

`server/analysis/` đọc nhật ký gỡ lỗi của client và so các lần chạy của **cùng
một cuộc họp** — kiểm tra là cùng cuộc họp trước khi so bất cứ số nào.

## 6. Cấu trúc thư mục

```
client/          thu âm, WebSocket, UI
server/          8 tầng + FastAPI
  analysis/      đọc và so nhật ký gỡ lỗi, không cần GPU
  data/          danh sách chặn/giữ và từ vựng, sửa được không cần code
  pipeline/      một file mỗi tầng
common/          hợp đồng audio và protocol, dùng chung
docs/            hướng dẫn tinh chỉnh
```
