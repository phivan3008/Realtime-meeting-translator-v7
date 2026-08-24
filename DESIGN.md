# SYSTEM ARCHITECTURE DESIGN: Real-time VI-JA Meeting Translator

## 1. Overview
Hệ thống phiên dịch thời gian thực hai chiều Việt Nam (VI) - Nhật Bản (JA) dành cho môi trường họp online.
- **Môi trường phát triển & ngôn ngữ:** Toàn bộ sử dụng Python 3.11.
- **Topology:**
  - **Dev PC:** Nơi viết code, commit/push lên GitHub, chạy các mock test cơ bản.
  - **Windows Client PC:** Pull code từ GitHub, chạy Client App, capture âm thanh thực tế.
  - **GPU Server (Pod H100 80GB):** Chạy Server App, kết nối trực tiếp từ Dev PC/Client PC qua VSCode -> Kubernetes -> Attach VSCode SSH Server.

## 2. Client Architecture (Windows App)
Chịu trách nhiệm thu thập âm thanh, truyền lên Server và giao tiếp UI. Không xử lý ML.
- **Audio Capture:** `pyaudiowpatch` qua WASAPI Loopback (16kHz, 16-bit PCM, Mono). Bắt toàn bộ âm thanh cuộc họp (Zoom, Teams...).
- **Streaming:** Đóng gói audio thành các chunk 200ms (6400 bytes), gửi liên tục lên Server qua WebSocket. Client KHÔNG lọc gì cả, gửi toàn bộ audio kể cả khoảng lặng (~256 kbps).
- **UI Render:** Lắng nghe JSON từ Server, hiển thị 2 trạng thái: Partial (chữ mờ - đang dự đoán) và Final (chữ đậm - chốt câu + bản dịch).

## 3. Server Architecture (GPU Pod)
Đường ống (Pipeline) xử lý khép kín tuần tự:
1. **VAD (Voice Activity Detection):** `Silero VAD` (CPU, 512 sample/frame @ 16kHz). Cắt stream thành các segment speech có timestamp, drop khoảng lặng trước khi vào các tầng nặng. Giữ pre-roll 256ms để không cụt âm đầu từ.
2. **Stream Buffer Manager:** Gom audio speech từ VAD thành các câu (utterance). Kích hoạt "Sự kiện chốt" (Finalize Event) khi: Pause > 400ms, Overlap (đổi người nói — hook đã có, chờ tầng diarization), hoặc Max duration (>7s). Cắt vì quá dài thì lùi về khung 32ms yên tĩnh nhất trong 500ms gần nhất để không cắt giữa từ. Trong lúc câu chưa chốt, cứ 600ms lại xuất một cửa sổ partial cho ASR dự đoán.
3. **Deep Noise Filter:** `YAMNet` (chạy CPU, không chiếm VRAM của Whisper/vLLM). Phân loại từng utterance đã chốt, drop cái là tiếng gõ phím, tiếng ho (không phải Speech). Chính sách **rụt rè có chủ đích**: chỉ drop khi speech score < 0.2 **VÀ** có lớp non-speech đạt ít nhất 0.3 — hai điểm số gần bằng 0 không phải bằng chứng, đó là model đang không biết gì, mà không biết thì giữ lại — bỏ nhầm một câu thật thì mất luôn, còn để lọt một tiếng ho chỉ tốn một lần gọi Whisper. Utterance bị drop vẫn được báo về client kèm nhãn (`kept: false`), không xoá âm thầm.
4. **Overlap Resolver (DSP):** `pedalboard` (Noise Gate & Compressor). Đè bẹp giọng nhỏ, ưu tiên giọng mạnh hơn khi bị chồng lấn. **Đây không phải tách nguồn** — gate không tách được hai giọng; nó chỉ hạ những gì nằm thấp hơn hẳn giọng đang át. Ngưỡng đặt tương đối theo **percentile 90 của đường bao đỉnh** của chính utterance đó: RMS toàn cục bị kéo tụt bởi khoảng lặng hangover, còn detector của gate so theo đỉnh chứ không theo RMS. Đo thực tế với giọng phụ thấp hơn 20 dB: ngưỡng theo RMS chỉ hạ được 0.1 dB, ngưỡng theo đỉnh hạ 24 dB mà giọng chính không suy hao.
5. **Speaker Diarization:** ECAPA-TDNN, checkpoint `speechbrain/spkrec-ecapa-voxceleb`, gọi trực tiếp qua `speechbrain` chứ không qua `pyannote.audio` — wrapper của pyannote 4.0.7 truyền `token`/`huggingface_cache_dir`/`revision` xuống speechbrain 1.1.0 vốn không nhận, nên lỗi trước cả khi nạp model. Cùng bộ trọng số, ít hơn một tầng. Trích voiceprint cho từng câu từ **audio thô, chưa qua Overlap Resolver** — đo trên hai bản ghi một người, gate trước khi embed làm mất 0.06 cosine cùng-giọng (0.677 thô so với 0.616 đã gate) vì gate cắt cả âm tiết nhỏ trong câu, mà âm tiết nhỏ vẫn mang chất giọng. Overlap Resolver phục vụ tầng ASR, không phục vụ tầng này. so khớp Cosine Similarity với các giọng đã nghe, gắn nhãn `Speaker_01`, `Speaker_02`... Đây là **nhận dạng trực tuyến, không phải diarization ngoại tuyến**: mỗi câu phải được gán nhãn ngay khi Buffer Manager chốt, nên thuật toán là tham lam và phụ thuộc thứ tự — một giọng đến muộn không thể sửa ngược một nhầm lẫn trước đó. Câu ngắn dưới 600ms gắn `Speaker_unknown` chứ **không đoán theo người nói trước**: câu chêm ngắn ("à ra vậy") thường là của người đang *nghe*, nên phép đoán đó sai đúng vào chỗ nó hấp dẫn nhất.
6. **Language ID (LID):** `SpeechBrain` (VoxLingua107 ECAPA). Trả về `'vi'`, `'ja'`, hoặc **rỗng**. Model biết 107 ngôn ngữ nhưng chỉ đọc điểm của đúng hai ngôn ngữ cuộc họp có thể chứa — để tự do chọn thì tiếng Nhật hay bị trả về là Hàn hoặc Trung, một nhầm lẫn hợp lý với model nhưng vô dụng với ta, vì việc duy nhất tầng sau làm với kết quả này là **ép `language` của Whisper**. Khi hai điểm số quá gần nhau (margin < 0.30) thì trả rỗng để Whisper tự nhận diện: **ép sai ngôn ngữ không báo lỗi** — Whisper vẫn trả về văn bản trôi chảy, tự tin, và sai, rồi tầng dịch dịch trung thành cái vô nghĩa đó.
7. **ASR Engine:** `faster-whisper` (large-v3).
   - *Partial Mode:* Chạy trên cửa sổ partial (mỗi 600ms), giải mã tham lam (beam=1) vì partial bị thay thế ngay sau đó. Ép `language` theo LID; LID trả rỗng thì để Whisper tự nhận diện.
   - *Final Mode:* Chạy khi có Sự kiện chốt, beam search (beam=5), trên audio **đã qua Overlap Resolver**.
   - **Whisper bịa văn bản, và bịa một cách tự tin.** Gặp gần-im-lặng nó không trả về rỗng mà trả về một câu trôi chảy chưa ai nói ("Thank you for watching"), đôi khi lặp một cụm để lấp thời gian. Ba lớp chắn: `no_speech_prob`, `avg_logprob`, và `compression_ratio` (gzip của tiếng nói tự nhiên rơi vào 1.5–2.0; nén tốt hơn hẳn nghĩa là đang lặp). Đoạn bị loại được **đếm và ghi log**, không xoá âm thầm.
   - Hai mặc định của faster-whisper bị tắt: `vad_filter` (Silero đã chạy ở đầu pipeline, chạy VAD lần hai sẽ cắt mất pre-roll giữ phụ âm đầu) và `condition_on_previous_text` (nó đưa câu trước làm prompt cho câu sau — đúng cơ chế biến một câu bịa thành cả đoạn bịa).
8. **Translation Engine:** Qwen qua `vLLM`, chạy **tiến trình riêng** sau API OpenAI-compatible, không import vào tiến trình audio — vLLM giữ trước một phần GPU lúc nạp, để chung sẽ phải tự tay cân với chỗ Whisper chiếm; tách ra thì mỗi bên nhìn thấy một GPU có thể lập luận được, và LLM restart được mà không rớt cuộc họp. Tên model nằm trong lệnh khởi động vLLM, code chỉ hỏi server đang phục vụ model nào.
   - Prompt gồm: 3 câu hội thoại gần nhất (chỉ để tham chiếu, **không dịch**) + LID + Final Transcript.
   - **Model sẽ cố trò chuyện với bạn.** Model instruction-tuned được yêu cầu dịch sẽ trả về "Sure! Here is the translation:" rồi mới tới bản dịch, hoặc thêm ghi chú, hoặc bọc trong ngoặc kép — tất cả sẽ hiện lên màn hình như thể có người vừa nói ra. Nên câu trả lời được **bóc sạch** phần thừa rồi **kiểm tra**: câu trả lời dài gấp nhiều lần câu gốc không phải bản dịch, đó là model đang giải thích, và bị từ chối.
   - `temperature = 0`: cùng một câu phải cho cùng một bản dịch.

## 3b. Ghi chú kiến trúc: vì sao VAD nằm ở Server

Thiết kế ban đầu đặt VAD ở Client để tiết kiệm băng thông. Đã chuyển lên Server vì:
- `silero-vad` import `torchaudio` vô điều kiện, kéo theo `torch` lên máy Windows Client. Trên máy client thực tế, `torchaudio` fail với `OSError: [WinError 127]` khi nạp `_torchaudio.pyd` (lệch ABI với torch) và không sửa được.
- Buffer Manager ở Server cần biết pause > 400ms để chốt câu. VAD ở Client buộc phải phát thêm event `speech_start`/`speech_end` báo cho Server biết những pause đã bị xóa. Đặt VAD cạnh Buffer Manager thì protocol phụ đó biến mất.
- Client không còn dependency ML nào — chỉ `PyAudioWPatch`, `numpy`, `soxr`, `websockets`, `PySide6`.

Giá phải trả: client stream liên tục 16kHz mono 16-bit = 32 KB/s = 256 kbps, kể cả khi không ai nói.

## 4. WebSocket JSON Protocol

Một kết nối WebSocket cho mỗi phiên họp. Endpoint: `ws://<host>:<port>/ws/stream`.
Định nghĩa duy nhất nằm ở `common/protocol.py` — cả client và server đều import từ đó,
để định dạng audio không bao giờ lệch nhau giữa hai bên.

- **Text frame:** JSON điều khiển, cả hai chiều.
- **Binary frame:** PCM thô, chỉ client -> server. Không header, không framing:
  mỗi binary frame đúng 6400 bytes (200ms, 16kHz mono 16-bit LE). Chunk N+1 nối
  tiếp đúng chỗ chunk N dừng.

**Bắt tay:**
1. Client gửi `hello` (JSON) trước mọi audio.
2. Server kiểm tra định dạng audio, trả `ready`, hoặc trả `error` rồi đóng kết nối.
3. Client stream binary chunk cho đến khi gửi `bye` hoặc rớt mạng.
4. Server đẩy `vad` / `partial` / `final` bất kỳ lúc nào sau `ready`.

Server chỉ phục vụ **một phiên tại một thời điểm**: Silero là mô hình hồi quy,
trạng thái ẩn thuộc về đúng một luồng audio. Kết nối thứ hai bị từ chối (code 1013)
chứ không phục vụ kém cho cả hai.

**Hello (Client -> Server):**
```json
{
  "type": "hello",
  "session_id": "a1b2c3d4e5f6",
  "protocol_version": 1,
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2,
  "chunk_ms": 200,
  "client": "windows-client"
}
```
Nếu bất kỳ trường audio nào lệch với server, server trả `error` và đóng. Đây là
chốt chặn quan trọng: audio sai định dạng không làm crash gì cả, nó chỉ âm thầm
khiến mọi bản dịch sai.

**Ready (Client <- Server):**
```json
{"type": "ready", "session_id": "a1b2c3d4e5f6", "protocol_version": 1, "sample_rate": 16000, "chunk_bytes": 6400}
```

**VAD event (Client <- Server):**
```json
{"type": "vad", "event": "speech_start", "at_ms": 1234.6}
```
`at_ms` tính từ mẫu đầu tiên server nhận được. `speech_start` trỏ vào mẫu đầu tiên
được chuyển tiếp (đã tính cả pre-roll), `speech_end` trỏ ngay sau mẫu cuối — nên
một cặp start/end cắt đúng khớp đoạn audio đưa xuống tầng dưới.

**Utterance (Client <- Server):**
```json
{"type": "utterance", "index": 0, "start_ms": 0.0, "end_ms": 6400.0, "duration_ms": 6400.0, "reason": "pause", "continues_previous": false, "kept": true, "label": "", "speech_score": 0.87}
```
Ranh giới câu do Stream Buffer Manager chốt, gửi **trước khi** có transcript, để UI mở sẵn một dòng cho câu đó. `reason` là một trong: `pause` (VAD đóng segment), `max_duration` (nói liên tục quá 7s), `speaker_change` (dành cho tầng diarization, chưa nối), `end_of_stream` (phiên kết thúc khi đang nói).

`kept: false` là phán quyết của Deep Noise Filter — câu đó nghe như `label` (ví dụ `Computer keyboard`) chứ không phải tiếng nói, và sẽ không được đưa xuống ASR. Vẫn gửi về client để index liên tục và để thấy ngay nếu bộ lọc bắt đầu ăn nhầm tiếng nói thật.

`continues_previous: true` nghĩa là câu này là phần tiếp của câu trước bị cắt vì quá dài — tầng dịch cần biết để không coi nó là một câu độc lập.

**Error (Client <- Server):**
```json
{"type": "error", "message": "unsupported audio format: sample_rate=48000 (server wants 16000)", "fatal": true}
```

**Partial Message (Client <- Server):**
```json
{
  "type": "partial",
  "speaker_id": "Speaker_01",
  "lang_code": "vi",
  "transcript": "hôm nay chúng ta họp về"
}
**Partial Message (Client <- Server):**
```json
{
  "type": "final",
  "speaker_id": "Speaker_01",
  "lang_code": "vi",
  "transcript": "Hôm nay chúng ta họp về tiến độ dự án.",
  "translation": "今日、私たちはプロジェクトの進捗について会議をします。"
}