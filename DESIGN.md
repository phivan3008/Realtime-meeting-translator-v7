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
3. **Deep Noise Filter:** `YAMNet` (hoặc AST thu gọn). Phân loại âm thanh, drop các chunk là tiếng gõ phím, tiếng ho (không phải Speech).
4. **Overlap Resolver (DSP):** `pedalboard` (Noise Gate & Compressor). Đè bẹp giọng nhỏ, ưu tiên giọng có năng lượng RMS cao hơn khi bị chồng lấn.
5. **Speaker Diarization:** `pyannote.audio` (ECAPA-TDNN). Trích xuất Voiceprint, so khớp Cosine Similarity. Gắn nhãn Speaker (e.g., Speaker_01).
6. **Language ID (LID):** `SpeechBrain` (VoxLingua107). Trả về 'vi' hoặc 'ja' nhanh chóng.
7. **ASR Engine:** `faster-whisper` (large-v3). 
   - *Partial Mode:* Nhận audio stream, ép ngôn ngữ (force_language) theo LID, trả Text dự đoán.
   - *Final Mode:* Chạy khi có Sự kiện chốt, trả Text chính xác tuyệt đối.
8. **Translation Engine:** `Qwen3.5-9B` (via `vLLM`). Chạy Text-to-Text. Nhận Prompt gồm: Lịch sử 2-3 câu hội thoại + LID + Final Transcript. Trả về văn bản dịch.

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
{"type": "utterance", "index": 0, "start_ms": 0.0, "end_ms": 6400.0, "duration_ms": 6400.0, "reason": "pause", "continues_previous": false}
```
Ranh giới câu do Stream Buffer Manager chốt, gửi **trước khi** có transcript, để UI mở sẵn một dòng cho câu đó. `reason` là một trong: `pause` (VAD đóng segment), `max_duration` (nói liên tục quá 7s), `speaker_change` (dành cho tầng diarization, chưa nối), `end_of_stream` (phiên kết thúc khi đang nói).

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