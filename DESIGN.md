# SYSTEM ARCHITECTURE DESIGN: Real-time VI-JA Meeting Translator

## 1. Overview
Hệ thống phiên dịch thời gian thực hai chiều Việt Nam (VI) - Nhật Bản (JA) dành cho môi trường họp online.
- **Môi trường phát triển & ngôn ngữ:** Toàn bộ sử dụng Python 3.11.
- **Topology:**
  - **Dev PC:** Nơi viết code, commit/push lên GitHub, chạy các mock test cơ bản.
  - **Windows Client PC:** Pull code từ GitHub, chạy Client App, capture âm thanh thực tế.
  - **GPU Server (Pod H100 80GB):** Chạy Server App, kết nối trực tiếp từ Dev PC/Client PC qua VSCode -> Kubernetes -> Attach VSCode SSH Server.

## 2. Client Architecture (Windows App)
Chịu trách nhiệm thu thập âm thanh, lọc rác cơ bản và giao tiếp UI.
- **Audio Capture:** `pyaudiowpatch` qua WASAPI Loopback (16kHz, 16-bit PCM, Mono). Bắt toàn bộ âm thanh cuộc họp (Zoom, Teams...).
- **VAD (Voice Activity Detection):** `Silero VAD` (CPU). Drop các khoảng lặng tĩnh để tiết kiệm băng thông mạng.
- **Streaming:** Đóng gói audio thành các chunk ~200-500ms, gửi liên tục lên Server qua WebSocket.
- **UI Render:** Lắng nghe JSON từ Server, hiển thị 2 trạng thái: Partial (chữ mờ - đang dự đoán) và Final (chữ đậm - chốt câu + bản dịch).

## 3. Server Architecture (GPU Pod)
Đường ống (Pipeline) xử lý khép kín tuần tự:
1. **Stream Buffer Manager:** `asyncio` buffer. Gom chunk audio thành cửa sổ trượt (Sliding Window). Kích hoạt "Sự kiện chốt" (Finalize Event) khi: Pause > 400ms, Overlap (đổi người nói), hoặc Max duration (>7s).
2. **Deep Noise Filter:** `YAMNet` (hoặc AST thu gọn). Phân loại âm thanh, drop các chunk là tiếng gõ phím, tiếng ho (không phải Speech).
3. **Overlap Resolver (DSP):** `pedalboard` (Noise Gate & Compressor). Đè bẹp giọng nhỏ, ưu tiên giọng có năng lượng RMS cao hơn khi bị chồng lấn.
4. **Speaker Diarization:** `pyannote.audio` (ECAPA-TDNN). Trích xuất Voiceprint, so khớp Cosine Similarity. Gắn nhãn Speaker (e.g., Speaker_01).
5. **Language ID (LID):** `SpeechBrain` (VoxLingua107). Trả về 'vi' hoặc 'ja' nhanh chóng.
6. **ASR Engine:** `faster-whisper` (large-v3). 
   - *Partial Mode:* Nhận audio stream, ép ngôn ngữ (force_language) theo LID, trả Text dự đoán.
   - *Final Mode:* Chạy khi có Sự kiện chốt, trả Text chính xác tuyệt đối.
7. **Translation Engine:** `Qwen3.5-9B` (via `vLLM`). Chạy Text-to-Text. Nhận Prompt gồm: Lịch sử 2-3 câu hội thoại + LID + Final Transcript. Trả về văn bản dịch.

## 4. WebSocket JSON Protocol
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