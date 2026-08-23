# CLAUDE AGENT INSTRUCTIONS & CONSTRAINTS

Bạn là một AI Developer Agent đang phát triển một hệ thống Real-time Translation App. Hãy đọc kỹ và tuân thủ NGHIÊM NGẶT các quy tắc sau:

## 1. Môi trường (Environment Constraints)
- **Ngôn ngữ:** Chỉ sử dụng Python 3.11 cho toàn bộ mã nguồn.
- **Cấu trúc mạng lưới máy tính:**
  - `Dev PC`: Nơi bạn (Claude Code) đang hoạt động, tạo code, chạy các unit test/mock test cơ bản.
  - `Windows Client PC`: Máy tính thật dùng để chạy test Audio Loopback thực tế.
  - `GPU Server (K8s Pod)`: Máy chủ chứa GPU H100 chạy Inference thật. Giao tiếp qua VSCode SSH.

## 2. Quy trình viết code và Kiểm thử (Strict Testing Protocol)
Bạn KHÔNG ĐƯỢC PHÉP tự chạy các bài kiểm thử thực tế (real hardware tests) liên quan đến Audio Device (Loopback) hoặc GPU Inference (CUDA/vLLM) trong Agent Loop của bạn. Hãy tuân thủ quy trình 3 bước sau đối với mọi module:

- **Bước 1 (Coding & Test):** Viết source code cho module. Tự viết và chạy các script unit test để đảm bảo logic code không có lỗi cú pháp.
- **Bước 2 (Prepare Real Test):** BẮT BUỘC phải có những kiểm thử thực tế với dữ liệu thật. 
- **Bước 3 (Instruct & Wait):** Khi đã có real test:
  1. Hướng dẫn chi tiết máy nào (Client hay Server) cần chạy test này.
  2. Hướng dẫn chi tiết từng bước chạy test.
  3. Kết quả kỳ vọng (Expected output).
  4. Chờ user chạy test và gửi lại kết quả cho mình check, OK hết thì mới đi tiếp. 
  -> **TUYỆT ĐỐI CHỜ ĐỢI USER PHẢN HỒI, KHÔNG TỰ ĐI TIẾP.**

## 3. Các Module Bắt Buộc Phải Có Real Test
Mọi tính năng trong `DESIGN.md` đều phải có test thực tế độc lập trước khi ghép nối vào app chính:
- **Client:** WASAPI Loopback capture, PCM 16kHz conversion, Silero VAD detection.
- **Server:** YAMNet classifier, Pedalboard RMS gating, PyAnnote embedding, SpeechBrain LID, Faster-Whisper ASR, vLLM Qwen3.5 translation.
Khi ghép nối các module, cũng cần phải test thực tế để đảm bảo các module ghép nối thành công.

## 4. Quản lý Source Code
- Phân chia thư mục rõ ràng: `/client` và `/server`.
- Luôn tạo `requirements.txt` riêng cho Client và Server.