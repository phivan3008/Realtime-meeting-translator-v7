# GPU Server — chỉ phục vụ model

Pod H100 **không chứa code pipeline**. Nó chạy đúng hai server:

| Cổng | Server | Model |
| --- | --- | --- |
| 8001 | vLLM (OpenAI-compatible) | `google/gemma-4-12b-it` — dịch |
| 8002 | `whisper_server.py` (FastAPI + faster-whisper) | Whisper `large-v3`, float16 — nhận dạng tiếng nói |

Toàn bộ pipeline (VAD, cắt câu, người nói, LID, chính sách streaming ASR, chặn
câu bịa, prompt dịch) chạy ở máy Windows và gọi hai cổng này qua HTTP.

## Vì sao Whisper không chạy bằng vLLM

Pipeline streaming cần, cho mỗi lần giải mã:

- **thời điểm từng từ** — để chốt từ mà hai lần giải mã liên tiếp cùng thấy;
- **`avg_logprob`, `no_speech_prob`, `compression_ratio` từng đoạn** — để chặn
  câu Whisper bịa;
- **beam search 5** khi chốt câu.

Endpoint `/v1/audio/transcriptions` của vLLM giải mã tham lam và không trả thời
điểm từng từ. `whisper_server.py` gọi faster-whisper bằng **đúng lời gọi**
pipeline từng gọi khi Whisper còn chạy chung tiến trình, nên kết quả giống hệt.
File này **không có logic pipeline**: nó nhận audio và tham số, trả về mọi
đoạn và mọi từ kèm điểm, còn quyết định là việc của máy Windows.

## Chép gì lên pod

Chỉ hai file, **không clone cả repo**:

```
gpu_server/whisper_server.py
gpu_server/requirements.lock.txt
```

Kéo thả bằng VSCode Remote-SSH, hoặc `scp`. Đặt vào một thư mục riêng, ví dụ
`~/models-srv/`.

## Cài một lần

```bash
cd ~/models-srv
python3.11 -m venv .venv && source .venv/bin/activate
python --version                         # phải in 3.11.x
python3.11 -m pip install -r requirements.lock.txt
```

Lock này là **tập con đúng phiên bản** của lock pod đã dùng trước đây
(vllm 0.26.0, transformers 5.14.1, torch 2.11.0, faster-whisper 1.2.1). Một
venv cũ còn trên pod đã có đủ — dùng lại được, kiểm bằng
`pip install --dry-run -r requirements.lock.txt` (không có gì cần cài là đúng).

Pod không có internet thì tải sẵn checkpoint ở chỗ khác, chép lên, rồi trỏ
`--model` vào thư mục đó. Cache Whisper cũ nằm ở `models/whisper` của thư mục
repo cũ; dùng lại bằng `--download-root <đường dẫn>`.

## Khởi động

**vLLM trước.** Nó giữ 85% **cả card** lúc nạp và từ chối khởi động nếu không
đủ chỗ; phần còn lại (~12 GB trên H100 80 GB) đủ cho Whisper large-v3 float16.

**Terminal 1 — vLLM** (tmux / screen để không chết khi rớt SSH):

```bash
source ~/models-srv/.venv/bin/activate
python3.11 -m vllm.entrypoints.openai.api_server --model google/gemma-4-12b-it --port 8001 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 --trust-remote-code --host 127.0.0.1
```

Lệnh này sinh từ `server/config.py` — ở máy Windows chạy
`python server\launch_vllm.py --print` để in lại, và một unit test giữ hai
nơi khớp nhau.

Chờ `Application startup complete`, rồi:

```bash
curl -s http://127.0.0.1:8001/v1/models
```

Phải thấy `"id":"google/gemma-4-12b-it"` và `"max_model_len":4096`.

**Terminal 2 — Whisper** (sau khi vLLM sẵn sàng):

```bash
source ~/models-srv/.venv/bin/activate
cd ~/models-srv
python3.11 whisper_server.py --model large-v3 --port 8002
```

Chờ `Whisper ready on cuda: large-v3 (float16)` rồi `Uvicorn running on
http://127.0.0.1:8002`, rồi:

```bash
curl -s http://127.0.0.1:8002/health
```

Phải là `{"status":"ok","api_version":1,"model":"large-v3","device":"cuda","compute_type":"float16","decodes":1}`
(`decodes` là 1 vì lần khởi động đã giải mã thử một giây im lặng).

## Bảo mật

Cả hai server **chỉ nghe ở `127.0.0.1`** và không có xác thực: ai tới được
cổng là bắt được GPU làm việc. Máy Windows nối vào bằng SSH tunnel hoặc tab
Ports của VSCode — xem `README.md` ở gốc repo. Đừng đổi sang `--host 0.0.0.0`
nếu chưa có lớp xác thực phía trước.

## API của `whisper_server.py`

| Đường dẫn | Việc |
| --- | --- |
| `GET /health` | Model, thiết bị, kiểu tính, số lần đã giải mã, `api_version` |
| `POST /v1/decode?language=ja&beam_size=5&prompt=...` | Body là PCM16 LE mono 16 kHz thô. Trả về ngôn ngữ và mọi đoạn kèm từ và điểm |

Tuỳ chọn trong query: `language`, `beam_size` (1–10), `prompt`, `temperature`,
`condition_on_previous_text`, `no_speech_threshold`, `log_prob_threshold`,
`compression_ratio_threshold`. Tên lạ thì bị từ chối 400 — một tuỳ chọn gõ sai
mà bị lờ đi là một lần giải mã khác cái được yêu cầu. Cố định phía server:
`word_timestamps=True`, `vad_filter=False` (Silero đã chạy ở Windows, chạy lần
hai cắt mất pre-roll).

Đổi hình dạng request hoặc response thì tăng `API_VERSION` ở cả
`whisper_server.py` lẫn `server/pipeline/asr_remote.py`; pipeline từ chối một
server nói phiên bản khác.
