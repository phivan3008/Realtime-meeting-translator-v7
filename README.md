# Realtime Meeting Translator (VI ↔ JA)

Phiên dịch hai chiều Việt–Nhật theo thời gian thực cho họp online. Âm thanh
cuộc họp được thu **và xử lý** ở máy Windows; GPU server chỉ phục vụ hai model
lớn (Whisper và Gemma 4) qua HTTP. Kết quả là phụ đề kèm bản dịch.

- Thiết kế: [`DESIGN.md`](DESIGN.md)
- Kiến trúc theo code hiện tại, kèm sơ đồ: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Mọi thông số và phép đo đứng sau chúng: [`docs/TUNING.md`](docs/TUNING.md)
- Pod chỉ phục vụ model: [`gpu_server/README.md`](gpu_server/README.md)
- Test thật trên Windows: [`server/tests_real/README.md`](server/tests_real/README.md)

---

## 1. Kiến trúc

Từ 2026-11-12 GPU pod **chỉ được chứa model và chạy server model**. Không có
code pipeline nào trên pod. Toàn bộ pipeline chạy ở máy Windows (chỉ có CPU)
và gọi hai model trên pod qua HTTP.

| Máy | Vai trò |
| --- | --- |
| **Dev PC** | Viết code, chạy unit test, commit/push |
| **Windows Client PC** (chỉ CPU, không có GPU NVIDIA) | Thu âm qua WASAPI loopback, **chạy toàn bộ pipeline** (`server/`, ở `127.0.0.1:8000`), hiển thị |
| **GPU Server** (pod H100) | **Chỉ phục vụ model**: vLLM cho Gemma 4 và `gpu_server/whisper_server.py` cho Whisper. Không chứa code pipeline |

Bốn cổng, tất cả chỉ nghe ở `127.0.0.1` của máy mình:

| Cổng | Ở đâu | Tiến trình | Ai gọi |
| --- | --- | --- | --- |
| 8000 | Windows | Pipeline (`uvicorn server.app:app`), WebSocket `/ws/stream` | Client (cửa sổ phiên dịch) |
| 8001 | Pod → Windows qua tunnel | vLLM, `google/gemma-4-12b-it` | Tầng dịch của pipeline |
| 8002 | Pod → Windows qua tunnel | `whisper_server.py`, Whisper `large-v3` float16 | Tầng ASR của pipeline |

Client và pipeline vẫn là **hai tiến trình riêng** trên cùng máy Windows,
nói chuyện bằng WebSocket như trước. Chỉ Whisper và Gemma đi qua mạng.

### Vì sao Whisper không chạy bằng vLLM

Chính sách streaming ASR cần, cho mỗi lần giải mã: thời điểm từng từ (để chốt
từ hai lần giải mã cùng thấy), `avg_logprob` / `no_speech_prob` /
`compression_ratio` từng đoạn (để chặn câu Whisper bịa), và beam search 5 khi
chốt câu. Endpoint `/v1/audio/transcriptions` của vLLM không có thứ nào trong
ba thứ đó. `gpu_server/whisper_server.py` là một server nhỏ (FastAPI +
faster-whisper) gọi **đúng lời gọi** pipeline từng gọi khi Whisper còn chạy
chung tiến trình. Nó **không có logic pipeline** — nhận audio và tham số, trả
về mọi đoạn và mọi từ kèm điểm; quyết định nằm ở máy Windows
(`server/pipeline/asr_remote.py`).

Đã kiểm chứng: replay 3 phút của cuộc họp chuẩn trên Windows (Whisper qua
HTTP) cho log **giống hệt từng byte** lần replay trên pod ngày 2026-09-17
(Whisper trong tiến trình), trừ cột giờ đồng hồ — 255/255 từ trùng.

### Đường đi của một câu nói

```
Client (Windows)          Pipeline (cùng máy Windows, CPU)        GPU pod
──────                    ────────────────────────────────        ───────
WASAPI loopback           1. VAD (Silero)            cắt thành đoạn có tiếng nói
16 kHz mono 16-bit        2. Buffer Manager          gom thành câu, chốt khi ngắt
chunk 200 ms ─────────►  6b. Language split         cắt câu chứa hai ngôn ngữ
   (ws://127.0.0.1:8000)  3. Noise Filter (AST)      mặc định tắt
                          4. Overlap Resolver        hạ giọng chồng lấn
                          5. Diarization (ECAPA)     ai đang nói
                          6. Language ID             tiếng Việt hay tiếng Nhật
                          7. ASR (Whisper large-v3)  streaming ──HTTP──► whisper_server.py :8002
      ◄──── partial ───      (mỗi 600 ms)
      ◄──── final ─────   8. Translation (Gemma)      ───────HTTP──► vLLM :8001
                         5b. Gom cụm lại            chỉ đo, không gửi (mặc định)
      ◄─ translation ──
```

Câu và bản dịch là **hai message riêng**: câu gửi đi ngay khi Whisper chốt,
bản dịch theo sau và được ghép bằng `sentence_id`. Một lần gọi LLM chậm không
được phép giữ đường vào của âm thanh.

---

## 2. Cài đặt lần đầu

Python **3.11** ở cả ba máy. Không phải 3.12. Máy Windows và pod **pull code
từ GitHub** (Windows) hoặc nhận đúng hai file (pod), nên mọi thay đổi phải được
push trước khi chạy thử.

### 2.1 GPU pod — chỉ hai file

Chi tiết và kết quả kỳ vọng từng lệnh: [`gpu_server/README.md`](gpu_server/README.md).
Tóm tắt:

1. Chép **đúng hai file** vào một thư mục riêng trên pod, ví dụ `~/models-srv/`
   (kéo thả bằng VSCode Remote-SSH hoặc `scp`). **Không clone repo.**
   ```
   gpu_server/whisper_server.py
   gpu_server/requirements.lock.txt
   ```
2. Tạo venv và cài:
   ```bash
   cd ~/models-srv
   python3.11 -m venv .venv && source .venv/bin/activate
   python --version                          # phải in 3.11.x
   python3.11 -m pip install -r requirements.lock.txt
   ```
   Lock này là tập con đúng phiên bản của lock pod cũ (vllm 0.26.0,
   faster-whisper 1.2.1, torch 2.11.0) — venv cũ trên pod dùng lại được.

### 2.2 Máy Windows — client và pipeline

```powershell
git clone <repo>
cd Realtime-meeting-translator-v7
git checkout windows-pipeline

py -3.11 -m venv .venv                     # py -3.11 CHỈ dùng ở dòng này
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.version.split()[0], sys.prefix)"
```

Phải in `3.11.x` và một đường dẫn **kết thúc bằng `.venv`**. Rồi cài:

```powershell
python -m pip install -r server\requirements.lock.txt
python -m pip install -r client\requirements.lock.txt
python -c "import torch, torchaudio, speechbrain; print(torch.__version__, torchaudio.__version__, speechbrain.__version__)"
```

Kỳ vọng: `2.11.0+cpu 2.11.0+cpu 1.1.2`.

> **Sau khi bật venv, luôn gõ `python`, không gõ `py -3.11`.** Launcher `py`
> khi được chỉ rõ phiên bản sẽ **bỏ qua venv đang bật** và chạy Python hệ
> thống. Lệnh vẫn chạy, chỉ là chạy bằng bộ thư viện khác — ngày 2026-11-12
> một lần replay chạy như vậy đã nạp SpeechBrain cũ, mất tầng LID, và cho ra
> câu tiếng Nga, Trung, Ý. Không có lỗi nào báo ra.
>
> Đừng cài vào Python hệ thống. Nó hạ cấp numpy và protobuf bên dưới mọi thứ
> khác đang dùng chung.

`server\requirements.lock.txt` là pipeline chạy CPU: torch CPU, Silero,
SpeechBrain 1.1.2 (ECAPA, VoxLingua107), pedalboard, transformers (AST),
truststore. **Không** có vLLM hay faster-whisper — hai model đó ở pod. Hai
lock dùng chung phiên bản cho mọi gói trùng nhau, nên cài chung một venv được.

**Lần chạy đầu tiên tải model từ huggingface.co** (cần internet):

| Model | Cỡ | Lưu ở |
| --- | --- | --- |
| ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) | ~85 MB | `models\speaker\` |
| VoxLingua107 (`speechbrain/lang-id-voxlingua107-ecapa`) | ~85 MB | `models\lid\` |
| Silero VAD | đi kèm gói `silero-vad` | — |
| AST (chỉ khi `ENABLE_NOISE_FILTER=1`) | ~350 MB | cache Hugging Face |

Hai việc đã được xử lý sẵn trong code, ghi ra đây để khỏi sửa lại:

- **Mạng công ty soi HTTPS.** huggingface.co trình ra chứng chỉ do gốc của
  công ty ký; Windows tin gốc đó, bộ `certifi` của Python thì không, nên mọi
  lần tải báo `CERTIFICATE_VERIFY_FAILED`. Package `server` gọi
  `truststore.inject_into_ssl()` lúc import (`server/tls.py`) để Python kiểm
  chứng chỉ bằng kho của Windows như trình duyệt. **Vẫn kiểm chứng chỉ**, chỉ
  đổi nguồn tin cậy. Đừng tắt kiểm tra SSL.
- **Windows không cho tạo symlink** nếu không có quyền admin hoặc Developer
  Mode (`WinError 1314`). Hai loader SpeechBrain chép file ra khỏi cache
  (`LocalStrategy.COPY`) thay vì symlink. Cảnh báo `huggingface_hub ... symlinks`
  vẫn hiện và vô hại; tắt bằng `$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"`.

**File ghi âm để test** (`recordings\`, không có trong git): `meeting_speech.wav`,
`quiet_room.wav`, `meeting_30min.wav`, `alice.wav`, `bob.wav`, `keyboard.wav`,
`cough.wav`, `input_vi.wav`, `input_ja.wav`. Nếu chúng còn ở repo cũ trên pod,
chép về `recordings\` của máy Windows. Mọi WAV phải là 16 kHz mono 16-bit:
`ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 out.wav`.

### 2.3 Kiểm tra sau khi cài (một lần)

Theo đúng thứ tự; bước sau dựa vào bước trước. Chi tiết từng script ở
[`server/tests_real/README.md`](server/tests_real/README.md). Kết quả dưới đây
là lần đo ngày 2026-11-12.

| # | Lệnh (Windows, trong venv) | Cần pod | Kết quả kỳ vọng |
| --- | --- | --- | --- |
| 1 | `python server\tests_real\test_real_remote_models.py --wav recordings\meeting_speech.wav` | có | `RESULT: PASS (11 checks)`; partial p95 < 600 ms (đo được 287 ms) |
| 2 | `python server\tests_real\test_real_vad.py --speech recordings\meeting_speech.wav --silence recordings\quiet_room.wav` | không | `RESULT: PASS (20 checks)` |
| 3 | `python server\tests_real\test_real_diarization.py --voice recordings\alice.wav --voice recordings\bob.wav --device cpu` | không | `RESULT: PASS`; khác người ≤ 0.2, cùng người ≥ 0.39 |
| 4 | `python server\tests_real\test_real_lid.py --vi recordings\input_vi.wav --ja recordings\input_ja.wav --device cpu` | không | `RESULT: PASS`; 13/13 câu đúng ngôn ngữ |
| 5 | `python server\tests_real\test_real_streaming.py --wav recordings\meeting_30min.wav --limit-seconds 180 --baseline recordings\meeting-20260911-144023.debug.txt --baseline recordings\meeting-20260910-144120.debug.txt` | có | Mọi check chất lượng PASS; xem mục 4 về check CPU |
| 6 | `python -m server.analysis.compare_logs server\tests_real\output\streaming.debug.txt recordings\replay-0917c.debug.txt` | không | Mọi chỉ số trùng `replay-0917c` |

`test_real_noise.py` (AST) phân loại đúng nhưng trên CPU tốn 0.42 thời lượng
audio — đó là lý do AST **tắt** trên máy này.

---

## 3. Chạy một cuộc họp

> **Không cần đặt biến môi trường nào** nếu hai cổng của pod được nối về
> `127.0.0.1:8001` và `127.0.0.1:8002`. Nếu ai đó bảo bạn đặt một biến, đó là
> để **thử nghiệm** — xem [Biến môi trường](#6-biến-môi-trường), và nhớ mở
> terminal mới khi chạy lại bình thường.
>
> Khởi động in ra `No environment overrides; running on the defaults`. Nếu
> thay vào đó là `Environment overrides in effect: {...}` thì có biến còn sót
> từ terminal trước, và nó đang đổi hành vi.

### Bước 1 — pod: hai server model

Dùng `tmux` / `screen` để chúng không chết khi rớt SSH. **vLLM trước**: nó giữ
85% cả card lúc nạp và từ chối khởi động nếu không đủ chỗ.

```bash
# Terminal pod 1
source ~/models-srv/.venv/bin/activate
python3.11 -m vllm.entrypoints.openai.api_server --model google/gemma-4-12b-it --port 8001 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 --trust-remote-code --host 127.0.0.1
```

Chờ `Application startup complete`, rồi:

```bash
# Terminal pod 2
source ~/models-srv/.venv/bin/activate
cd ~/models-srv
python3.11 whisper_server.py --model large-v3 --port 8002
```

Chờ `Whisper ready on cuda: large-v3 (float16)`. Kiểm tra trên pod:

```bash
curl -s http://127.0.0.1:8001/v1/models    # có "id":"google/gemma-4-12b-it" và "max_model_len":4096
curl -s http://127.0.0.1:8002/health       # {"status":"ok","api_version":1,"model":"large-v3","device":"cuda","compute_type":"float16","decodes":1}
```

### Bước 2 — Windows terminal 1: nối hai cổng của pod về máy này

```powershell
ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 <user>@<pod-ssh-host>
```

Để cửa sổ này chạy suốt cuộc họp. Hoặc dùng tab **Ports** của VSCode
Remote-SSH: forward cổng 8001 và 8002. Kiểm tra từ Windows:

```powershell
curl.exe -s http://127.0.0.1:8001/v1/models
curl.exe -s http://127.0.0.1:8002/health
```

Phải ra đúng hai dòng như ở bước 1. Lần đầu trong ngày, chạy thêm
`python server\tests_real\test_real_remote_models.py` (mục 2.3, dòng 1).

### Bước 3 — Windows terminal 2: pipeline

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Chờ `Application startup complete`, rồi kiểm tra:

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

**Kiểm `in_venv` trước tiên.** Nếu là `false`, pipeline đang chạy bằng
interpreter khác — nó vẫn khởi động, vẫn nạp gần hết pipeline, vẫn phục vụ
cuộc họp, chỉ thiếu đúng những gói môi trường đó tình cờ không có. Mọi phép đo
lấy từ một lần chạy như vậy đều không dùng được.

Rồi kiểm **từng cờ**:

| Cờ | Phải là |
| --- | --- |
| `in_venv` | `true` |
| `overrides` | `{}` |
| `vad_loaded` | `true` |
| `noise_filter_loaded` | `false` — mặc định tắt; trên CPU Windows AST tốn 0.42 thời lượng audio |
| `overlap_resolver_loaded` | `true` |
| `speaker_model_loaded` | `true` |
| `language_model_loaded` | `true` |
| `asr_loaded` | `true`, và `asr_source` là `whisper large-v3 at http://127.0.0.1:8002 on cuda (float16)` |
| `translation_loaded` | `true`, và `translation_source` chỉ về `http://127.0.0.1:8001/v1` |

Cái nào sai thì dòng `*_error` bên cạnh nói lý do. Pipeline vẫn khởi động khi
thiếu một tầng — thiếu một tầng còn hơn từ chối cuộc họp — nhưng lúc đó phụ đề
không đáng tin, và **không có gì trong lúc chạy nói cho bạn biết**.

> **Pod chưa chạy hoặc tunnel chưa mở lúc khởi động** thì `asr_loaded` và
> `translation_loaded` là `false`, và chúng **không tự nối lại**. Mở tunnel,
> rồi khởi động lại pipeline.

### Bước 4 — Windows terminal 3: cửa sổ phiên dịch

```powershell
.\.venv\Scripts\Activate.ps1
python -m client.ui.main --url ws://127.0.0.1:8000
```

Bấm **Bắt đầu**, rồi phát âm thanh cuộc họp. Chữ mờ nghiêng là dự đoán đang
chạy — phần đầu của nó đã được chốt và sẽ không đổi; chữ đậm là câu đã chốt,
bản dịch màu xanh ngay dưới. Câu nào không dịch được sẽ nói rõ lý do và nguyên
văn model đã trả lời, thay vì để trống.

Đóng cửa sổ sẽ gửi `bye` và **chờ pipeline dịch xong câu cuối**, nên đừng tắt
bằng cách kill tiến trình. Pipeline nhận **một cuộc họp mỗi lần**; kết nối thứ
hai bị từ chối với mã 1013.

Thứ tự tắt: cửa sổ phiên dịch → pipeline (Ctrl+C) → tunnel → (nếu cần) hai
server trên pod.

### Cuộc họp được ghi lại

Mỗi lần bấm **Bắt đầu** sinh ra hai file trong `recordings/` (đổi bằng
`--out-dir`):

| File | Cho ai đọc |
| --- | --- |
| `meeting-<ngày>-<giờ>.txt` | Biên bản: từng câu đã chốt kèm bản dịch, người nói, giờ |
| `meeting-<ngày>-<giờ>.debug.txt` | Mọi message theo đúng thứ tự tới, kể cả chữ mờ |

Cả hai là UTF-8 và được ghi xuống đĩa ngay. Biên bản được **viết lại** mỗi khi
có bản dịch tới muộn hoặc pipeline sửa nhãn người nói, nên nó luôn khớp với màn
hình.

So các lần chạy của **cùng một cuộc họp** (kiểm tra là cùng cuộc họp trước khi
in số nào):

```powershell
python -m server.analysis.compare_logs recordings\meeting-A.debug.txt recordings\meeting-B.debug.txt
```

Hoặc chạy bằng script kiểm thử, không giao diện:

```powershell
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 120
```

---

## 4. Hiệu năng đã đo và giới hạn đã biết

Đo ngày 2026-11-12, replay 180 s của `meeting_30min.wav`:

| Đo gì | Kết quả |
| --- | --- |
| Một lần giải mã chữ mờ (4 s, beam 1), đi về qua tunnel | trung vị 204 ms, p95 287 ms, max 547 ms — ngân sách 600 ms |
| Một câu chốt (7 s, beam 5) | 524 ms — ngân sách 2450 ms |
| Một câu dịch Gemma | ~940 ms |
| Thời gian chờ pod (`partial_asr` + `asr`) | 46.3 s / 180 s = 25.7% |
| Thời gian CPU Windows (LID chữ mờ 47.8 s, giọng nói 16.0 s, tách ngôn ngữ 14.0 s, LID câu 8.8 s) | 86.8 s / 180 s = **48.2%** |
| Cả buổi replay | 138 s / 180 s = 76.7% — vẫn nhanh hơn thời gian thực |

Check "This machine's own stages keep well inside real time" của
`test_real_streaming.py` (ngân sách 35%) vì thế **đang FAIL** trên máy này.
Đó là trạng thái đã biết và đã chấp nhận: chưa đổi hành vi pipeline để kết quả
còn giống hệt baseline. Phần lớn là LID chạy trên mỗi lần chữ mờ (26.6%); nếu
sau này cần thêm khoảng dư CPU, đó là chỗ để giảm.

Giới hạn khác:

- **Mạng tới pod rớt ~2 giây** = ba lần ASR hỏng liên tiếp = ASR **tắt cho cả
  cuộc họp** (`STAGE_FAILURE_LIMIT=3`, quy tắc có từ trước). Khởi động lại
  pipeline để có lại ASR.
- Lỗi từ server có chứa `cuda` hoặc `out of memory` tắt ASR **ngay lần đầu**
  (lỗi thiết bị không được thử lại).
- Client (giao diện, thu loopback) chạy chung CPU với pipeline. Đừng chạy việc
  nặng khác trên máy Windows trong lúc họp.
- Hai server trên pod **không có xác thực** và chỉ nghe ở `127.0.0.1`. Đừng đổi
  sang `--host 0.0.0.0` nếu chưa có lớp xác thực phía trước.

---

## 5. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách sửa |
| --- | --- | --- |
| Đường dẫn trong lỗi là `...\Programs\Python\Python311\Lib\site-packages`, không phải `.venv` | Chạy bằng `py -3.11` sau khi bật venv | Gõ `python`; kiểm `python -c "import sys; print(sys.prefix)"` |
| `CERTIFICATE_VERIFY_FAILED` khi tải model | Mạng công ty soi HTTPS; `truststore` chưa được cài | `python -m pip install -r server\requirements.lock.txt`; không đặt `SYSTEM_CERTS=0` |
| `[WinError 1314] A required privilege is not held by the client` | SpeechBrain cũ tạo symlink | `git pull` và cài lại lock (đã chuyển sang chép file) |
| `Lazy import of LazyModule(... k2_fsa ...) failed` khi nạp LID | SpeechBrain 1.1.0 trên Windows | Cài lại lock; `speechbrain.__version__` phải là `1.1.2` |
| `asr_loaded: false`, `asr_error` nói không nối được `127.0.0.1:8002` | Tunnel chưa mở hoặc `whisper_server.py` chưa chạy | `curl.exe -s http://127.0.0.1:8002/health`; mở tunnel; khởi động lại pipeline |
| `asr_error` nói model khác `ASR_MODEL` hoặc `api_version` khác | Pod chạy checkpoint khác, hoặc `whisper_server.py` cũ | Chép lại `whisper_server.py` từ cùng commit; `--model large-v3` |
| `translation_error` nói model khác | vLLM đang phục vụ model khác `TRANSLATE_MODEL` | Khởi động lại vLLM bằng đúng lệnh ở bước 1 |
| vLLM không khởi động vì thiếu bộ nhớ GPU | `whisper_server.py` đã chạy trước và giữ chỗ | Tắt Whisper, chạy vLLM trước, rồi Whisper |
| Giữa cuộc họp chữ ngừng hiện, log báo ASR bị tắt | Mạng rớt đủ lâu cho ba lần hỏng liên tiếp | Kiểm tunnel, khởi động lại pipeline |
| Câu ra tiếng Nga / Trung / Ý… | Tầng LID không nạp (`language_model_loaded: false`) | Xem `language_model_error`; thường là chạy sai interpreter |

---

## 6. Biến môi trường

Không cần biến nào để chạy. Bảng này để **thử nghiệm**, và để đối chiếu khi
`/health` báo `overrides` không rỗng. Danh sách được lấy thẳng từ
`server/config.py`, nơi duy nhất trong server đọc biến môi trường.

| Biến | Mặc định | Đặt khi nào |
| --- | --- | --- |
| `ENABLE_NOISE_FILTER` | không đặt | `=1` để bật tầng lọc nhiễu. **Đừng bật trên Windows CPU**: AST tốn 0.42 thời lượng audio |
| `NOISE_DEVICE` | `cpu` | `cuda` để thử AST trên GPU. Từng đổ ở cuDNN rồi segfault |
| `DISABLE_OVERLAP` | không đặt | `=1` cho ASR ăn audio thô |
| `LANGUAGE_SPLIT` | `1` | `=0` để thôi cắt câu chứa hai ngôn ngữ |
| `SPEAKER_RECLUSTER` | không đặt | `=1` để gửi nhãn người nói đã gom cụm lại. Đo được: làm nhãn **tệ hơn** trên họp 4 người |
| `ASR_PROMPT_ON_PARTIALS` | không đặt | `=1` để cả chữ mờ cũng nhận từ vựng mồi — chỉ để đo |
| `MEETING_DATA_DIR` | `server/data` | Đọc danh sách chặn/giữ và từ vựng ở chỗ khác |
| `SYSTEM_CERTS` | `1` | `=0` để kiểm chứng chỉ HTTPS bằng `certifi` thay vì kho của Windows |
| `ASR_BACKEND` | `remote` | `local` để chạy faster-whisper trong tiến trình — chỉ trên máy có GPU |
| `ASR_SERVER_URL` | `http://127.0.0.1:8002` | `whisper_server.py` ở địa chỉ khác, không qua tunnel |
| `ASR_TIMEOUT_S` | `10` | Khi mạng tới pod chậm |
| `ASR_DEVICE`, `LID_DEVICE`, `SPEAKER_DEVICE` | tự chọn | `cuda` / `cpu`; máy Windows không có GPU nên tự chọn ra `cpu` |
| `ASR_MODEL`, `LID_MODEL`, `SPEAKER_EMBEDDING_MODEL`, `AST_MODEL_ID` | xem `config.py` | Đổi checkpoint |
| `ASR_COMPUTE_TYPE` | tự chọn | `float16` / `int8`, chỉ cho `ASR_BACKEND=local` |
| `ASR_CACHE_DIR`, `LID_CACHE_DIR`, `SPEAKER_CACHE_DIR` | `models/...` | Đổi chỗ tải model về |
| `TRANSLATE_MODEL` | `google/gemma-4-12b-it` | Phải khớp với model vLLM đang phục vụ |
| `TRANSLATE_BASE_URL` | `http://127.0.0.1:8001/v1` | Khi vLLM ở cổng khác |
| `TRANSLATE_TIMEOUT_S` | `20` | Khi vLLM chậm |
| `VLLM_PORT`, `VLLM_GPU_MEMORY_UTILIZATION` | `8001`, `0.85` | Đọc bởi `server/launch_vllm.py --print`, in ra lệnh vLLM để dán lên pod |

**Sau mọi thử nghiệm, mở terminal mới.** Xoá từng biến dễ sót; terminal mới
thì không.

---

## 7. Kiểm thử

```powershell
# Dev PC: toàn bộ unit test, không cần GPU, không cần sound card
.venv\Scripts\python.exe -m pytest -q
```

Test thật cần phần cứng thật và nằm riêng:

| Thư mục | Chạy ở đâu |
| --- | --- |
| `server/tests_real/` | Windows (pipeline, CPU), gọi model trên pod — xem [README](server/tests_real/README.md) |
| `gpu_server/` | GPU pod: chỉ khởi động và `curl` hai server — xem [README](gpu_server/README.md) |
| `client/tests_real/` | Windows Client PC — xem [README](client/tests_real/README.md) |

Đổi hình dạng request/response của `whisper_server.py` thì tăng `API_VERSION`
ở cả `gpu_server/whisper_server.py` lẫn `server/pipeline/asr_remote.py`, rồi
chép lại file lên pod; pipeline từ chối một server nói phiên bản khác.

## 8. Tinh chỉnh

| Muốn đổi gì | Đọc file nào |
| --- | --- |
| Ngưỡng VAD, cắt câu, streaming ASR, nhận dạng người nói, dịch… | [`docs/TUNING.md`](docs/TUNING.md) |
| Câu Whisper bịa cần chặn / giữ | [`server/data/README.md`](server/data/README.md) |
| Tên người, tên dự án, thuật ngữ Whisper nghe sai | [`server/data/vocabulary.txt`](server/data/vocabulary.txt) |

Mỗi thông số trong `TUNING.md` đi kèm **phép đo đã chọn ra nó** và điều gì
hỏng nếu chỉnh sai hướng. Đọc trước khi đổi — phần lớn các số đó nằm giữa hai
vùng đo được rất hẹp.

## 9. Những điều đã học được và đừng làm lại

- **Whisper bịa văn bản với chỉ số tự tin hơn khi phiên âm thật.** Ba lớp
  chắn thống kê không bắt được. Xem `server/data/README.md`.
- **Ép sai ngôn ngữ không báo lỗi.** Whisper trả về văn bản trôi chảy, tự tin
  và sai, rồi tầng dịch dịch trung thành cái vô nghĩa đó.
- **Đừng tự chèn khoảng trắng giữa các từ của Whisper.** Tiếng Nhật không có
  khoảng trắng; nối bằng dấu cách làm 88% câu tiếng Nhật của một lần chạy bị
  cách từng chữ.
- **Hai lần giải mã đặt cùng một từ lệch nhau một chút.** Xét ranh giới theo
  cuối từ thì từ chốt cuối hiện lại hai lần — 45% câu của cùng lần chạy đó.
- **Khoảng lặng sau câu không phải chỗ trống.** VAD chuyển tiếp ~480 ms
  hangover, và Whisper trả lời im lặng bằng chữ.
- **Lịch sử dịch phải chụp lúc câu được chốt.** Đọc lúc dịch thì nó đã chứa
  chính câu đang dịch, và model trả lại nguyên văn.
- **Điểm số của bộ lọc nhiễu không phân biệt được câu bịa với câu thật.** Đã
  thử và đã bác bỏ bằng số liệu.
- **Lịch sử hội thoại có bản dịch trong đó sẽ lái model dịch sai ngôn ngữ.**
  Nó đọc lên như một chuỗi ví dụ mẫu.
- **Một tầng hỏng từng ăn cả cuộc họp.** Giờ lỗi bắt ở mức *tầng*: câu đi
  tiếp thiếu tầng đó, và tầng hỏng liên tục thì bị tắt và báo về client.
- **Lỗi CUDA thì đừng thử lại.** Lần gọi lại vào đúng tầng đó từng segfault cả
  tiến trình. Tầng gặp lỗi thiết bị bị tắt ngay lần đầu.
- **Đừng bật `condition_on_previous_text`.** Đó là cách một câu bịa thành cả
  đoạn bịa.
- **`no_speech_prob` một mình không được phép loại đoạn nào** — trừ trên mẩu
  audio dưới 600 ms, nơi mọi câu bịa đã xác nhận xuất hiện.
- **Voiceprint trên cửa sổ 1 giây không phân biệt được giọng.** Cùng một
  đoạn cho 0.12 theo cửa sổ và 0.77 theo cả câu.
- **So hai lần chạy thì phải chắc là cùng một cuộc họp.** Đã từng kết luận sai
  vì không để ý hai bản ghi bắt đầu lệch nhau vài chục giây.
- **Whisper qua HTTP phải trả đủ thứ chính sách streaming đọc** — thời điểm
  từng từ, ba điểm số mỗi đoạn, beam search. Endpoint transcription của vLLM
  không có, nên Whisper có server riêng.
- **`py -3.11` bỏ qua venv đang bật.** Một lần replay đã chạy bằng Python hệ
  thống, mất LID mà không báo lỗi gì.
- **Python không dùng kho chứng chỉ của Windows.** Sau proxy soi HTTPS, mọi lần
  tải model hỏng cho tới khi `truststore` được bật.

## 10. Cấu trúc thư mục

```
client/          thu âm, gửi WebSocket, UI, biên bản           (Windows)
server/          pipeline + FastAPI, gọi model qua HTTP         (Windows)
  analysis/      đọc và so nhật ký gỡ lỗi, không cần GPU
  data/          danh sách chặn/giữ và từ vựng, sửa được không cần code
  pipeline/      một file cho mỗi tầng; asr_remote.py gọi Whisper trên pod
  tests_real/    test thật, chạy trên Windows
gpu_server/      whisper_server.py + requirements cho pod      (GPU pod)
common/          hợp đồng audio và protocol, dùng chung hai phía
docs/            hướng dẫn tinh chỉnh
```

`common/protocol.py` là nguồn duy nhất cho định dạng audio và các message.
Hai phía import từ đó nên không thể lệch nhau — lệch định dạng audio làm hỏng
tiếng nói một cách âm thầm chứ không ném lỗi.
