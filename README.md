# Realtime Meeting Translator (VI ↔ JA)

Phiên dịch hai chiều Việt–Nhật theo thời gian thực cho họp online. Âm thanh
cuộc họp được thu **và xử lý** ở máy Windows; GPU server chỉ phục vụ hai model
lớn (Whisper và Gemma 4) qua HTTP. Kết quả là phụ đề kèm bản dịch.

- Thiết kế: [`DESIGN.md`](DESIGN.md)
- Kiến trúc theo code hiện tại, kèm sơ đồ: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Mọi thông số và phép đo đứng sau chúng: [`docs/TUNING.md`](docs/TUNING.md)

## Ba máy

| Máy | Vai trò |
| --- | --- |
| **Dev PC** | Viết code, chạy unit test, commit/push |
| **Windows Client PC** (chỉ CPU) | Thu âm qua WASAPI loopback, **chạy toàn bộ pipeline** (`server/`, ở `127.0.0.1:8000`), hiển thị |
| **GPU Server** (pod H100) | **Chỉ phục vụ model**: vLLM cho Gemma 4 (cổng 8001) và `gpu_server/whisper_server.py` cho Whisper (cổng 8002). Không chứa code pipeline |

Máy Windows **pull code từ GitHub**, nên mọi thay đổi phải được push trước khi
chạy thử. Lên pod chỉ chép `gpu_server/whisper_server.py` và
`gpu_server/requirements.lock.txt` — xem [`gpu_server/README.md`](gpu_server/README.md).

## Đường đi của một câu nói

```
Client (Windows)          Pipeline (cùng máy Windows, CPU)        GPU pod
──────                    ────────────────────────────────        ───────
WASAPI loopback           1. VAD (Silero)            cắt thành đoạn có tiếng nói
16 kHz mono 16-bit        2. Buffer Manager          gom thành câu, chốt khi ngắt
chunk 200 ms ─────────►  6b. Language split         cắt câu chứa hai ngôn ngữ
                          3. Noise Filter (AST)      mặc định tắt
                          4. Overlap Resolver        hạ giọng chồng lấn
                          5. Diarization (ECAPA)     ai đang nói
                          6. Language ID             tiếng Việt hay tiếng Nhật
                          7. ASR (Whisper large-v3)  streaming ──HTTP──► whisper_server.py
      ◄──── partial ───      (mỗi 600 ms)
      ◄──── final ─────   8. Translation (Gemma)      ───────HTTP──► vLLM
                         5b. Gom cụm lại            chỉ đo, không gửi (mặc định)
      ◄─ translation ──
```

Câu và bản dịch là **hai message riêng**: câu gửi đi ngay khi Whisper chốt,
bản dịch theo sau và được ghép bằng `sentence_id`. Một lần gọi LLM chậm không
được phép giữ đường vào của âm thanh.

## Cài đặt

Python **3.11** ở cả ba máy. Không phải 3.12.

### GPU Server

Chỉ chạy model. Xem [`gpu_server/README.md`](gpu_server/README.md): chép hai
file lên pod, khởi động vLLM rồi `whisper_server.py`.

### Windows (client + pipeline)

```powershell
git clone <repo>; cd Realtime-meeting-translator-v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                      # phải in 3.11.x
python -m pip install -r server\requirements.lock.txt
python -m pip install -r client\requirements.lock.txt
```

`server\requirements.lock.txt` là pipeline chạy CPU: torch CPU, Silero,
SpeechBrain (ECAPA, VoxLingua107), pedalboard. **Không** có vLLM hay
faster-whisper — hai model đó ở pod. Hai lock dùng chung phiên bản cho mọi gói
trùng nhau, nên cài chung một venv được.

> Đừng cài vào Python hệ thống. Nó hạ cấp numpy và protobuf bên dưới mọi thứ
> khác đang dùng chung — dự án này đã mất một buổi chiều vì chuyện đó.

## Chạy

> **Không cần đặt biến môi trường nào** nếu hai cổng của pod được nối về
> `127.0.0.1:8001` và `127.0.0.1:8002` (port-forward hoặc SSH tunnel). Nếu ai
> đó bảo bạn đặt một biến, đó là để **thử nghiệm** — xem bảng ở
> [Biến môi trường](#biến-môi-trường) bên dưới, và nhớ mở terminal mới khi
> chạy lại bình thường.
>
> Khởi động in ra `No environment overrides; running on the defaults`. Nếu
> thay vào đó là `Environment overrides in effect: {...}` thì có biến còn sót
> từ terminal trước, và nó đang đổi hành vi.

**Bước 0 — pod:** vLLM và `whisper_server.py` đang chạy
([`gpu_server/README.md`](gpu_server/README.md)).

**Terminal 1 (Windows) — nối hai cổng của pod về máy này**, để đó:

```powershell
ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 <user>@<pod-ssh-host>
```

Hoặc dùng tab **Ports** của VSCode Remote-SSH: forward cổng 8001 và 8002.

Kiểm tra pod trả lời được từ Windows trước khi chạy gì khác:

```powershell
python server\tests_real\test_real_remote_models.py
```

**Terminal 2 (Windows) — pipeline:**

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
| `noise_filter_loaded` | `false` — mặc định tắt: 237 utterance, bỏ được 0 câu, tốn 22% luồng đọc socket |
| `overlap_resolver_loaded` | `true` |
| `speaker_model_loaded` | `true` |
| `language_model_loaded` | `true` |
| `asr_loaded` | `true`, và `asr_source` chỉ về `http://127.0.0.1:8002 on cuda (float16)` |
| `translation_loaded` | `true`, và `translation_source` chỉ về `http://127.0.0.1:8001/v1` |

Cái nào sai thì dòng `*_error` bên cạnh nói lý do. Pipeline vẫn khởi động khi
thiếu một tầng — thiếu một tầng còn hơn từ chối cuộc họp — nhưng lúc đó phụ đề
không đáng tin, và **không có gì trong lúc chạy nói cho bạn biết**. Pod chưa
chạy hoặc tunnel chưa mở lúc khởi động thì `asr_loaded` là `false`: khởi động
lại pipeline sau khi pod sẵn sàng.

> Đây không phải chuyện giả định. `overlap_resolver_loaded` từng `false` suốt
> nhiều lần chạy vì thiếu `pedalboard`, và nó không nằm trong danh sách README
> bảo phải kiểm. Một phép đo A/B về chính tầng đó đã chạy và cho kết quả vô
> nghĩa, vì cả hai lần chạy đều không có tầng đó.

**Terminal 3 (Windows) — cửa sổ phiên dịch:**

```powershell
python -m client.ui.main --url ws://127.0.0.1:8000
```

Bấm **Bắt đầu**, rồi phát âm thanh cuộc họp. Chữ mờ nghiêng là dự đoán đang
chạy — phần đầu của nó đã được chốt và sẽ không đổi; chữ đậm là câu đã chốt,
bản dịch màu xanh ngay dưới. Câu nào không dịch được sẽ nói rõ lý do và nguyên
văn model đã trả lời, thay vì để trống.

Đóng cửa sổ sẽ gửi `bye` và **chờ pipeline dịch xong câu cuối**, nên đừng tắt
bằng cách kill tiến trình.

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

Pipeline nhận **một cuộc họp mỗi lần**; kết nối thứ hai bị từ chối với mã 1013.

## Biến môi trường

Không cần biến nào để chạy. Bảng này để **thử nghiệm**, và để đối chiếu khi
`/health` báo `overrides` không rỗng. Danh sách được lấy thẳng từ
`server/config.py`, nơi duy nhất trong server đọc biến môi trường.

| Biến | Mặc định | Đặt khi nào |
| --- | --- | --- |
| `ENABLE_NOISE_FILTER` | không đặt | `=1` để bật tầng lọc nhiễu. Đo được: 237 utterance, bỏ 0 câu, tốn 22% luồng đọc socket |
| `NOISE_DEVICE` | `cpu` | `cuda` để thử AST trên GPU. Từng đổ ở cuDNN rồi segfault |
| `DISABLE_OVERLAP` | không đặt | `=1` cho ASR ăn audio thô |
| `LANGUAGE_SPLIT` | `1` | `=0` để thôi cắt câu chứa hai ngôn ngữ |
| `SPEAKER_RECLUSTER` | không đặt | `=1` để gửi nhãn người nói đã gom cụm lại. Đo được: làm nhãn **tệ hơn** trên họp 4 người |
| `ASR_PROMPT_ON_PARTIALS` | không đặt | `=1` để cả chữ mờ cũng nhận từ vựng mồi — chỉ để đo |
| `MEETING_DATA_DIR` | `server/data` | Đọc danh sách chặn/giữ và từ vựng ở chỗ khác |
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

**Sau mọi thử nghiệm, mở terminal mới.** `unset` từng biến dễ sót; terminal mới
thì không.

## Kiểm thử

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

## Tinh chỉnh

| Muốn đổi gì | Đọc file nào |
| --- | --- |
| Ngưỡng VAD, cắt câu, streaming ASR, nhận dạng người nói, dịch… | [`docs/TUNING.md`](docs/TUNING.md) |
| Câu Whisper bịa cần chặn / giữ | [`server/data/README.md`](server/data/README.md) |
| Tên người, tên dự án, thuật ngữ Whisper nghe sai | [`server/data/vocabulary.txt`](server/data/vocabulary.txt) |

Mỗi thông số trong `TUNING.md` đi kèm **phép đo đã chọn ra nó** và điều gì
hỏng nếu chỉnh sai hướng. Đọc trước khi đổi — phần lớn các số đó nằm giữa hai
vùng đo được rất hẹp.

## Những điều đã học được và đừng làm lại

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

## Cấu trúc thư mục

```
client/          thu âm, gửi WebSocket, UI, biên bản        (Windows)
server/          pipeline + FastAPI, gọi model qua HTTP      (Windows)
gpu_server/      whisper_server.py + requirements cho pod   (GPU pod)
  analysis/      đọc và so nhật ký gỡ lỗi, không cần GPU
  data/          danh sách chặn/giữ và từ vựng, sửa được không cần code
  pipeline/      một file cho mỗi tầng
common/          hợp đồng audio và protocol, dùng chung hai phía
docs/            hướng dẫn tinh chỉnh
```

`common/protocol.py` là nguồn duy nhất cho định dạng audio và các message.
Hai phía import từ đó nên không thể lệch nhau — lệch định dạng audio làm hỏng
tiếng nói một cách âm thầm chứ không ném lỗi.
