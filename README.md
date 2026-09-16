# Realtime Meeting Translator (VI ↔ JA)

Phiên dịch hai chiều Việt–Nhật theo thời gian thực cho họp online. Âm thanh
cuộc họp được thu ở máy Windows, xử lý trên GPU server, và trả về phụ đề kèm
bản dịch.

- Thiết kế: [`DESIGN.md`](DESIGN.md)
- Kiến trúc theo code hiện tại, kèm sơ đồ: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Mọi thông số và phép đo đứng sau chúng: [`docs/TUNING.md`](docs/TUNING.md)

## Ba máy

| Máy | Vai trò |
| --- | --- |
| **Dev PC** | Viết code, chạy unit test, commit/push |
| **Windows Client PC** | Thu âm thanh cuộc họp qua WASAPI loopback, gửi lên server, hiển thị |
| **GPU Server** (pod H100) | Chạy toàn bộ pipeline và vLLM |

Máy test **pull code từ GitHub**, nên mọi thay đổi phải được push trước khi
chạy thử.

## Đường đi của một câu nói

```
Client                    Server
──────                    ──────
WASAPI loopback           1. VAD (Silero)            cắt thành đoạn có tiếng nói
16 kHz mono 16-bit        2. Buffer Manager          gom thành câu, chốt khi ngắt
chunk 200 ms ─────────►  6b. Language split         cắt câu chứa hai ngôn ngữ
                          3. Noise Filter (AST)      mặc định tắt
                          4. Overlap Resolver        hạ giọng chồng lấn
                          5. Diarization (ECAPA)     ai đang nói
                          6. Language ID             tiếng Việt hay tiếng Nhật
                          7. ASR (Whisper large-v3)  streaming, chốt dần từng từ
      ◄──── partial ───      (mỗi 600 ms)
      ◄──── final ─────   8. Translation (Gemma/vLLM) chạy ngoài luồng audio
                         5b. Gom cụm lại            chỉ đo, không gửi (mặc định)
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

### Windows Client

```powershell
git clone <repo>; cd Realtime-meeting-translator-v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                      # phải in 3.11.x
py -3.11 -m pip install -r client\requirements.lock.txt
```

Client **không chạy ML**. Silero VAD nằm ở server — xem `DESIGN.md` mục 3.

## Chạy

> **Không cần đặt biến môi trường nào.** Mọi thông số đều có mặc định dùng
> được. Nếu ai đó bảo bạn đặt một biến, đó là để **thử nghiệm** — xem bảng ở
> [Biến môi trường](#biến-môi-trường) bên dưới, và nhớ mở terminal mới khi
> chạy lại bình thường.
>
> Khởi động in ra `No environment overrides; running on the defaults`. Nếu
> thay vào đó là `Environment overrides in effect: {...}` thì có biến còn sót
> từ terminal trước, và nó đang đổi hành vi.

Server cần **hai tiến trình**, và **vLLM phải khởi động trước**: nó giữ 85%
toàn bộ GPU lúc nạp và từ chối khởi động nếu không đủ chỗ.

**Terminal 1 — vLLM:**
```bash
source .venv/bin/activate
python3.11 server/launch_vllm.py          # thêm --print để chỉ xem lệnh
```

**Terminal 2 — server âm thanh** (sau khi vLLM báo sẵn sàng):
```bash
source .venv/bin/activate
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Chờ `Application startup complete`, rồi kiểm tra:

```bash
curl -s http://127.0.0.1:8000/health
```

**Kiểm `in_venv` trước tiên.** Nếu là `false`, server đang chạy bằng
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
| `asr_loaded` | `true` |
| `translation_loaded` | `true` |

Cái nào sai thì dòng `*_error` bên cạnh nói lý do. Server vẫn khởi động khi
thiếu một tầng — thiếu một tầng còn hơn từ chối cuộc họp — nhưng lúc đó phụ đề
không đáng tin, và **không có gì trong lúc chạy nói cho bạn biết**.

> Đây không phải chuyện giả định. `overlap_resolver_loaded` từng `false` suốt
> nhiều lần chạy vì thiếu `pedalboard`, và nó không nằm trong danh sách README
> bảo phải kiểm. Một phép đo A/B về chính tầng đó đã chạy và cho kết quả vô
> nghĩa, vì cả hai lần chạy đều không có tầng đó.

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
chạy — phần đầu của nó đã được chốt và sẽ không đổi; chữ đậm là câu đã chốt,
bản dịch màu xanh ngay dưới. Câu nào không dịch được sẽ nói rõ lý do và nguyên
văn model đã trả lời, thay vì để trống.

Đóng cửa sổ sẽ gửi `bye` và **chờ máy chủ dịch xong câu cuối**, nên đừng tắt
bằng cách kill tiến trình.

### Cuộc họp được ghi lại

Mỗi lần bấm **Bắt đầu** sinh ra hai file trong `recordings/` (đổi bằng
`--out-dir`):

| File | Cho ai đọc |
| --- | --- |
| `meeting-<ngày>-<giờ>.txt` | Biên bản: từng câu đã chốt kèm bản dịch, người nói, giờ |
| `meeting-<ngày>-<giờ>.debug.txt` | Mọi message theo đúng thứ tự tới, kể cả chữ mờ |

Cả hai là UTF-8 và được ghi xuống đĩa ngay. Biên bản được **viết lại** mỗi khi
có bản dịch tới muộn hoặc máy chủ sửa nhãn người nói, nên nó luôn khớp với màn
hình.

So các lần chạy của **cùng một cuộc họp** (kiểm tra là cùng cuộc họp trước khi
in số nào):

```powershell
py -3.11 -m server.analysis.compare_logs recordings\meeting-A.debug.txt recordings\meeting-B.debug.txt
```

Hoặc chạy bằng script kiểm thử, không giao diện:

```powershell
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 120
```

Server nhận **một cuộc họp mỗi lần**; kết nối thứ hai bị từ chối với mã 1013.

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
| `ASR_DEVICE`, `LID_DEVICE`, `SPEAKER_DEVICE` | tự chọn | `cuda` / `cpu` |
| `ASR_MODEL`, `LID_MODEL`, `SPEAKER_EMBEDDING_MODEL`, `AST_MODEL_ID` | xem `config.py` | Đổi checkpoint |
| `ASR_COMPUTE_TYPE` | tự chọn | `float16` / `int8` |
| `ASR_CACHE_DIR`, `LID_CACHE_DIR`, `SPEAKER_CACHE_DIR` | `models/...` | Đổi chỗ tải model về |
| `TRANSLATE_MODEL` | `google/gemma-4-12b-it` | Phải khớp với model vLLM đang phục vụ |
| `TRANSLATE_BASE_URL` | `http://127.0.0.1:8001/v1` | Khi vLLM ở cổng khác |
| `TRANSLATE_TIMEOUT_S` | `20` | Khi vLLM chậm |
| `VLLM_PORT`, `VLLM_GPU_MEMORY_UTILIZATION` | `8001`, `0.85` | Đọc bởi `server/launch_vllm.py` |

**Sau mọi thử nghiệm, mở terminal mới.** `unset` từng biến dễ sót; terminal mới
thì không.

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
client/          thu âm, gửi WebSocket, UI, biên bản
server/          pipeline + FastAPI
  analysis/      đọc và so nhật ký gỡ lỗi, không cần GPU
  data/          danh sách chặn/giữ và từ vựng, sửa được không cần code
  pipeline/      một file cho mỗi tầng
common/          hợp đồng audio và protocol, dùng chung hai phía
docs/            hướng dẫn tinh chỉnh
```

`common/protocol.py` là nguồn duy nhất cho định dạng audio và các message.
Hai phía import từ đó nên không thể lệch nhau — lệch định dạng audio làm hỏng
tiếng nói một cách âm thầm chứ không ném lỗi.
