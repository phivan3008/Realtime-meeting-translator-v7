# Hướng dẫn lấy source code từng giai đoạn và tạo bằng chứng

Tài liệu đi kèm báo cáo, không đưa vào báo cáo.

## 1. Mốc code của từng giai đoạn

Tất cả các commit dưới đây đã có trên GitHub (`phivan3008/Realtime-meeting-translator-v7`).

| Giai đoạn | Nhánh | Commit cuối | Model dịch | Client có cửa sổ họp? | Nhật ký cũ đã có trong `recordings/` |
| --- | --- | --- | --- | --- | --- |
| 0 — Hiện trạng ban đầu | `main` | `40c768e` | Qwen3.5-9B | Không | — |
| 1 — Tách người nói | `main` | `e430765` | Qwen3.5-9B | Có, nhưng chưa ghi nhật ký | — |
| 2 — Chặn nội dung bịa | `main` | `6e8fa0a` | Qwen3.5-9B | Có | `meeting-20260828-104348` |
| 3 — Nguyên nhân câu chốt kém | `improve-from-40c768e` | `8383306` | Qwen3.5-9B | Không | `meeting-20260911-003757` (bản ngay trước), `meeting-20260911-104145` |
| 4 — Hai thứ tiếng trong một câu | `improve-from-40c768e` | `599031a` | Qwen3.5-9B | Không | `meeting-20260911-144023` (chạy ở `939ced7`, chỉ trước bước thêm từ vựng) |
| 5 — ASR streaming, đổi model | `alt-from-40c768e` | `d431726` | gemma-4-12b-it | Không | `meeting-20260910-144120` (bản streaming gốc, trước khi đưa vào repo) |
| 6 — Hoàn thiện streaming | `alt-from-40c768e` | `28bbeae` | gemma-4-12b-it | Có | `meeting-20260916-210402` |
| 7 — Giữ đủ lượt nói | `alt-from-40c768e` | `b7f4217` | gemma-4-12b-it | Có | `meeting-20260917-081820` (`a07ab71`), `-114032` (`6b6e1bb`), `-135442` (`95a68fd`), `-155715` (`b7f4217`) |

`recordings/` không nằm trong git. Các nhật ký cũ hiện ở thư mục `recordings/` trên
Dev PC; hãy sao lưu chúng trước khi làm gì khác.

## 2. Lấy source code của một giai đoạn

**Cách 1 — tải file zip, không cần git:**

```
https://github.com/phivan3008/Realtime-meeting-translator-v7/archive/<commit>.zip
```

Ví dụ giai đoạn 2: `.../archive/6e8fa0a.zip`.

**Cách 2 — mỗi giai đoạn một thư mục riêng (khuyến nghị, trên cả pod lẫn client):**

```bash
git clone https://github.com/phivan3008/Realtime-meeting-translator-v7.git rmt
cd rmt
git worktree add ../rmt-gd0 40c768e
git worktree add ../rmt-gd1 e430765
git worktree add ../rmt-gd2 6e8fa0a
git worktree add ../rmt-gd3 8383306
git worktree add ../rmt-gd4 599031a
git worktree add ../rmt-gd5 d431726
git worktree add ../rmt-gd6 28bbeae
git worktree add ../rmt-gd7 b7f4217
```

## 3. Điều kiện để các lần chạy so sánh được với nhau

- **Cùng một cuộc họp:** bản ghi 4 người, khoảng 27 phút, đã dùng cho mọi lần chạy
  trước.
  - Trên pod là `recordings/meeting_30min.wav`.
  - Trên Client PC là file audio gốc của cuộc họp đó.
- **Phát từ đầu tới hết.** Các nhật ký cũ dài khoảng 1640 s. Không tua, giữ
  nguyên âm lượng hệ thống và thiết bị phát như các lần trước.
- **Mỗi giai đoạn chạy trên một pod "sạch":** tắt hết tiến trình của giai đoạn
  trước và mở terminal mới (để không còn biến môi trường cũ).
- **Mỗi giai đoạn một venv riêng:** giai đoạn 0–4 dùng `transformers==5.15.1`,
  giai đoạn 5–7 dùng `transformers==5.14.1`.

## 4. Chạy server trên GPU pod

### 4.1 Cài môi trường (một lần cho mỗi thư mục giai đoạn)

```bash
cd ../rmt-gdN
python3.11 -m venv .venv && source .venv/bin/activate
python --version                                   # 3.11.x
python3.11 -m pip install -r server/requirements.lock.txt
```

### 4.2 Khởi động vLLM (terminal 1, luôn chạy trước)

**Giai đoạn 0–4 (Qwen):**

```bash
source .venv/bin/activate
python3.11 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.55
```

**Giai đoạn 5–7 (Gemma):**

```bash
source .venv/bin/activate
python3.11 server/launch_vllm.py
```

### 4.3 Khởi động server âm thanh (terminal 2), ghi log ra file

**Giai đoạn 0, 1, 3, 4, 5.** Ở các mốc này tầng lọc nhiễu mặc định bật và chạy
GPU; trên pod nó từng lỗi cuDNN rồi làm sập tiến trình. Ép nó chạy CPU:

```bash
source .venv/bin/activate
export NOISE_DEVICE=cpu
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000 2>&1 | tee server-gdN.log
```

> **Không đặt `DISABLE_NOISE_FILTER` ở các mốc này.** Code cũ có lỗi làm server bỏ
> qua luôn việc nạp Whisper và bộ dịch khi biến này được đặt (lỗi được sửa ở giai
> đoạn 2).

**Giai đoạn 2, 6, 7** (không cần biến môi trường):

```bash
source .venv/bin/activate
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000 2>&1 | tee server-gdN.log
```

Chờ `Application startup complete`, rồi lưu lại tình trạng các tầng:

```bash
curl -s http://127.0.0.1:8000/health | tee health-gdN.json
```

Kiểm tra `asr_loaded` và `translation_loaded` đều là `true`. Từ giai đoạn 2 trở đi
còn kiểm thêm `in_venv: true` và `overrides: {}`.

## 5. Chạy client trên Windows Client PC

```powershell
# Tunnel (cửa sổ riêng, để mở suốt buổi)
ssh -N -L 8000:127.0.0.1:8000 <user>@<pod-ssh-host>
```

**Giai đoạn 2, 6, 7** — dùng client của chính giai đoạn đó:

```powershell
cd ..\rmt-gdN
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -r client\requirements.lock.txt
py -3.11 -m client.ui.main --url ws://127.0.0.1:8000 --out-dir recordings-gdN
```

**Giai đoạn 0, 1, 3, 4, 5** — các mốc này chưa có cửa sổ họp, hoặc cửa sổ chưa ghi
nhật ký (giai đoạn 1). Chạy server của giai đoạn đó, nhưng dùng client của
**giai đoạn 7**.
- Giao thức giữa hai bên không đổi (phiên bản 1), đây cũng là cách các nhật ký
  ngày 11/09 đã được tạo ra.

```powershell
cd ..\rmt-gd7
py -3.11 -m client.ui.main --url ws://127.0.0.1:8000 --out-dir recordings-gdN
```

**Quy trình ghi một cuộc họp:**

1. Bấm **Bắt đầu**, đợi thanh trạng thái báo "Đang nghe".
2. Phát file audio cuộc họp từ đầu, để chạy hết.
3. Chụp màn hình cửa sổ ở vài đoạn đáng chú ý (xem mục 7).
4. Đóng cửa sổ bình thường (không kill), chờ nó dịch xong câu cuối.
5. Trong `recordings-gdN\` sẽ có:
   - `meeting-<ngày>-<giờ>.txt`: biên bản;
   - `meeting-<ngày>-<giờ>.debug.txt`: nhật ký dùng để đo.
6. Trên pod, dừng server (Ctrl+C) để in dòng tổng kết vào `server-gdN.log`.

## 6. Đo và so sánh

Chạy từ thư mục giai đoạn 7 (có công cụ đo mới nhất), trên máy nào cũng được:

```powershell
cd ..\rmt-gd7
py -3.11 -m server.analysis.compare_logs `
    ..\rmt-gd0\recordings-gd0\meeting-<...>.debug.txt `
    ..\rmt-gd2\recordings-gd2\meeting-<...>.debug.txt `
    ..\rmt-gd4\recordings-gd4\meeting-<...>.debug.txt `
    ..\rmt-gd6\recordings-gd6\meeting-<...>.debug.txt `
    ..\rmt-gd7\recordings-gd7\meeting-<...>.debug.txt
```

- Dòng cuối bảng phải báo mọi nhật ký là **same meeting**. Nếu không, lần chạy đó
  không so sánh được: kiểm tra lại file audio và thời điểm bắt đầu phát.
- File đầu tiên là mốc so sánh; nên đặt nhật ký giai đoạn 0 lên đầu.
- Nhật ký cũ trong `recordings/` cũng đưa được vào cùng bảng.

**Real test phát lại (chỉ có từ giai đoạn 6), chạy trên pod:**

```bash
cd ../rmt-gd7 && source .venv/bin/activate
python3.11 server/tests_real/test_real_streaming.py \
    --wav /workspace/Realtime-meeting-translator-v7/recordings/meeting_30min.wav \
    --limit-seconds 180
```

File WAV không nằm trong git, nên thư mục mới tạo bằng `git worktree` không có
nó; hãy trỏ `--wav` tới bản đang có trên pod như trên.

Kết quả `RESULT: PASS (15 checks)` là bằng chứng của giai đoạn 7. Hai file
`server/tests_real/output/streaming.debug.txt` và `streaming.server.log` ghi chi tiết
từng lần cắt câu.

## 7. Bằng chứng cần thu cho từng giai đoạn

| Giai đoạn | Số đo (`compare_logs`) | Thêm từ log server / màn hình |
| --- | --- | --- |
| 0 | Toàn bộ bảng — làm mốc | Ảnh chụp: chữ mờ đúng nhưng câu chốt thiếu hoặc sai; câu bịa kiểu "Cảm ơn các bạn đã theo dõi" |
| 1 | — | Chạy thêm một lần với `export SPEAKER_CHANGE_ENABLED=1`. Dòng `voice comparisons: N checks, M cuts, ... deciles [...]` trong `server-gd1.log` chứng minh phép cắt theo giọng không có ngưỡng dùng được; so số câu và nhãn người nói với lần chạy không bật |
| 2 | `near_block_list`, `far_from_partial`, `lost_turns` | Trong log: các dòng `ASR kept a segment scored as silence` (câu thật được giữ lại); `stages {...}` không còn mục `noise`; `slowest sentence` |
| 3 | `language_disagrees`, `far_from_partial` | Trong log: `language retries`, `shown as their running text` |
| 4 | `lost_turns`, `language_disagrees` | Trong log: dòng `split on a language change over N probes` |
| 5 | `update_survival`, `updates_wiping_half`, `japanese_spaced`, `with_repeats` | Ảnh chụp: chữ mờ đứng yên; câu tiếng Nhật bị cách từng chữ; từ lặp đôi |
| 6 | `japanese_spaced`, `with_repeats`, `refused_translations`, `language_disagrees`, `mixed_language` | Dòng `ASR: ... repeated words removed at a join`; `echoes: ... retried` |
| 7 | Toàn bộ bảng, nhất là `lost_turns`, `language_disagrees`, `mixed_language` | Output `RESULT: PASS (15 checks)`; dòng `changed language part way, N cut there`; `refused by the running text`; `speakers (measured only): ... sized [...]` |

Tìm nhanh các dòng này trong log server:

```bash
grep -E "finished:|language splits|decoded whole again|ASR:|echoes|speakers|voice comparisons" server-gdN.log
```
