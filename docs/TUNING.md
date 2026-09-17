# Hướng dẫn tinh chỉnh

Mọi thông số nằm trong `server/config.py` và `client/config.py`. Tài liệu này
giải thích **mỗi số nghĩa là gì, nó được chọn từ phép đo nào, và điều gì hỏng
nếu chỉnh sai hướng**.

Danh sách câu bị chặn và từ vựng mồi không nằm ở đây — xem
[`server/data/README.md`](../server/data/README.md).

## Cách đọc tài liệu này

Mỗi thông số có một dòng **Đo được**. Đó là dữ liệu thật từ các lần chạy trên
máy client Windows và pod H100, không phải giá trị mặc định của thư viện.
Nếu bạn chỉnh một số, hãy đo lại theo đúng cách đó rồi cập nhật lại đây.

Sau khi sửa `config.py` phải **khởi động lại uvicorn**.

**Trước mọi phép đo, kiểm `"in_venv": true` trong `/health`.** Server chạy
được bằng interpreter khác mà không kêu ca gì — nó nạp gần hết pipeline và
phục vụ cuộc họp bình thường. Ba phép đo của dự án này đã phải vứt vì lý do đó.

## Mốc so sánh

Cuộc họp thật, 593.6 giây audio, pod H100, trong venv, tầng lọc nhiễu tắt.
Đo trên pipeline **trước** ASR streaming (partial cửa sổ trượt, final giải mã
cả câu), nên chi phí `partial_asr`/`asr` của nhánh này sẽ khác — đo lại bằng
`server/tests_real/test_real_streaming.py`.

```
118 utterances (0 dropped as noise, 115 shaped, 118 identified,
                109 with a language)
682 transcripts, 87 translations, 711 partials
slowest sentence 0.4 s, slowest running text 1.5 s
stages {'partial_asr': 80.5, 'asr': 19.9, 'partial_language': 6.9,
        'speaker': 1.5, 'language': 0.8, 'overlap': 0.2, 'recluster': 0.0}
5 sentences came out in a different language than the running text predicted
7 speakers after 6 reclustering runs, 13 labels corrected
```

Đọc ra được:

| | |
| --- | --- |
| pipeline chiếm | **18.5%** luồng đọc socket (109.8 / 593.6 s) |
| chữ mờ chiếm | **73%** của con số đó (80.5 s) — gấp 4 lần các câu đã chốt |
| tầng chồng lấn | **0.2 s**, tức 0.03%. Nó động vào 115/118 câu |
| bất đồng ngôn ngữ | 5 / 118 câu, **4%** |

Chữ mờ là chỗ duy nhất còn dư địa đáng kể. Nó đã bị giới hạn 4 giây
(`PARTIAL_WINDOW_SECONDS`); nới `PARTIAL_INTERVAL_MS` lên là cách rẻ tiếp
theo, đổi lại chữ mờ giật hơn. **Chưa cần** — hiện không có cảnh báo
`held the socket` nào.

## Quy tắc chung của cả pipeline

**Bỏ sót đắt hơn báo nhầm.** Một câu bị xoá là câu không ai nghe được và
không ai biết đã mất. Một tiếng ho lọt qua chỉ tốn một lần gọi Whisper. Mọi
ngưỡng trong dự án này đều nghiêng về phía giữ lại.

**Ngưỡng phải nằm giữa hai vùng đo được, không nằm sát mép.** Cửa sổ giữa
"đúng" và "sai" chỉ hẹp đi khi có thêm dữ liệu, không bao giờ rộng ra.

---

## 1. VAD — phát hiện tiếng nói

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `VAD_THRESHOLD` | `0.5` | Xác suất Silero để coi một khung 32 ms là tiếng nói |
| `VAD_MIN_SPEECH_MS` | `96` | Phải có ngần này tiếng nói mới mở đoạn |
| `VAD_MIN_SILENCE_MS` | `500` | Phải im lặng ngần này mới đóng đoạn |
| `VAD_SPEECH_PAD_MS` | `256` | Audio giữ lại phía trước đoạn |

**`VAD_MIN_SPEECH_MS`** — Silero bắn xác suất cao trong chớp nhoáng khi có
tiếng gõ phím hay đóng cửa. Ba khung liên tiếp (96 ms) loại được chúng.

- Giảm xuống: bắt được câu chêm rất ngắn, nhưng mỗi tiếng gõ phím thành một
  đoạn và tốn một lần gọi Whisper.
- Tăng lên: mất các câu đáp ngắn (`はい`, `Vâng`) — vốn là phần lớn một cuộc
  họp tiếng Nhật.

**`VAD_MIN_SILENCE_MS`** — Phải **lớn hơn** `FINALIZE_PAUSE_MS`. Nếu không,
đoạn sẽ đóng trước khi buffer kịp nhìn thấy khoảng lặng gây ra việc đóng đó,
và câu cuối không bao giờ được chốt.

**`VAD_SPEECH_PAD_MS`** — Silero cần vài khung mới chắc chắn, nên khi nó
chắc thì phụ âm đầu từ đã trôi qua. 256 ms giữ lại phần đó. Đặt 0 thì
Whisper nghe "ôm nay" thay vì "hôm nay".

---

## 2. Buffer — cắt thành câu

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `FINALIZE_PAUSE_MS` | `400` | Im lặng dài hơn ngần này là hết câu |
| `FINALIZE_MAX_DURATION_MS` | `7_000` | Nói liên tục quá lâu thì cắt |
| `PARTIAL_INTERVAL_MS` | `600` | Bao lâu cập nhật chữ mờ một lần |
| `SPLIT_SEARCH_MS` | `500` | Tìm chỗ yên tĩnh nhất trong khoảng này để cắt |
| `PARTIAL_WINDOW_SECONDS` | `4.0` | Chữ mờ chỉ giải mã ngần này giây cuối |

**`FINALIZE_MAX_DURATION_MS`** — Người nói không dừng thì vẫn phải cắt, nếu
không người xem ngồi nhìn chữ mờ mãi. Cắt không rơi đúng 7000 ms mà lùi về
khung 32 ms yên tĩnh nhất trong `SPLIT_SEARCH_MS` gần đó, để không cắt giữa
từ — Whisper biến nửa từ thành một từ khác.

**`PARTIAL_WINDOW_SECONDS`** — Chữ mờ giải mã lại **toàn bộ** câu đang mở mỗi
`PARTIAL_INTERVAL_MS`, nên một câu 7 giây bị giải mã 11 lần ở các độ dài
0.6, 1.2 … 7.0 giây — khoảng 45 giây audio cho 7 giây tiếng nói.

> **Đo được (10 phút họp thật):** giải mã partial tốn **97.8 giây**, so với
> 21.6 giây cho *toàn bộ* câu đã chốt cộng lại. Một lần chạm 4.7 giây trong
> khi câu chậm nhất chỉ 0.4 giây. Tất cả đều chạy trên đúng luồng đọc socket.

Đặt giới hạn 4 giây:

| | trước | sau |
| --- | --- | --- |
| lag tệ nhất | 8448 ms | **1398 ms** |
| partial chậm nhất | 4.7 s | 3.9 s |
| tổng `partial_asr` | 97.8 s | 88.7 s |

Giới hạn chặn **trường hợp xấu nhất**; nó gần như không giảm trường hợp phổ
biến, vì phần lớn câu trong họp thật chỉ dài 1–3 giây và không bao giờ chạm
tới giới hạn.

- Giảm xuống: chữ mờ chỉ hiện đoạn rất ngắn đang nói, khó theo dõi.
- Tăng lên: lag tăng trở lại ở những câu dài.
- Muốn giảm **tổng** chi phí thì tăng `PARTIAL_INTERVAL_MS`, không phải chỉnh
  số này. Nhưng với ASR streaming, `PARTIAL_INTERVAL_MS` cũng là nhịp của phép
  so khớp: chữ chỉ được chốt khi hai lần giải mã liên tiếp đồng ý, nên nhịp
  thưa hơn là chữ chốt chậm hơn.

**Hangover.** Câu chốt vì ngắt mang theo `VAD_MIN_SILENCE_MS` trừ một khung
(≈ 480 ms) khoảng lặng ở cuối — VAD đã chuyển tiếp nó trong lúc chờ chắc chắn.
Buffer ghi độ dài đó vào `Utterance.trailing_silence_ms`, và ASR không giải mã
quá `speech_end + ASR_STREAM_FINAL_POST_ROLL_SECONDS`.

> **Đo được:** chữ mờ `"thì bạn vẫn nhờ mình fix à?"`, câu chốt
> `"thì bác vẫn nhờ mình thích à? Không, biết thôi đẹp."` — vế sau được viết
> lên 400 ms không ai nói. Cắt cố định 468 ms ở **mọi** câu thì hỏng câu bị cắt
> vì quá dài (giữa chừng tiếng nói): 12 câu rỗng thêm trên 268. Nên chỉ trừ
> hangover ở câu thật sự có hangover.

---

## 3. Lọc nhiễu (AST)

**Tầng này mặc định TẮT.** Bật bằng `ENABLE_NOISE_FILTER=1`.

Lý do không phải chi phí, mà là **nó không bỏ được gì**.

> **Đo được (hai cuộc họp thật trong venv, AST trên CPU):**
>
> | | lần 1 | lần 2 |
> | --- | --- | --- |
> | audio | 593.6 s | 718.2 s |
> | utterance | 119 | 119 |
> | **bỏ được** | **0** | **0** |
> | `noise` | 155.8 s | 156.4 s |
> | `slowest sentence` | 2.4 s | 2.1 s |
>
> **237 utterance, không bỏ được câu nào.** Chi phí cố định **1.31 giây mỗi
> utterance** — không phụ thuộc độ dài câu, vì AST luôn xử lý spectrogram
> 10.24 giây dù câu chỉ 3.6 giây. Tức 22–26% luồng đọc socket.
>
> Tắt nó thì `slowest sentence` xuống **0.4 s**. Đó là độ trễ người dùng cảm
> nhận, nhỏ đi sáu lần.

Những lần duy nhất tầng này từng bỏ câu — `Music 0.82`, `Beatboxing 0.78` —
đều là **mảnh vụn do lỗi cắt câu theo giọng sinh ra**, thứ đã gỡ bỏ. Nó chưa
bao giờ bỏ được tiếng ồn thật trên audio của dự án này.

Việc nó làm cũng đã có lớp khác làm: các câu bịa mà nó nhắm tới đều bị
[`server/data/`](../server/data/README.md) và ba ngưỡng thống kê của ASR chặn.

**Đường GPU chưa đo lại trong venv.** Lần đổ ở cuDNN là từ conda base, nên có
thể do môi trường. Muốn có con số cho đủ bộ:

```bash
NOISE_DEVICE=cuda ENABLE_NOISE_FILTER=1 python3.11 -m uvicorn server.app:app     --host 0.0.0.0 --port 8000
```

Nhưng nó **không đổi quyết định**: rẻ đi cũng không làm một tầng chưa từng kích
hoạt trở nên đáng bật. Chỉ bật lại khi họp thật có tiếng ồn thật — gõ phím, ho,
quạt — và tầng này bỏ được chúng.

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ENABLE_NOISE_FILTER` | `False` | Biến môi trường `=1` để bật tầng này |
| `NOISE_MIN_SPEECH_SCORE` | `0.2` | Dưới ngưỡng này mới xét bỏ |
| `NOISE_MIN_NOISE_SCORE` | `0.3` | ...và chỉ bỏ khi model chắc chắn nghe thấy thứ khác |
| `NOISE_WINDOW_SECONDS` | `10.0` | Cửa sổ AST đọc mỗi lần |
| `AST_MODEL_ID` | `MIT/ast-finetuned-…` | Đổi qua biến môi trường `AST_MODEL_ID` |
| `NOISE_DEVICE` | `cpu` | Đặt `cuda` để thử GPU — xem cảnh báo dưới |

**Cần cả hai điều kiện, và đó là điểm mấu chốt.** Điểm tiếng nói thấp một
mình không phải bằng chứng.

> **Đo được:** tiếng gõ phím thật đạt 0.87, tiếng ho thật 0.83. Audio mà
> model không xếp được vào đâu chỉ đạt khoảng 0.1. Dưới `NOISE_MIN_NOISE_SCORE`
> nghĩa là "không biết", và không biết thì giữ.

Từng có một câu tiếng Nhật thật bị xoá vì so sánh hai điểm số gần 0 với nhau.

> **Đo được:** AST chấm **0.00** cho câu chêm tiếng Nhật rất ngắn mang sắc
> thái biểu cảm ("à ra vậy"). Đó là tiếng nói thật.

Vì vậy **đừng siết `NOISE_MIN_SPEECH_SCORE` lên** mà không chạy lại real test
với dữ liệu có câu chêm ngắn.

### `NOISE_DEVICE` — vì sao mặc định là CPU

`DESIGN.md` vốn định cho AST chạy CPU để nhường VRAM cho Whisper và vLLM.

> **Gặp thật trên pod:** AST trên GPU đổ ở đường attention của cuDNN —
> `RuntimeError: cuDNN Frontend error: No valid execution plans built` — rồi
> lần gọi kế tiếp vào đúng tầng đó **giết tiến trình**:
> `Segmentation fault (core dumped)`.
>
> Mốc thời gian nói rõ thủ phạm. Lỗi cuDNN lúc `01:31:40.447`; ECAPA chạy
> `01:31:40.478`; Whisper chạy bình thường tới `01:31:45.988`; rồi chết. CUDA
> **không** hỏng ngay — thứ giết tiến trình là **vào lại tầng đã hỏng**.

Vì vậy một lỗi có chữ `cuda`, `cudnn`, `cublas` hay `out of memory` làm tầng
đó **tắt ngay lần đầu**, không chờ đủ ba lần như lỗi thường. Ba lần với lỗi
thiết bị là ba cơ hội để segfault.

Muốn thử lại trên GPU thì đặt `NOISE_DEVICE=cuda`. Đo bằng mục `noise` trong
`stages` ở dòng tổng kết — đó là số giây **luồng đọc socket** bỏ ra cho tầng
này trong cả cuộc họp. Trên GPU nó dưới 1 giây cho 3 phút họp; nếu trên CPU nó
vọt lên vài chục giây thì cái giá quá đắt, và nên tắt hẳn tầng lọc nhiễu chứ
không chạy nó trên CPU.

---

## 4. Tách chồng lấn (DSP)

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `OVERLAP_GATE_BELOW_DB` | `12.0` | Chặn những gì thấp hơn giọng chính ngần này |
| `OVERLAP_LEVEL_PERCENTILE` | `90.0` | Percentile của đường bao đỉnh dùng làm "mức giọng" |
| `OVERLAP_ENVELOPE_MS` | `20` | Cửa sổ tính đường bao |
| `OVERLAP_GATE_RATIO` | `4.0` | Độ dốc của cổng |
| `OVERLAP_GATE_ATTACK_MS` | `2.0` | Cổng đóng nhanh cỡ nào |
| `OVERLAP_GATE_RELEASE_MS` | `120.0` | Cổng mở lại chậm cỡ nào |
| `OVERLAP_COMPRESSOR_ABOVE_DB` | `3.0` | Nén phần cao hơn mức giọng ngần này |
| `OVERLAP_COMPRESSOR_RATIO` | `3.0` | Tỉ lệ nén |
| `OVERLAP_COMPRESSOR_ATTACK_MS` | `5.0` | Nén bắt đầu nhanh cỡ nào |
| `OVERLAP_COMPRESSOR_RELEASE_MS` | `120.0` | Nén nhả ra chậm cỡ nào |
| `OVERLAP_MIN_LEVEL_DBFS` | `-55.0` | Nhỏ hơn thì bỏ qua, không xử lý |
| `DISABLE_OVERLAP` | `False` | Biến môi trường `=1` để tắt tầng này |

**Đây không phải tách nguồn.** Cổng nhiễu không tách được hai giọng; nó chỉ
hạ những gì nằm thấp hơn hẳn giọng đang át.

**Ngưỡng lấy theo đỉnh, không theo RMS** — và điều này quan trọng hơn vẻ ngoài
của nó. Một utterance mang theo khoảng lặng hangover và mọi quãng nghỉ giữa
từ, nên RMS toàn cục bị kéo tụt rất sâu.

> **Đo được:** khung 20 ms trung vị nằm **28 dB dưới** mức giọng. Với giọng
> phụ thấp hơn 20 dB: ngưỡng theo RMS chỉ hạ được **0.1 dB**, ngưỡng theo
> đỉnh hạ **24 dB**, giọng chính không suy hao trong cả hai trường hợp.

Lý do: bộ dò của `pedalboard` so ngưỡng với **đỉnh** tín hiệu, không phải RMS.

**`OVERLAP_GATE_RELEASE_MS`** và **`OVERLAP_COMPRESSOR_RELEASE_MS`** — Đặt
ngắn sẽ cắt cụt đuôi từ. 120 ms đủ dài để giữ nguyên phần đuôi.

**`OVERLAP_COMPRESSOR_ABOVE_DB`** — Bộ nén chỉ ghìm phần đỉnh cao hơn mức
giọng; trên tiếng nói thật chúng chỉ nhô lên khoảng 3 dB, nên ngưỡng thấp
hơn sẽ bóp bẹp chính giọng nói.

### `DISABLE_OVERLAP` — chưa ai đo tầng này giúp gì cho ASR

Người dùng thật phản ánh: **chữ mờ đôi khi chính xác hơn câu đã chốt**. Đúng
một phần, và có ba khác biệt giữa hai lần giải mã:

| | chữ mờ | câu đã chốt |
| --- | --- | --- |
| audio | **thô** | **đã qua gate + compressor** |
| cửa sổ | 4 giây cuối | cả câu, tới 7 giây |
| beam | 1 | 5 |

Beam 5 chỉ tốt hơn. Hai cái còn lại đều có thể làm xấu đi.

> **Thấy trong log thật:** final làm hỏng đúng đoạn mà partial đã nghe ra —
> `cái X23 cái timet` thành `cái hay là ba cái tên biết`, `y2 x2.3` thành
> `i2x23`. Đó là dấu hiệu của gate cắt mất âm tiết yếu.
>
> Nhưng chiều ngược lại cũng có: một câu tiếng Nhật mà partial trôi dần
> (`YAM` → `山本` → `皆`) thì final lại chốt đúng `YAM`.

Một phần cảm nhận là **hiệu ứng chọn lọc**: partial làm mới mỗi 600 ms, mắt
người nhớ bản đúng nhất; final chỉ có một lần.

Phần còn lại thì không. Chú thích trong `_analyse` viết *"Shaping helps the
ASR and nothing else"*, nhưng điều đó **chưa từng được đo trên độ chính xác
phiên âm**. Cái đã đo là gate làm mất 0.06 cosine của voiceprint — và đó chính
là lý do tầng người nói đọc audio thô.

Tầng chồng lấn **chỉ có một khách hàng là ASR**, nên tắt nó tức là cho Whisper
ăn audio thô:

```bash
DISABLE_OVERLAP=1 python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

> **Đã thử đo và phép đo VÔ HIỆU.** Chạy cùng một đoạn 120 giây hai lần, một
> lần có `DISABLE_OVERLAP=1`. Kết quả 22 câu so với 21, 14 câu giống hệt.
>
> Nhưng `/health` sau đó cho thấy `overlap_resolver_loaded: false` với lý do
> `pedalboard is not installed` — **cả hai lần chạy đều không có tầng này**.
> Phép đo so không-có-tầng với không-có-tầng. Khác biệt 3–2 đến từ hai lần
> chạy lệch nhau 2.2 giây nên ranh giới câu khác nhau, không từ gate.

**Tầng chồng lấn vẫn chưa được đo.** Trước khi đo lại, kiểm tra
`overlap_resolver_loaded` trong `/health` phải là `true` — nếu không thì
`pip install -r server/requirements.lock.txt`.

Rồi chạy cùng một đoạn ghi âm hai lần, có và không có `DISABLE_OVERLAP=1`, so
hai file biên bản trong `recordings/`. Chọn đoạn có **cả hai tình huống**:
giọng chồng lấn, và người nói một mình liền mạch.

---

## 5. Nhận dạng người nói

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `SPEAKER_MATCH_THRESHOLD` | `0.30` | Cosine trên ngưỡng này là cùng một người |
| `SPEAKER_MIN_DURATION_MS` | `600` | Ngắn hơn thì gắn `Speaker_unknown` |
| `SPEAKER_MAX_SPEAKERS` | `12` | Quá số này thì ngừng tạo người mới |
| `SPEAKER_CENTROID_MOMENTUM` | `0.7` | Giữ lại bao nhiêu phần voiceprint cũ |

**`SPEAKER_MATCH_THRESHOLD`** là thông số nhạy nhất trong cả tầng này.

> **Đo được (ba bản ghi một người, mỗi bản 45 giây, hai trong ba cùng giới):**
>
> | | khoảng cosine |
> | --- | --- |
> | cùng giọng | 0.361 … 0.994 |
> | khác giọng | −0.129 … 0.232 |
>
> Bất kỳ ngưỡng nào trong (0.232, 0.361) đều tách được.

Chọn 0.30 vì nó nằm **giữa**, không sát mép nào. Cửa sổ này chỉ hẹp đi khi có
thêm người: thêm giọng thứ ba cùng giới với giọng thứ nhất đã đẩy trần
khác-giọng từ 0.199 lên 0.232 và sàn cùng-giọng từ 0.394 xuống 0.361.

Mặc định của SpeechBrain là 0.25 — nằm trong cửa sổ nhưng chỉ cách trần
khác-giọng 0.018. Thêm một cặp giọng giống nhau nữa là nó gộp hai người
làm một.

- Giảm xuống: hai người bị gộp làm một. Rất khó phát hiện khi đọc log.
- Tăng lên: một người bị tách thành nhiều `Speaker_0x`.

**`SPEAKER_MIN_DURATION_MS`** — Câu ngắn hơn 600 ms không đủ chất giọng.
Gắn `Speaker_unknown` chứ **không đoán theo người nói trước**: câu chêm ngắn
thường là của người đang *nghe*, nên phép đoán đó sai đúng vào chỗ nó hấp dẫn
nhất.

### Gom cụm lại — sửa nhãn đã gán

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `SPEAKER_RECLUSTER_EVERY` | `15` | Cứ bấy nhiêu câu thì gom cụm lại cả cuộc họp |
| `SPEAKER_RECLUSTER_MAX` | `300` | Số voiceprint giữ lại; chi phí tăng theo bình phương |

`SpeakerIdentifier` phải trả lời **ngay**, từ một voiceprint, dựa trên những
gì đã nghe **tính đến lúc đó**. Hai thứ nó sai mà không ngưỡng nào chữa được:

- câu trả lời **phụ thuộc thứ tự** cuộc họp diễn ra. Câu đầu tiên không có gì
  để so nên luôn tạo người mới; các câu sau so với centroid đã dịch chuyển.
- trả lời rồi là xong. Một nhầm lẫn ở phút đầu sống sót qua mười phút bằng
  chứng phía sau.

> **Đo được (họp thật, hơn 4 phút):** mọi câu đều ra `Speaker_01`. Cơ chế là
> `_update` chạy với **mọi** lần khớp, kể cả lần vừa đủ 0.31, và kéo centroid
> 30% về phía đó. Gán một utterance trộn giọng vào một người làm centroid
> người đó pha thêm, pha thêm thì khớp được nhiều người hơn. Vòng lặp dương,
> không có gì kéo ngược.

Gom cụm nhìn **cả cuộc họp cùng lúc** nên không phụ thuộc thứ tự, và sửa được
nhãn đã lỡ gán. Thuật toán là agglomerative liên kết trung bình trên cosine,
cắt tại đúng `SPEAKER_MATCH_THRESHOLD` — cùng con số đã đo, áp lên đúng loại
voiceprint cả câu mà nó được đo trên đó. Liên kết **trung bình** chứ không
phải gần nhất: một câu ở ranh giới không được phép nối hai người thành một.

Nhãn được chọn để **đứng yên**: mỗi cụm giữ cái tên mà phần lớn thành viên của
nó đang mang, nên một lần sửa chỉ dịch chuyển vài câu sai chứ không đổi tên
tất cả. Khi hai cụm cùng đòi một tên, cụm lớn giữ.

Server gửi lại message `speakers` chứa `{sentence_id: speaker_id}`, chỉ những
hàng đổi. Client khoá hàng theo `sentence_id` nên sửa tại chỗ.

- `SPEAKER_RECLUSTER_EVERY` giảm: sửa nhanh hơn, tốn CPU trên **luồng đọc
  socket** thường xuyên hơn. Thời gian đo được nằm ở `stages` mục `recluster`.
- `SPEAKER_RECLUSTER_MAX` tăng: cụm chính xác hơn với họp dài, nhưng chi phí
  gom cụm tăng theo **bình phương**.

| `SPEAKER_RECLUSTER_THRESHOLD` | `0.30` | Liên kết trung bình dưới ngưỡng này thì thôi gộp |
| `SPEAKER_RECLUSTER_CONFIRMATIONS` | `2` | Số lần gom liên tiếp phải cùng đề xuất một nhãn mới |
| `SPEAKER_RECLUSTER` | `False` | Biến môi trường `=1` để **gửi** nhãn đã sửa; mặc định chỉ đo |

**Mặc định chỉ đo, không gửi.**

> **Đo được (họp thật 30 phút, 4 người nói, 09-16):**
>
> | | nhãn | phân bố câu |
> | --- | --- | --- |
> | bộ khớp trực tiếp | 5 | 113 / 91 / 67 / 18 / 1 — **4 người chính** |
> | sau gom cụm (140 câu bị đổi nhãn) | 11 | 183 / 66 / 16 / 12 + bảy cụm 1–3 câu |
>
> Gom cụm **gộp hai người thật làm một** và đẻ ra cụm lẻ từ các câu ngoại lai.
> Mọi lần gộp bị từ chối nằm ở 0.218–0.299, sát ngưỡng — hạ ngưỡng chỉ làm gộp
> thêm.

Gom cụm vẫn chạy và dòng tổng kết in kết quả của nó cạnh nhãn trực tiếp
(`measured only ... sized [...] ... would have moved`), để đo tiếp mà không
làm nhảy tên trên màn hình.

> **Đo được (họp thật 30 phút, trước hai luật dưới):** `22 speakers after 21
> reclustering runs, 329 labels corrected` trên 313 câu — hơn một lần sửa mỗi
> câu, tức tên người nhảy liên tục trên màn hình.

Hai luật sửa chuyện đó:

- Gom cụm **không để quá `SPEAKER_MAX_SPEAKERS` cụm** — bộ khớp trực tiếp
  vốn đã tuân, gom cụm thì không nên tự đẻ ra 22 nhãn. Những lần gộp bị ép dưới
  ngưỡng vì luật này được đếm (`merges forced by the speaker cap`).
- Một nhãn chỉ đổi khi **`SPEAKER_RECLUSTER_CONFIRMATIONS` lần gom liên tiếp**
  cùng đề xuất nó. Đề xuất không lặp lại ở lần sau thì bị quên.

**`SPEAKER_RECLUSTER_THRESHOLD` chưa được đo.** `0.30` được đo cho phép so
**một cặp**; trung bình giữa hai **cụm** là đại lượng khác, và 22 người ở trên
là dấu hiệu nó đang tách quá tay. Mỗi lần gom ghi log **lần gộp tốt nhất bị từ
chối**, và dòng tổng kết in decile của các giá trị đó:

```
speakers: 5 after 20 reclustering runs, 12 labels corrected,
  0 merges forced by the speaker cap, refused merges [0.21, 0.24, ...]
```

Ngưỡng nên nằm **dưới** phần lớn các giá trị bị từ chối giữa hai người thật và
**trên** các giá trị giữa hai nửa của cùng một người. Đọc chúng trên một cuộc
họp mà bạn biết có bao nhiêu người trước khi đổi số này.

**Voiceprint lấy từ audio thô**, chưa qua tầng chồng lấn.

> **Đo được:** gate trước khi trích voiceprint làm mất **0.06** cosine
> cùng-giọng (0.677 thô so với 0.616 đã gate). Cổng cắt cả âm tiết nhỏ trong
> câu, mà âm tiết nhỏ vẫn mang chất giọng.

---

## 5b. Cắt câu khi đổi người nói — KHÔNG CÓ Ở NHÁNH NÀY

Đã làm và đo ở nhánh `main` (`server/pipeline/speaker_change.py`), và đã tắt
ở đó. Nhánh này không mang nó sang. Phần dưới ghi lại vì sao, để không ai làm
lại theo cùng cách.

VAD chỉ đóng đoạn sau `VAD_MIN_SILENCE_MS` (500 ms) im lặng. Người sau nói
tiếp nhanh hơn thế thì **hai giọng nằm chung một utterance**, và utterance đó
chỉ được một voiceprint, một lần nhận dạng ngôn ngữ, một lần ASR.

> **Đo được (họp thật một tiếng, 30 phút đầu):** hai người nói tiếng Việt
> cùng ra `Speaker_01`. 3–5 lần một câu tiếng Việt ngắn (5–7 chữ) mất hẳn:
> nó bị nuốt vào utterance của câu tiếng Nhật nối ngay sau, trôi khỏi cửa sổ
> partial 4 giây, và không bao giờ được chốt.

Cách làm: giây đầu utterance là **mốc**, mỗi nhịp partial so giây gần nhất với
mốc, lệch quá ngưỡng thì cắt ngay trước cửa sổ lệch.

### Vì sao tắt

> **Đo được (họp thật, 175 giây audio, 137 phép so):**
>
> | | |
> | --- | --- |
> | số nhát cắt | **73 / 137 phép so — 53%** |
> | decile của cosine | 0.017, 0.092, 0.127, 0.166, 0.207, 0.244, 0.258, 0.283, 0.332, 0.379, 0.558 |
>
> Phân bố **liền một mạch, một cụm duy nhất, không có khoảng trống**. Ngưỡng
> 0.25 rơi đúng trung vị.

Đây không phải chọn sai số. Cả cách làm dựa trên giả định có **hai cụm** để
tách ra, và phép đo nói là không có. Cửa sổ 1 giây không phân biệt được giọng
trên audio họp.

Cùng một đoạn audio, hai thước đo khác hẳn nhau:

| utterance | cosine cửa sổ 1 giây | cosine cả câu (`SpeakerIdentifier`) |
| --- | --- | --- |
| 34 | 0.122 | 0.765 |
| 46 | 0.017 | 0.640 |
| 74 | 0.193 | 0.752 |
| 80 | 0.131 | 0.741 |

0.017 là gần như vuông góc — hai giọng người khác nhau cũng hiếm khi rời nhau
đến thế (dải khác-giọng đo được cho cả câu là −0.129…0.232).

**Hậu quả khi bật**, đo trên cùng lần chạy: 93 utterance trong 175 giây, tức
**một câu mỗi 1.9 giây**. Kéo theo:

- registry chạm trần `SPEAKER_MAX_SPEAKERS` (12) sau 21 utterance, và từ đó
  mọi câu bị ép gán vào centroid gần nhất **bất kể ngưỡng** — log hiện
  `similarity 0.219` cho một câu được coi là khớp
- mảnh vụn gần-im-lặng làm Whisper bịa thêm, và bị loại `no speech` nhiều hơn
- **mất câu nhiều hơn hẳn** so với khi tắt

Nghĩa là bật tính năng này làm hỏng đúng ba thứ nó định sửa.

### Muốn bật lại thì phải đo gì trước

Chạy `server/tests_real/` với bản ghi **một người nói duy nhất**, đủ dài, rồi
dựng hai phân bố: cùng-giọng (hai cửa sổ trong cùng bản ghi) và khác-giọng
(cửa sổ từ hai bản ghi khác nhau). Chỉ khi hai phân bố **tách rời** thì mới có
ngưỡng để đặt. Cần quét cả `SPEAKER_CHANGE_WINDOW_MS` — 1000 ms chỉ hơn
`SPEAKER_MIN_DURATION_MS` (600 ms) một chút, và chính tầng nhận dạng người nói
đã coi 600 ms là quá ngắn để tin.

Cũng cần kiểm tra **mốc**: giây đầu utterance chứa `VAD_SPEECH_PAD_MS`
(256 ms) đệm trước và phần chớm tiếng, nên có thể là giây ít đại diện nhất
trong cả câu. Nếu mốc là thủ phạm thì đổi mốc rẻ hơn nhiều so với bỏ cách làm.

Chi phí khi bật: **một lần embed ECAPA mỗi 600 ms**. Mốc chỉ embed một lần cho
mỗi utterance.

---

## 6. Nhận dạng ngôn ngữ

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LID_LANGUAGES` | `("vi", "ja")` | Hai ngôn ngữ cuộc họp có thể chứa |
| `LID_MIN_MARGIN` | `0.30` | Chênh lệch tối thiểu để dám kết luận |
| `LID_MIN_DURATION_MS` | `600` | Ngắn hơn thì không đoán |

Model biết 107 ngôn ngữ nhưng **chỉ đọc điểm của đúng hai ngôn ngữ trong
`LID_LANGUAGES`**, rồi chuẩn hoá lại giữa hai điểm đó. Để nó tự do chọn thì
tiếng Nhật hay bị trả về là Hàn hoặc Trung — hợp lý với model, vô dụng với ta,
vì việc duy nhất tầng sau làm là **ép `language` của Whisper**.

**Ép sai ngôn ngữ không báo lỗi.** Whisper vẫn trả về văn bản trôi chảy, tự
tin, và sai, rồi tầng dịch dịch trung thành cái vô nghĩa đó.

> **Đo được:** khi để Whisper tự nhận diện trên câu ngắn, nó trả về Thuỵ Điển
> (0.66), Phần Lan (0.50), Trung (0.18), Anh (0.29) cho một cuộc họp Việt–Nhật.

Nên khi LID không quyết được, hệ thống **không** để Whisper tự đoán mà dùng
ngôn ngữ cuối cùng cuộc họp đã xác định chắc chắn. Sai nhiều nhất là 50 % và
chỉ tại thời điểm đổi ngôn ngữ; Whisper đoán thì sai 100 %.

- `LID_MIN_MARGIN` giảm: kết luận liều hơn, ép sai ngôn ngữ nhiều hơn.
- Tăng: rơi về ngôn ngữ trước nhiều hơn, an toàn hơn nhưng chậm nhận đổi ngôn ngữ.

### Hai ngôn ngữ trong một câu — cắt ra

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `LANGUAGE_SPLIT` | `True` | Biến môi trường `LANGUAGE_SPLIT=0` để tắt |
| `LANGUAGE_SPLIT_PROBE_MS` | `1000` | Độ dài mỗi lần thăm dò |
| `LANGUAGE_SPLIT_EDGE_MS` | `300` | Lùi vào từ mỗi đầu trước khi thăm dò |
| `LANGUAGE_SPLIT_MIN_MARGIN` | `0.50` | Biên LID tối thiểu để một lần thăm dò được tính |
| `LANGUAGE_SPLIT_MIN_PART_MS` | `1200` | Nửa ngắn nhất được phép, tính bằng **tiếng nói** |
| `LANGUAGE_SPLIT_REVIEW_MS` | `2500` | Cửa sổ thăm dò lại mỗi nửa, lấy ở giữa nửa |
| `LANGUAGE_SPLIT_SNAP_MS` | `200` | Cửa sổ tìm khung im nhỏ nhất |
| `LANGUAGE_SPLIT_MAX_STEPS` | `3` | Số bước tìm nhị phân |

> **Đo được (họp thật 10 phút, 119 câu):** chữ mờ và câu chốt bất đồng ngôn ngữ
> ở **8 câu**. Đọc lại tám chỗ đó: **4 thật sự mất hẳn một lượt nói** — không
> phải dịch sai, mà không còn trong bản ghi.

VAD đóng đoạn sau 500 ms im lặng, người ta đáp nhau nhanh hơn thế, nên câu trả
lời bằng ngôn ngữ kia nằm chung utterance; LID phải chọn một, Whisper bị ép
theo cho cả đoạn.

Cách làm: thăm dò **gần** đầu và cuối câu. Cùng ngôn ngữ, hoặc một đầu chưa đủ
chắc, thì **không cắt**. Khác nhau chắc chắn thì tìm nhị phân ranh giới, bắt vào
khung im nhất trong khoảng tìm được, rồi **thăm dò lại hai nửa sẽ gửi đi** —
phải khác nhau chắc chắn mới cắt. Mỗi nửa mang luôn ngôn ngữ của nó xuống ASR.

Lịch sử của từng số, đo trên cùng một cuộc họp 30 phút:

- **Thăm dò ngay mép** (phiên bản đầu): 76 lần cắt, **26/41** lần cắt nhìn thấy
  được cho hai nửa **cùng** một ngôn ngữ, 34/41 để lại nửa sau ≤ 25 ký tự. Mép
  đầu là pre-roll, mép cuối là hangover — hai giây kém đại diện nhất. Nên
  `LANGUAGE_SPLIT_EDGE_MS`, thăm dò lại hai nửa, và biên `0.50` thay vì `0.30`
  của LID: cắt thừa sinh mảnh vụn mà Whisper lấp bằng câu bịa, cắt thiếu chỉ
  mất một lượt nói.
- **Sau đó:** 76 → **27** lần cắt, cặp nhìn thấy 41 → 9, cặp cùng ngôn ngữ
  26 → 3. Còn lại một kiểu: nửa sau là `ありがとうございました` hoặc
  `Alright, they will` — nửa đuôi 1200 ms chỉ có tối đa 700 ms tiếng nói vì
  500 ms cuối là hangover. Nên sàn bên phải **cộng thêm hangover** của chính câu
  đó (0 với câu cắt vì quá dài), và phép thăm dò lại chỉ đọc **giữa** mỗi nửa,
  tối đa `LANGUAGE_SPLIT_REVIEW_MS` — không có giới hạn đó câu chậm nhất đi từ
  0.7 s lên 1.2 s.
- **Mảnh vụn:** một phiên bản sớm trả về ranh giới cách đầu câu vài trăm ms, và
  nửa kia ra `Các bạn nhớ đăng ký kênh để ủng hộ kênh của mình nhé.` 31 ms sau
  câu trước. **Mảnh quá ngắn để phiên âm là mảnh Whisper lấp vào.**

Chi phí: **2 lần thăm dò** cho câu thường, khoảng 7 khi phải cắt.

Mọi lần từ chối được đếm theo lý do, và dòng tổng kết in ra:

```
language splits: 27 of 330 utterances held two languages
  (280 one language, 15 undecided at an end, 11 too short,
   4 would leave a fragment, 10 refused on review), 760 probes
```

**Đọc `undecided` trước.** Trên probe 600 ms biên thật đo được chỉ
**0.12–0.13** — đó là lý do probe giờ dài 1 giây. Nếu `undecided` vẫn lớn thì
nới probe, không hạ `LANGUAGE_SPLIT_MIN_MARGIN`. Nếu `one_language` lớn mà
chữ mờ và câu chốt vẫn hay bất đồng (`language_flips`) thì tín hiệu sai bản
chất và phải nghĩ lại.

- `LANGUAGE_SPLIT_MAX_STEPS` tăng: ranh giới chính xác hơn, thêm một lần thăm dò
  mỗi bước.
- `LANGUAGE_SPLIT_MIN_PART_MS` giảm: bắt được lượt nói ngắn hơn, và mảnh vụn
  quay lại.

---

## 7. ASR (Whisper)

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ASR_MODEL` | `large-v3` | Checkpoint faster-whisper |
| `ASR_BEAM_SIZE_PARTIAL` | `1` | Chữ mờ giải mã tham lam |
| `ASR_BEAM_SIZE_FINAL` | `5` | Câu chốt được beam search |
| `ASR_NO_SPEECH_THRESHOLD` | `0.6` | Coi là im lặng — **cần thêm điều kiện logprob** |
| `ASR_LOG_PROB_THRESHOLD` | `-1.0` | Dưới ngưỡng này là đoán mò |
| `ASR_SHORT_UTTERANCE_MS` | `600` | Ngắn hơn thì `no_speech_prob` **một mình** đủ để loại |
| `ASR_NO_SPEECH_CERTAIN` | `0.95` | Trên ngưỡng này thì `no_speech_prob` một mình đủ để loại ở mọi độ dài |
| `ASR_PROMPT_MAX_CHARS` | `200` | Giới hạn `initial_prompt` dựng từ `vocabulary.txt` |
| `ASR_PROMPT_ON_PARTIALS` | `False` | Biến môi trường `=1` để chữ mờ cũng nhận prompt — chỉ để đo |
| `MEETING_DATA_DIR` | `server/data` | Chỗ đọc danh sách chặn/giữ và từ vựng |
| `ASR_MAX_COMPRESSION_RATIO` | `2.4` | Trên ngưỡng này là đang lặp |
| `ASR_CONDITION_ON_PREVIOUS` | `False` | **Đừng bật** |

**`ASR_CONDITION_ON_PREVIOUS = False`** — Bật lên là Whisper lấy câu trước làm
prompt cho câu sau, đúng cơ chế biến **một câu bịa thành cả đoạn bịa**.

### Streaming: chốt dần từng từ

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ASR_STREAM_MIN_AGREEMENT` | `2` | Số lần giải mã liên tiếp phải cùng thấy một từ |
| `ASR_STREAM_COMMIT_MARGIN_SECONDS` | `1.0` | Không chốt từ nằm gần mép âm thanh mới nhất hơn ngần này |
| `ASR_STREAM_WORD_TOLERANCE_SECONDS` | `0.45` | Hai lần giải mã đặt cùng một từ lệch nhau tối đa ngần này |
| `ASR_STREAM_HISTORY` | `5` | Số lần giải mã giữ lại để so |
| `ASR_STREAM_LANGUAGE_VOTES` | `2` | Số cửa sổ liên tiếp phải chắc chắn cùng một ngôn ngữ để chốt, hoặc đổi, ngôn ngữ của chữ mờ |
| `ASR_LANGUAGE_OVERRIDE_MARGIN` | `0.50` | Biên LID cần có để lấn chữ mờ: giải mã lại câu chốt, hoặc ép một cửa sổ khi ngôn ngữ chưa chốt |
| `ASR_STREAM_FINAL_OVERLAP_SECONDS` | `1.2` | Audio đã chốt được giải mã lại phía trước phần đuôi, làm ngữ cảnh |
| `ASR_STREAM_FINAL_POST_ROLL_SECONDS` | `0.20` | Audio giữ lại sau chỗ tiếng nói kết thúc |

Chữ mờ cũ (cửa sổ trượt 4 giây, thay toàn bộ mỗi 600 ms) và câu chốt là hai lần
giải mã độc lập, nên câu chốt có thể mất thứ chữ mờ đã nghe đúng
(`"về Solution"` → `"về sau lưu sinh"`), và chữ mờ tự nó giật: đầu câu trôi khỏi
cửa sổ.

> **Đo được trên nhật ký client của cùng một cuộc họp:**
>
> | | cửa sổ trượt | streaming (09-10) |
> | --- | --- | --- |
> | phần chữ trên màn hình còn giữ sau mỗi lần cập nhật | 34% | **64%** |
> | lần cập nhật xoá hơn nửa dòng đang đọc | 65% | **33%** |
> | câu tiếng Nhật bị cách từng chữ | 9/180 | **144/163** |
> | câu có một từ lặp đôi | 16/295 | **131/294** |

Streaming làm đúng việc của nó — chữ đứng yên — nhưng lần chạy đó có hai lỗi
hiển thị, cả hai đã sửa và có test:

- **Cách từng chữ tiếng Nhật:** văn bản được dựng bằng cách nối các từ bằng dấu
  cách. Giờ dùng nguyên chuỗi từ của Whisper (đã có khoảng trắng đầu từ với
  tiếng Việt, không có với tiếng Nhật).
- **Từ lặp đôi** (`AMD AMD`, `đồ đồ`, `朝、 朝、`): phần đuôi được lọc theo
  *cuối* từ, nên bản sao của từ chốt cuối — đặt lệch một chút — lọt qua. Giờ xét
  theo *điểm giữa* từ, và bản sao nằm ngay chỗ nối bị bỏ.

Hai điều nữa đổi so với lần chạy đó:

- Phần đuôi của câu chốt được giải mã **beam 5** kèm từ vựng mồi. Greedy làm số
  từ lặp tăng gấp đôi trên một cuộc họp thật (`"mở mở mở mở mở mở"`).
- Whisper tự đặt khoảng trắng tiếng Nhật ở chỗ người nói ngừng
  (`その結果 本人は`); 13 câu của một lần chạy mang nó. Khoảng trắng **giữa hai
  ký tự tiếng Nhật** bị bỏ; quanh chữ Latin thì giữ.

### Ngôn ngữ của chữ mờ và của câu chốt

> **Đo được (họp thật 30 phút, 09-16):** 37 câu có ngôn ngữ lệch với chữ mờ,
> so với 8 ở bản cửa sổ trượt. Khoảng 30 câu trong số đó là người nói tiếng
> Nhật, chữ mờ bị chốt tiếng Việt ở cửa sổ đầu và **bịa** suốt câu
> (`Các bạn có thể nhận thêm thông tin về các bài hát...`), trong khi LID của cả
> câu ra tiếng Nhật đúng. Và vài câu **trộn hai ngôn ngữ**
> (`これからこのタスは có thểので những次 tiếp theo...`): từ của chữ mờ tiếng
> Việt được hợp nhất với phần đuôi tiếng Nhật.

Ba luật, mỗi luật có test dựng lại từ đúng những câu đó:

- **Ngôn ngữ của chữ mờ** được chốt khi `ASR_STREAM_LANGUAGE_VOTES` cửa sổ
  liên tiếp cùng chắc chắn một ngôn ngữ, và **đổi** khi chừng ấy cửa sổ cùng
  chắc chắn ngôn ngữ kia — chữ mờ khi đó bắt đầu lại (`running texts restarted
  in another language`). LID vẫn chạy mỗi cửa sổ; nó rẻ (vài giây cho 30 phút).
- **Chỉ các lần giải mã cùng ngôn ngữ** được so khớp và hợp nhất với nhau. Câu
  trộn hai ngôn ngữ không còn đường nào để xuất hiện.
- **Câu chốt theo LID của cả câu khi LID chắc chắn** — biên từ
  `ASR_LANGUAGE_OVERRIDE_MARGIN` trở lên. Nếu chữ mờ đang ở ngôn ngữ khác, mọi
  thứ chữ mờ đã làm bị bỏ và **cả câu được giải mã lại** (đếm vào
  `language_flips`). Dưới biên đó thì ngôn ngữ của chữ mờ đứng.
- **Trước khi ngôn ngữ được chốt**, một cửa sổ chỉ dùng câu trả lời của chính
  nó khi biên ≥ `ASR_LANGUAGE_OVERRIDE_MARGIN`; không thì dùng ngôn ngữ gần
  nhất của cuộc họp.
- **Chữ mờ đổi ngôn ngữ thì cắt câu ở đó.** Ngay lúc đổi, phép cắt theo ngôn
  ngữ (mục 6) được hỏi trên phần audio đang mở; nếu nó thấy đúng cặp ngôn ngữ
  chữ mờ vừa thấy, nửa đầu được chốt thành câu riêng (lý do
  `language_change`, giải mã nguyên nửa) và phần sau đi tiếp trong ngôn ngữ mới
  (`running texts changed language, N cut there`).

> **Đo được (09-17, sau ba luật đầu):** câu lệch ngôn ngữ 37 → 24, câu trộn
> 2 → 0. Đọc từng câu trong 24:
>
> - khoảng 13 là **hai lượt nói hai thứ tiếng trong một câu, lượt kia mất**
>   (`Đi kiểm chứng tiếp` ba giây rồi `ステップ011の方は…`, câu chốt chỉ còn tiếng
>   Nhật). Chữ mờ đã đổi ngôn ngữ 35 lần trong câu, nhưng phép cắt ở cuối câu từ
>   chối gần hết — thăm dò ở cuối cả câu thì phải đọc qua lượt kia. Hỏi ngay lúc
>   đổi thì đuôi audio chính là lượt vừa bắt đầu. Nên luật cắt ở trên.
> - khoảng 7 là **chữ mờ bịa sai ngôn ngữ** ở vài cửa sổ đầu, từ những câu trả
>   lời LID không chắc (`Bên mặt của nó sẽ là` trên nền tiếng Nhật). Nên biên
>   0.50 trước khi chốt.
> - khoảng 5 là tiếng đệm (`ừ ừ` cho `うん`) — vô hại.
> - Một câu phát lại có sáu giây chữ mờ tiếng Việt đúng bị giải mã lại thành
>   `はい、で、ウェル` trên một câu trả lời LID biên 0.30. Nên biên 0.50 để lấn chữ
>   mờ.

Luật cũ — giữ ngôn ngữ của chữ mờ khi đã có chữ chốt — dựa trên phép đo của
nhánh improve (câu lệch ngôn ngữ với chữ mờ sai gấp 3–4 lần), nhưng chữ mờ ở
đó bỏ phiếu qua **mọi** cửa sổ. Chốt ở một cửa sổ ngắn thì nó kém tin hơn LID
của cả câu, và lần chạy 09-16 cho thấy đúng điều đó.

- `ASR_STREAM_LANGUAGE_VOTES` tăng: chốt chắc hơn, chữ mờ ở đầu câu đổi ngôn
  ngữ lâu hơn và chữ chốt tới muộn hơn.

- `ASR_LANGUAGE_OVERRIDE_MARGIN` giảm: LID của cả câu thắng chữ mờ thường hơn,
  kể cả khi nó sai; tăng: câu thật sự đổi ngôn ngữ ở cuối bị giữ ngôn ngữ đầu.

Đo lại bằng `python -m server.analysis.compare_logs` trên nhật ký của cùng cuộc
họp, hoặc `server/tests_real/test_real_streaming.py` trên pod. Hai dòng của
bảng dành cho đúng chuyện này: `mixed_language` (câu có cả tiếng Nhật lẫn chữ
tiếng Việt) và `lost_turns` (câu mà chữ mờ đã hiện ít nhất ba lần một ngôn ngữ
khác, mỗi lần từ sáu ký tự trở lên — tiếng đệm không tính). Mốc: 09-11 có 11
`lost_turns`, 09-16 có 11, 09-17 có 21.

- `ASR_STREAM_MIN_AGREEMENT` tăng: chữ chốt chắc hơn và chậm hơn — mỗi bậc thêm
  một nhịp `PARTIAL_INTERVAL_MS`.
- `ASR_STREAM_COMMIT_MARGIN_SECONDS` giảm: chữ chốt nhanh hơn, nhưng mép cửa sổ
  là nơi Whisper sửa nhiều nhất.
- `ASR_STREAM_WORD_TOLERANCE_SECONDS` tăng: khớp được nhiều hơn, và hai từ giống
  nhau đứng gần nhau bị coi là một.
- `ASR_STREAM_FINAL_POST_ROLL_SECONDS` tăng: giữ được đuôi từ bị VAD cắt sớm, và
  cho Whisper thêm im lặng để bịa.

**`ASR_MAX_COMPRESSION_RATIO`** — gzip của tiếng nói tự nhiên rơi vào 1.5–2.0.
Nén tốt hơn hẳn nghĩa là đang lặp một cụm để lấp thời gian.

**`ASR_NO_SPEECH_THRESHOLD` không tự nó loại đoạn nào.** Phải đồng thời
`avg_logprob <= ASR_LOG_PROB_THRESHOLD`. Đây là đúng luật của chính Whisper,
chép nguyên bình luận trong mã nguồn nó:

> `# don't skip if the logprob is high enough, despite the no_speech_prob`

> **Đo được (họp thật):** đọc `no_speech_prob` một mình đã loại nguyên một câu
> `6.8` giây của một người đang nói liền mạch — `'2011 thì mình đang lấy bởi
> vì là cái cả AMD mà bắt cung cấp'` — và một câu `7.0` giây nữa ngay sau đó.
> Cả hai đều đúng lời người nói.

Nó sai **hai chiều cùng lúc**, và đó là lý do phải bỏ: Whisper viết câu bịa
với `avg_logprob` **cao hơn** khi phiên âm thật, nên `no_speech_prob` một mình
vừa giết tiếng nói thật vừa để lọt câu bịa tự tin. Câu bịa là việc của
[`server/data/`](../server/data/README.md), không phải của lớp thống kê.

**Nhưng sự nghi ngờ đó chỉ dành cho đoạn đủ dài.** Dưới
`ASR_SHORT_UTTERANCE_MS` thì `no_speech_prob` một mình lại đủ để loại. Độ dài
được tính trên **cả utterance** (với chữ mờ: phần câu đã nghe tới lúc đó), không
phải trên phần đuôi được giải mã lại.

**`ASR_NO_SPEECH_CERTAIN` đến từ bản streaming và chưa được đo.** Trên một cuộc
họp 30 phút, 68 đoạn bị `no_speech_prob` một mình loại đều được giải mã tự tin;
chưa biết bao nhiêu trong số đó trên 0.95. Mỗi đoạn bị loại đều ghi log kèm cả
hai chỉ số — đọc chúng trước khi giữ hay bỏ số này.

> **Đo được (họp thật 12 phút, người dùng nghe lại xác nhận):** mọi câu
> `Cảm ơn...` lên tới câu chốt đều là bịa, và tất cả đều đến từ mảnh audio quá
> ngắn — nhiều câu mang nhãn `Speaker_unknown`, tức dưới 600 ms. Whisper vẫn
> trả lời, và trả lời **tự tin**: `no_speech 0.86, logprob -0.31`.

Độ dài là thứ tách được hai trường hợp: câu được cứu dài **6.8 giây**, mọi câu
bịa đã xác nhận đều **dưới 2 giây**. Con số 600 ms là đúng lằn ranh mà tầng
người nói (`SPEAKER_MIN_DURATION_MS`) và tầng ngôn ngữ (`LID_MIN_DURATION_MS`)
đã từ chối trả lời — Whisper là model duy nhất vẫn trả lời ở đó.

- Tăng lên: bắt được nhiều câu bịa ngắn hơn, nhưng câu trả lời thật một hai từ
  ("Vâng", "はい") bắt đầu bị mất.
- Giảm xuống: quay về trạng thái câu bịa ngắn lọt lên màn hình.

Mỗi đoạn bị loại giờ in kèm cả hai chỉ số, và mỗi đoạn **được giữ** dù
`no_speech_prob` vượt ngưỡng cũng in ra một dòng — đó là bằng chứng để đặt lại
ngưỡng này về sau:

```
ASR kept a segment scored as silence: no_speech 0.86, logprob -0.35, '...'
```

**Ba ngưỡng trên đều là thống kê, và chúng không bắt được câu bịa tự tin.**
Xem [`server/data/README.md`](../server/data/README.md).

### Từ vựng mồi

> **Đo được:** chữ mờ nghe đúng `Slack`, câu chốt biến thành **`quạt nắp`**.
> Cũng gặp `tab` thành `tắt`, `Claude Code` thành `cloud code`. Tầng dịch sau
> đó dịch trung thành cái vô nghĩa đó.

Whisper nhận `initial_prompt` — chỗ nói trước những từ sắp xuất hiện. Nó
**không ép**, chỉ nghiêng cán cân khi model đang lưỡng lự. Danh sách nằm ở
[`server/data/vocabulary.txt`](../server/data/vocabulary.txt), sửa được không
cần code.

`ASR_PROMPT_MAX_CHARS` giới hạn nó vì hai lý do: Whisper chỉ đọc phần đầu, và
một prompt nhồi nhét làm model **bịa ra chính những từ trong đó** khi gặp im
lặng. Chỉ thêm từ bạn đã thấy bị nghe sai trong transcript.

> **Đo được:** một danh sách 30 từ đoán mò (Zoom, Teams, GitHub, Jira…) làm câu
> bịa lọt lên câu chốt tăng vọt. Một lần chạy khác của cùng cuộc họp **có**
> danh sách từ thật đã nghe đúng `solution` và `dung lượng` ở chỗ bản không có
> danh sách nghe sai.

Chỉ **câu chốt** nhận prompt: chữ mờ giải mã thường gấp sáu lần, và nó không
phải chỗ lỗi. `near_block_list` trong `compare_logs` là thước đo phía chi phí —
số câu chốt đọc giống một câu đã chặn nhưng đổi vài từ. Trước khi có prompt:
12–16 trên cả cuộc họp.

Phần đáng giá nhất là **tên người và tên dự án** — Whisper không có cách nào
đoán được chúng, và đó là phần chỉ bạn điền được.

`vad_filter` của faster-whisper bị **tắt** cứng trong code: Silero đã chạy ở
đầu pipeline, chạy VAD lần hai sẽ cắt mất pre-roll giữ phụ âm đầu.

---

## 8. Dịch

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `TRANSLATE_MODEL` | `google/gemma-4-12b-it` | Phải khớp với model vLLM đang chạy |
| `TRANSLATE_TEMPERATURE` | `0.1` | Thấp nhưng không bằng 0, như khuyến nghị cho Gemma |
| `TRANSLATE_TOP_P` | `0.95` | |
| `TRANSLATE_SEED` | `0` | Cùng câu vẫn cho cùng bản dịch khi temperature > 0 |
| `TRANSLATE_STOP` | `<end_of_turn>`, `<eos>` | Không cho model viết tiếp sang lượt sau |
| `TRANSLATE_ENABLE_THINKING` | `False` | **Đừng bật** (Qwen); template Gemma bỏ qua |
| `TRANSLATE_HISTORY` | `3` | Số lượt hội thoại đưa vào prompt |
| `HISTORY_STYLE` | `"sources"` | Lịch sử chỉ chứa câu gốc |
| `SHORT_LINE_HINT_ENABLED` | `False` | Nhắc model rằng câu một từ vẫn phải dịch |
| `TRANSLATE_MAX_EXPANSION` | `{"vi": 2.0, "ja": 1.0}` | Bản dịch dài tối đa gấp bao nhiêu |
| `TRANSLATE_EXPANSION_SLACK` | `50` | Cộng thêm ngần này ký tự |
| `TRANSLATE_MAX_WRONG_SCRIPT` | `0.30` | Tỉ lệ chữ Nhật tối đa/tối thiểu |
| `TRANSLATION_MAX_LAG_SECONDS` | `10.0` | Chờ quá lâu thì bỏ dịch |
| `TRANSLATION_QUEUE_DEPTH` | `16` | Trần hàng đợi |

**vLLM** được khởi động bởi `server/launch_vllm.py` từ các số sau:

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `VLLM_DTYPE` | `bfloat16` | Gemma chạy float16 ra rác hoặc NaN, không ra lỗi |
| `VLLM_MAX_MODEL_LEN` | `4096` | Prompt chỉ vài trăm token; phần context không dùng là KV cache không phải cấp |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | Phần của **cả card**; vLLM phải khởi động trước server âm thanh |
| `VLLM_TRUST_REMOTE_CODE` | `True` | |
| `VLLM_PORT` | `8001` | |

**`SHORT_LINE_HINT_ENABLED = False`** — lần chạy thật đầu tiên đưa tám câu ngắn
cho gemma-4-12b-it cả có và không có câu nhắc; prompt trơn dịch được hết, kể cả
`はい` → `Vâng`. Real test vẫn thử cả hai.

**`TRANSLATE_ENABLE_THINKING = False`** — Qwen3 suy luận ra tiếng trước khi
trả lời. Với một câu dịch thì đó là toàn chi phí.

> **Đo được:** lần chạy đầu tiêu hết trọn 512 token vào khối `<think>` và
> **không trả về bản dịch nào**, mất 3.5 giây mỗi câu.

**`HISTORY_STYLE = "sources"`** — Lịch sử chỉ chứa câu gốc, không chứa bản
dịch. Có bản dịch trong đó, lịch sử đọc lên như một chuỗi **ví dụ mẫu**, và
khi vài lượt liên tiếp cùng chiều thì mọi ví dụ đều kết thúc bằng một ngôn
ngữ — model bắt chước ví dụ thay vì nghe chỉ dẫn.

> **Đo được (cùng model, cùng lịch sử, cùng lúc, câu `ここに作っているの?` → tiếng Việt):**
>
> | kiểu lịch sử | kết quả |
> | --- | --- |
> | `plain` | `ここで作っているの？` — vẫn tiếng Nhật |
> | `labelled` | `Đang tạo ở đây à?` |
>
> Nhưng `labelled` **không đủ** khi lịch sử sâu ba lượt cùng chiều. Chỉ
> `sources` sạch ở cả hai độ sâu. Số câu dịch được qua các lần chạy:
> 6/10 → 14/17 → 15/18.

Ngữ cảnh không mất đi khi bỏ bản dịch — Whisper nghe "confluence" thành
"công thần", model vẫn dịch đúng thành `コンフルスペース` nhờ câu gốc trước đó.

**Lịch sử là bản chụp lúc câu được chốt.** Dịch chạy sau, trên thread khác;
đọc lịch sử lúc đó thì nó đã chứa chính câu đang dịch, có khi cả câu sau, và
mỗi câu đã dịch nằm trong đó **hai lần** (session ghi một lần, translator ghi
thêm một lần). Giờ chỉ session ghi, và câu mang theo bản chụp của nó.

> **Đo được (họp thật 30 phút, trước khi sửa):** 18 câu bị trả lại nguyên văn
> khi có lịch sử; hỏi lại không kèm lịch sử thì **9** câu dịch được.

Câu bị trả lại nguyên văn vẫn được **hỏi lại một lần không kèm lịch sử**, và
dòng tổng kết in `echoes: N retried without the history, M rescued`. Nếu sau khi
sửa bản chụp mà tỉ lệ cứu vẫn cao, lịch sử vẫn là nguyên nhân và cách viết lịch
sử là chỗ phải đổi tiếp.

**`TRANSLATE_MAX_EXPANSION` phải bất đối xứng.**

> **Đo được (21 cặp dịch thật, `len(bản dịch)/len(câu gốc)` theo ký tự):**
>
> | chiều | khoảng |
> | --- | --- |
> | ja → vi | 1.17 … **4.44** |
> | vi → ja | 0.44 … **0.70** |

Tiếng Nhật chứa cùng lượng thông tin trong ít ký tự hơn nhiều. Một số dùng
chung cho cả hai chiều **sai ở cả hai**: nó từng từ chối một bản dịch đúng ở
chiều ja→vi, đồng thời cao đến mức không bao giờ chạm tới ở chiều ngược lại.

`TRANSLATE_EXPANSION_SLACK` là phần cộng thêm cho câu ngắn — với câu 9 ký tự
thì tỉ lệ gần như chỉ là nhiễu.

**`TRANSLATE_MAX_WRONG_SCRIPT`** — Việt và Nhật không chung hệ chữ, nên đây là
phép kiểm tra rẻ và gần như chắc chắn.

> **Đo được (11 cặp thật, tỉ lệ ký tự kana/kanji):**
>
> | | tỉ lệ |
> | --- | --- |
> | sang Việt, đúng | 0.00 (7 câu) |
> | sang Việt, **không dịch** | 1.00 |
> | sang Nhật, đúng | 0.86 … 1.00 (4 câu) |

Không mẫu nào rơi vào khoảng 0.00–0.86. Ngưỡng 0.30 nằm giữa cả hai khoảng.
Số 0.86 là câu tiếng Nhật mở đầu bằng `FCG` — tên riêng chữ Latin giữ nguyên
không được làm hỏng bản dịch của chính nó.

**Phép này không phân biệt được tiếng Việt với tiếng Anh** (cùng chữ Latin),
nên `プレー` → `Play` vẫn lọt.

**`TRANSLATION_MAX_LAG_SECONDS`** — Đặt ở chỗ bản dịch **hết hữu ích**, không
phải chỗ hàng đợi hết chịu nổi.

> **Đo được (66 khoảng cách giữa các câu, ba lần chạy):** trung vị **3.58
> giây**; nhịp dồn nhất 1.35 câu/giây; một bản dịch tốn 0.15 giây, tức mức
> chiếm dụng **3.7 %**.

Ở mức đó hàng đợi xả gần như tức thì và **độ trễ không cộng dồn qua các cú
vấp** — vấp 1 giây rồi vấp 2 giây tốn 2 giây, không phải 3. Nhưng đó là sự
thật về tốc độ vLLM hiện tại, không phải tính chất của thiết kế, nên giới hạn
vẫn có mặt.

Với khoảng cách trung vị 3.58 giây, 10 giây là đã ba câu trước — bản dịch
hiện ra dưới một câu người đọc đã lướt qua sẽ được đọc như bản dịch của câu
khác.

---

## 9. Client

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `CAPTURE_FRAMES_PER_BUFFER` | `1024` | Kích thước buffer WASAPI |
| `MAX_QUEUED_CHUNKS` | `250` | Trần hàng đợi gửi (50 giây audio) |

Client **không lọc gì cả** — gửi toàn bộ audio kể cả khoảng lặng, khoảng
256 kbps. Đó là chủ ý: VAD nằm ở server (xem `DESIGN.md` mục 3b).

Hàng đợi đầy thì **bỏ chunk cũ nhất**, không chặn luồng đọc audio. Luồng đó
không bao giờ được phép chờ.

---

## Đo lại sau khi chỉnh

```bash
# Pod: khởi động lại rồi chạy
python3.11 -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Hoặc phát lại một cuộc họp đã ghi, không cần client
python3.11 server/tests_real/test_real_streaming.py --wav recordings/meeting_30min.wav
```
```powershell
# Client
python client\tests_real\test_real_stream.py --url ws://127.0.0.1:8000 --seconds 600
```

Dòng tổng kết của server là nơi đọc chi phí từng tầng:

```
slowest sentence 0.6 s, slowest running text 3.9 s,
stages {'partial_asr': 88.7, 'asr': 22.8, 'partial_language': 7.1, ...}
```

Mọi con số ở đó là **thời gian socket không được đọc**. Tổng chia cho độ dài
cuộc họp là phần trăm thời gian pipeline chặn đường vào.

So nhật ký client của các lần chạy **cùng một cuộc họp**:

```powershell
py -3.11 -m server.analysis.compare_logs recordings\meeting-A.debug.txt recordings\meeting-B.debug.txt
```

Nó kiểm tra hai nhật ký là cùng cuộc họp (bỏ phiếu độ lệch thời gian trên
những từ chỉ xuất hiện một lần) trước khi in số nào. Đã từng kết luận sai vì
không làm bước này.
