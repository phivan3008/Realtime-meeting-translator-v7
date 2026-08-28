# Hướng dẫn tinh chỉnh

Mọi thông số nằm trong `server/config.py` và `client/config.py`. Tài liệu này
giải thích **mỗi số nghĩa là gì, nó được chọn từ phép đo nào, và điều gì hỏng
nếu chỉnh sai hướng**.

Danh sách câu bị chặn không nằm ở đây — xem [`server/data/README.md`](../server/data/README.md).

## Cách đọc tài liệu này

Mỗi thông số có một dòng **Đo được**. Đó là dữ liệu thật từ các lần chạy trên
máy client Windows và pod H100, không phải giá trị mặc định của thư viện.
Nếu bạn chỉnh một số, hãy đo lại theo đúng cách đó rồi cập nhật lại đây.

Sau khi sửa `config.py` phải **khởi động lại uvicorn**.

**Trước mọi phép đo, kiểm `"in_venv": true` trong `/health`.** Server chạy
được bằng interpreter khác mà không kêu ca gì — nó nạp gần hết pipeline và
phục vụ cuộc họp bình thường. Ba phép đo của dự án này đã phải vứt vì lý do đó.

## Mốc so sánh

Cuộc họp thật, 593.6 giây audio, pod H100, trong venv, mặc định hiện tại
(tầng lọc nhiễu tắt). Mọi phép đo sau nên so với những con số này.

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
  số này.

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
| `ENABLE_NOISE_FILTER` | không đặt | Đặt `=1` để bật tầng này |
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

**Voiceprint lấy từ audio thô**, chưa qua tầng chồng lấn.

> **Đo được:** gate trước khi trích voiceprint làm mất **0.06** cosine
> cùng-giọng (0.677 thô so với 0.616 đã gate). Cổng cắt cả âm tiết nhỏ trong
> câu, mà âm tiết nhỏ vẫn mang chất giọng.

---

## 5b. Cắt câu khi đổi người nói — ĐANG TẮT

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `SPEAKER_CHANGE_ENABLED` | `False` | Đặt biến môi trường `=1` để bật thử |
| `SPEAKER_CHANGE_WINDOW_MS` | `1000` | Độ dài đoạn đem đi so giọng |
| `SPEAKER_CHANGE_THRESHOLD` | `0.25` | Cosine dưới ngưỡng này là đã đổi người |

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
| `LANGUAGE_SPLIT_ENABLED` | `True` | Biến môi trường `LANGUAGE_SPLIT=0` để tắt |
| `LANGUAGE_SPLIT_PROBE_MS` | `600` | Độ dài mỗi lần thăm dò, bằng `LID_MIN_DURATION_MS` |
| `LANGUAGE_SPLIT_STEPS` | `3` | Số bước tìm nhị phân sau khi biết hai đầu khác nhau |

> **Đo được (họp thật 10 phút, 119 câu):** chữ mờ và câu chốt bất đồng ngôn ngữ
> ở **8 câu**. Đọc lại tám chỗ đó: **4 thật sự mất hẳn một lượt nói** — không
> phải dịch sai, mà không còn trong bản ghi.

Tín hiệu bất đồng đúng **50%**, quá thấp để cắt theo. Nên nó **không** phải
thứ quyết định — nó chỉ là cách rẻ để phát hiện câu hỏi đáng hỏi. Câu hỏi thật
là *audio này có chứa hai ngôn ngữ không*, và LID trả lời trực tiếp được.

Cách làm: thăm dò **đầu** và **cuối** câu. Hai đầu cùng ngôn ngữ, hoặc một đầu
không quyết được, thì **không cắt gì**. Chỉ khi hai đầu khác nhau một cách chắc
chắn mới tìm nhị phân ranh giới giữa chúng, rồi bắt vào khung im nhất gần đó.

So với phép cắt theo giọng đã phải bỏ: cái đó bắn **53%** số phép so vì
voiceprint 1 giây không phân biệt được giọng. Cái này hỏi model một câu hỏi nó
làm tốt, trên cửa sổ đúng bằng cỡ nó được đo.

Chi phí: **2 lần thăm dò** cho câu một ngôn ngữ (trường hợp thường), tối đa 5
khi phải cắt. LID tốn khoảng 6 ms mỗi lần.

- `LANGUAGE_SPLIT_STEPS` tăng: ranh giới chính xác hơn, thêm một lần thăm dò
  mỗi bước. Độ phân giải là độ dài câu chia 2^steps.
- `LANGUAGE_SPLIT_PROBE_MS` giảm xuống dưới `LID_MIN_DURATION_MS`: chính tầng
  LID đã coi cửa sổ ngắn hơn thế là quá ngắn để tin.

> **Chạy thật lần đầu (cùng đoạn 10 phút, hai lần):** bắn **31 trên 110**
> utterance — 28%, cao hơn hẳn con số 8 bất đồng. Nhưng **số câu mất vì chồng
> lấn giảm từ 4 xuống 1**, `slowest sentence` vẫn 0.4 s, và cả tầng tốn 2.0
> giây cho cả cuộc họp.

**Cả hai nửa phải dài ít nhất một lần thăm dò.** Lần chạy đầu lộ ra chỗ hổng:
một probe nằm vắt qua chỗ đổi ngôn ngữ đọc ra ngôn ngữ thứ hai, nên phép tìm
nhị phân đi quá cả ngôn ngữ nó xuất phát và trả về ranh giới cách đầu câu vài
trăm mili-giây. Bước bắt khung im còn kéo lùi thêm 500 ms nữa.

> **Gặp thật:** hai câu chốt cách nhau **31 mili-giây**, câu thứ hai là Whisper
> bịa lên chỗ gần-im-lặng:
>
> ```
> 16:12:51.928  final #50 [ja] と思うんだけどもうちょっと聞いた目線で一回整理
> 16:12:51.959  final #51 [vi] Các bạn nhớ đăng ký kênh để ủng hộ kênh của mình nhé.
> ```
>
> **Mảnh quá ngắn để phiên âm là mảnh Whisper lấp vào, không phải để trống.**

Nay có hai lớp chặn: phép tìm nhị phân bắt đầu cách hai đầu đúng một lần thăm
dò, và session từ chối nhát cắt nào làm một nửa ngắn hơn thế.

### Vì sao nó trượt — nghi phạm là biên của LID

> **Log thật, đúng ca nó sinh ra để bắt:**
>
> ```
> 01:26:06,501 lid: Language undecided: ja and vi are only 0.13 apart
> 01:26:09,132 utterance 3: running text was 'ja', sentence is 'vi'
> ```
>
> Không có dòng `language_split` nào. Splitter đã chạy và từ chối, **im lặng**.

`LID_MIN_MARGIN = 0.30`, mà biên thật đo được chỉ **0.12–0.13**. Probe dài
600 ms — đúng bằng `LID_MIN_DURATION_MS`, tức mức tối thiểu. LID trả về rỗng,
và `find()` bỏ cuộc.

Nay mọi lần từ chối đều được đếm và nêu lý do. Dòng tổng kết tách ba loại:

```
language splits: 30 of 110 utterances held two languages
  (61 one language, 19 undecided at an end, 9 too short), 310 probes
```

Và mỗi lần mất lượt nói, log nói luôn splitter đã thấy gì:

```
utterance 3: running text was 'ja', sentence is 'vi' - ... The splitter
said: undecided at an end (vi … ?)
```

**Đọc `undecided` trước.** Nếu nó lớn thì cách chữa là nới probe lên (ví dụ
900 ms) hoặc hạ `LID_MIN_MARGIN` **chỉ cho phép thăm dò**, không đụng quyết
định ngôn ngữ của cả câu. Nếu `one_language` lớn thì tín hiệu sai bản chất và
phải nghĩ lại.

Dòng tổng kết cuối phiên cho biết nó bắn bao nhiêu lần, để so với con số 8
bất đồng ở trên.

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
| `ASR_PROMPT_MAX_CHARS` | `400` | Giới hạn `initial_prompt` dựng từ `vocabulary.txt` |
| `ASR_MAX_COMPRESSION_RATIO` | `2.4` | Trên ngưỡng này là đang lặp |
| `ASR_CONDITION_ON_PREVIOUS` | `False` | **Đừng bật** |

**`ASR_CONDITION_ON_PREVIOUS = False`** — Bật lên là Whisper lấy câu trước làm
prompt cho câu sau, đúng cơ chế biến **một câu bịa thành cả đoạn bịa**.

### Hai ngôn ngữ trong một câu

> **Đo được trên cùng đoạn 120 giây đó, cả hai lần chạy:**
>
> ```
> partial [ja] それ3番目に入ってしまったんですね
> partial [ja] 3番目帰ってしまったんですね
> final   #4  [vi] Đó, cái chỗ không còn mục tiêu. Cái một là...
> ```
>
> Chữ mờ là tiếng Nhật hai lần liền. Câu chốt ra tiếng Việt, và nội dung tiếng
> Nhật **biến mất** — không phải dịch sai, nó không còn trong bản ghi.

Hai người nói trong một utterance, LID phải chọn một, Whisper bị ép theo lựa
chọn đó cho cả đoạn. Đây là cùng bài toán ở mục 5b, nhưng dữ liệu này cho một
tín hiệu **rẻ hơn nhiều** so với voiceprint: **LID của chữ mờ và LID của câu
chốt bất đồng**. Không cần ECAPA, không cần cửa sổ 1 giây — thứ đã đo và bác bỏ.

Hiện chỉ **đếm và ghi log**, chưa dùng để cắt câu. Lần trước xây trên một tín
hiệu chưa ai đo và nó xé nát bản ghi; lần này đo trước.

Dòng tổng kết cuối phiên sẽ cảnh báo nếu có, và mỗi lần in một dòng:

```
utterance 4: running text was 'ja', sentence is 'vi' - two languages in one
utterance, and one of them is lost
```

Con số cần thu thập trước khi làm gì tiếp: **tỉ lệ này trên một cuộc họp thật**,
và liệu mỗi lần bất đồng có thật sự đi kèm mất nội dung hay không.

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
`ASR_SHORT_UTTERANCE_MS` thì `no_speech_prob` một mình lại đủ để loại.

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

Phần đáng giá nhất là **tên người và tên dự án** — Whisper không có cách nào
đoán được chúng, và đó là phần chỉ bạn điền được.

`vad_filter` của faster-whisper bị **tắt** cứng trong code: Silero đã chạy ở
đầu pipeline, chạy VAD lần hai sẽ cắt mất pre-roll giữ phụ âm đầu.

---

## 8. Dịch

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `TRANSLATE_MODEL` | `Qwen/Qwen3.5-9B` | Phải khớp với model vLLM đang chạy |
| `TRANSLATE_TEMPERATURE` | `0.0` | Cùng câu phải cho cùng bản dịch |
| `TRANSLATE_ENABLE_THINKING` | `False` | **Đừng bật** |
| `TRANSLATE_HISTORY` | `3` | Số lượt hội thoại đưa vào prompt |
| `HISTORY_STYLE` | `"sources"` | Lịch sử chỉ chứa câu gốc |
| `SHORT_LINE_HINT_ENABLED` | `True` | Nhắc model rằng câu một từ vẫn phải dịch |
| `TRANSLATE_MAX_EXPANSION` | `{"vi": 2.0, "ja": 1.0}` | Bản dịch dài tối đa gấp bao nhiêu |
| `TRANSLATE_EXPANSION_SLACK` | `50` | Cộng thêm ngần này ký tự |
| `TRANSLATE_MAX_WRONG_SCRIPT` | `0.30` | Tỉ lệ chữ Nhật tối đa/tối thiểu |
| `TRANSLATION_MAX_LAG_SECONDS` | `10.0` | Chờ quá lâu thì bỏ dịch |
| `TRANSLATION_QUEUE_DEPTH` | `16` | Trần hàng đợi |

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
