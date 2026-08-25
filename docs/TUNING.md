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

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `NOISE_MIN_SPEECH_SCORE` | `0.2` | Dưới ngưỡng này mới xét bỏ |
| `NOISE_MIN_NOISE_SCORE` | `0.3` | ...và chỉ bỏ khi model chắc chắn nghe thấy thứ khác |
| `NOISE_WINDOW_SECONDS` | `10.0` | Cửa sổ AST đọc mỗi lần |
| `AST_MODEL_ID` | `MIT/ast-finetuned-…` | Đổi qua biến môi trường `AST_MODEL_ID` |
| `NOISE_DEVICE` | tự chọn | `cuda` / `cpu` |

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

**Voiceprint lấy từ audio thô**, chưa qua tầng chồng lấn.

> **Đo được:** gate trước khi trích voiceprint làm mất **0.06** cosine
> cùng-giọng (0.677 thô so với 0.616 đã gate). Cổng cắt cả âm tiết nhỏ trong
> câu, mà âm tiết nhỏ vẫn mang chất giọng.

---

## 5b. Cắt câu khi đổi người nói

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `SPEAKER_CHANGE_ENABLED` | `True` | Đặt biến môi trường `=0` để tắt hẳn |
| `SPEAKER_CHANGE_WINDOW_MS` | `1000` | Độ dài đoạn đem đi so giọng |
| `SPEAKER_CHANGE_THRESHOLD` | `0.25` | Cosine dưới ngưỡng này là đã đổi người |

VAD chỉ đóng đoạn sau `VAD_MIN_SILENCE_MS` (500 ms) im lặng. Người sau nói
tiếp nhanh hơn thế thì **hai giọng nằm chung một utterance**, và utterance đó
chỉ được một voiceprint, một lần nhận dạng ngôn ngữ, một lần ASR.

> **Đo được (họp thật một tiếng, 30 phút đầu):** hai người nói tiếng Việt
> cùng ra `Speaker_01`. 3–5 lần một câu tiếng Việt ngắn (5–7 chữ) mất hẳn:
> nó bị nuốt vào utterance của câu tiếng Nhật nối ngay sau, trôi khỏi cửa sổ
> partial 4 giây, và không bao giờ được chốt.

Giây đầu của utterance là **mốc** — người mở lời. Mỗi nhịp partial, giây gần
nhất được so với mốc đó. Lệch quá ngưỡng thì cắt **ngay trước** cửa sổ lệch,
không phải tại chỗ phát hiện, để giây của người mới không dính vào câu của
người cũ.

**`SPEAKER_CHANGE_THRESHOLD`** thấp hơn `SPEAKER_MATCH_THRESHOLD` (0.30) một
cách có chủ ý: đoạn so ở đây chỉ dài 1 giây thay vì cả câu, nên cosine
cùng-giọng nhiễu hơn và tụt xuống. Cắt nhầm tốn kém hơn bỏ sót — một câu bị
xé đôi thì cả hai nửa đều dịch kém, còn bỏ sót chỉ là giữ nguyên hành vi cũ.

> **Chưa đo trên dữ liệu thật.** 0.25 là chỗ đặt tạm giữa hai vùng đã đo cho
> cả câu. Mỗi lần so đều được ghi vào `ChangeStats.scores` và log ở mức DEBUG,
> nên một lần chạy thật đủ để dựng phân bố và chọn lại số này.

- Giảm xuống: ít cắt hơn, quay dần về hành vi cũ (hai người chung một câu).
- Tăng lên: cắt vụn. Một người đổi giọng — cười, hạ giọng, ho — cũng thành
  ranh giới câu, và ASR mất ngữ cảnh ở mỗi mảnh.

**`SPEAKER_CHANGE_WINDOW_MS`** quyết định phát hiện được sớm đến đâu: cần đủ
audio cho **hai** cửa sổ không chồng nhau, nên 1000 ms nghĩa là sớm nhất
2 giây sau khi utterance mở. Giảm xuống thì phát hiện sớm hơn nhưng voiceprint
của đoạn ngắn nhiễu hơn — dưới `SPEAKER_MIN_DURATION_MS` (600 ms) thì chính
tầng nhận dạng người nói đã coi là quá ngắn để tin.

Chi phí: **một lần embed ECAPA mỗi 600 ms**. Mốc chỉ embed một lần cho mỗi
utterance.

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

---

## 7. ASR (Whisper)

| Thông số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `ASR_MODEL` | `large-v3` | Checkpoint faster-whisper |
| `ASR_BEAM_SIZE_PARTIAL` | `1` | Chữ mờ giải mã tham lam |
| `ASR_BEAM_SIZE_FINAL` | `5` | Câu chốt được beam search |
| `ASR_NO_SPEECH_THRESHOLD` | `0.6` | Trên ngưỡng này coi như không có tiếng nói |
| `ASR_LOG_PROB_THRESHOLD` | `-1.0` | Dưới ngưỡng này là đoán mò |
| `ASR_MAX_COMPRESSION_RATIO` | `2.4` | Trên ngưỡng này là đang lặp |
| `ASR_CONDITION_ON_PREVIOUS` | `False` | **Đừng bật** |

**`ASR_CONDITION_ON_PREVIOUS = False`** — Bật lên là Whisper lấy câu trước làm
prompt cho câu sau, đúng cơ chế biến **một câu bịa thành cả đoạn bịa**.

**`ASR_MAX_COMPRESSION_RATIO`** — gzip của tiếng nói tự nhiên rơi vào 1.5–2.0.
Nén tốt hơn hẳn nghĩa là đang lặp một cụm để lấp thời gian.

**Ba ngưỡng trên đều là thống kê, và chúng không bắt được câu bịa tự tin.**
Xem [`server/data/README.md`](../server/data/README.md).

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
