# Báo cáo quá trình chỉnh sửa hệ thống phiên dịch cuộc họp VI ↔ JA

Kho mã: `phivan3008/Realtime-meeting-translator-v7`. Mốc khởi điểm: commit `40c768e`
(25/08). Mốc hiện tại: `b7f4217` trên nhánh `alt-from-40c768e` (17/09).

Mọi số đo ở mục "Bằng chứng" lấy từ nhật ký gỡ lỗi của client khi phát lại **cùng
một cuộc họp thật** (khoảng 27 phút, 4 người nói, xen kẽ tiếng Việt và tiếng Nhật),
đo bằng `server/analysis/compare_logs.py`.

## Các chỉ số dùng trong báo cáo

| Chỉ số | Ý nghĩa | Liên quan vấn đề |
| --- | --- | --- |
| `far_from_partial` | Số câu chốt khác hẳn chữ mờ ngay trước nó (viết lại > 60%) | 1 |
| `update_survival` | Phần chữ mờ trên màn hình còn giữ sau mỗi lần cập nhật | 1 |
| `refused_translations` | Số bản dịch bị từ chối (trả nguyên văn, sai ngôn ngữ, quá dài) | 1 |
| `lost_turns` | Số câu mà chữ mờ đã hiện một lượt nói bằng ngôn ngữ khác, nhưng câu chốt không còn lượt đó | 2 |
| `language_disagrees` | Số câu chốt khác ngôn ngữ với chữ mờ của chính nó | 2 |
| `mixed_language` | Số câu trộn cả tiếng Nhật lẫn tiếng Việt | 2 |
| `near_block_list` | Số câu chốt giống câu bịa đã biết nhưng đổi vài chữ | 3 |
| `japanese_spaced`, `with_repeats` | Số câu tiếng Nhật bị cách từng chữ; số câu lặp từ | 1 |

---

## Giai đoạn 0 — Hiện trạng ban đầu (`40c768e`, 25/08)

Hệ thống đủ 8 tầng: VAD → Buffer → lọc nhiễu AST → Overlap Resolver → nhận dạng
người nói → LID → Whisper → dịch Qwen3.5-9B. Chữ mờ giải mã lại 4 giây cuối mỗi
600 ms; câu chốt giải mã lại **toàn bộ** câu một lần nữa, độc lập với chữ mờ. Chưa
có giao diện client và chưa có nhật ký cuộc họp. Ba vấn đề khởi điểm được ghi nhận
trên phiên bản này.

---

## Giai đoạn 1 — Tách người nói khi nhiều người nói nối nhau (nhánh `main`, 25/08)

**Vấn đề cần giải quyết (VĐ2).** VAD chỉ đóng câu sau 500 ms im lặng, nên người
thứ hai nói ngay sau thường bị gộp vào cùng một câu: hai người thành một nhãn, và
cả câu bị ép một ngôn ngữ.

**Đã thay đổi.**
- Thêm phép cắt câu khi có giọng thứ hai, so voiceprint cửa sổ 1 giây với đầu câu.
- Thêm gom cụm lại toàn cuộc họp để sửa nhãn người nói đã gửi.
- Thêm giao diện client (cửa sổ họp) để quan sát.

**Kết quả.**
- Phép cắt theo giọng **không dùng được**: voiceprint 1 giây không phân biệt được
  người nói, và nó cắt 53% số lần so. Phép cắt này được **tắt**.
- Gom cụm lại được giữ lại. Giai đoạn 6–7 đo lại thấy nó làm nhãn tệ hơn, nên chuyển
  sang chế độ chỉ đo, không gửi nhãn đã sửa.

**Bằng chứng.**
- Họp thật 175 s, 137 lần so giọng: 73 lần cắt. Decile cosine từ 0.017 tới 0.558,
  liền một dải, không có khoảng trống để đặt ngưỡng.
- Cùng một đoạn audio: cosine theo cửa sổ 1 s là 0.122, theo cả câu là 0.765.
- Khi bật: 93 câu trong 175 s (1 câu mỗi 1.9 s), số người nói chạm trần 12 sau 21
  câu, số câu bị mất tăng.

---

## Giai đoạn 2 — Chặn nội dung bịa và chống mất câu (nhánh `main`, 25–28/08)

**Vấn đề cần giải quyết (VĐ3, một phần VĐ1).**
- Whisper tự điền câu không ai nói ("Cảm ơn các bạn đã theo dõi…") trên đoạn nhỏ
  tiếng hoặc gần im lặng.
- Ngược lại, câu thật dài bị loại nhầm là im lặng.
- Một tầng hỏng làm mất cả cuộc họp mà không báo gì.

**Đã thay đổi.**
- `no_speech_prob` chỉ được loại đoạn khi `avg_logprob` cũng thấp. Riêng mẩu audio
  dưới 600 ms thì `no_speech_prob` một mình vẫn đủ để loại.
- Danh sách câu bịa chuyển thành file dữ liệu sửa được, có thêm mẫu (regex) và danh
  sách giữ lại.
- Thêm từ vựng mồi (`initial_prompt`), chỉ gồm các từ đã thấy bị nghe sai.
- Lỗi được bắt theo từng tầng. Tầng lỗi 3 lần liên tiếp thì bị tắt; lỗi CUDA thì
  tắt ngay; client được báo.
- Tầng lọc nhiễu AST tắt mặc định.
- Tầng dịch: câu bị trả nguyên văn được hỏi lại một lần không kèm lịch sử.
- Bản đầu tiên của phép cắt câu chứa hai ngôn ngữ.
- `/health` báo interpreter và các biến môi trường đang đặt.

**Kết quả.**
- Câu dài không còn bị loại nhầm.
- Câu bịa có mẫu đã biết bị chặn.
- Pipeline không còn "chết im lặng".
- Câu chậm nhất giảm từ 2.1 s xuống 0.4 s.

**Bằng chứng.**
- Câu 6.8 s đúng lời người nói, trước đây bị loại là "no speech", nay được giữ.
- Lọc nhiễu AST: 237 câu, bỏ được 0 câu, tốn 1.31 s mỗi câu.
- Lượt nói bị mất do hai thứ tiếng trong một câu: 4 → 1 trên 10 phút họp.
- Nhật ký 28/08 (`meeting-20260828-104348`): `near_block_list` 13,
  `far_from_partial` 75, `lost_turns` 20, `update_survival` 33.6%.

---

## Giai đoạn 3 — Tìm nguyên nhân câu chốt kém chữ mờ (nhánh `improve`, 10–11/09)

**Vấn đề cần giải quyết (VĐ1).** Chữ mờ đúng, nhưng câu chốt thiếu ý hoặc sai ý
("về Solution và cái lý do" → "về sau lưu sinh và cái cái lý do").

**Đã thay đổi.**
- Công cụ đo độ lệch giữa câu chốt và chữ mờ, phát lại điểm số của các lớp chặn
  (không cần GPU), và đọc nhật ký client.
- Qua đo: beam search và việc cắt khoảng lặng cố định **không phải** nguyên nhân.
  Nguyên nhân là **một phần năm số câu không bao giờ lên màn hình**, vì bị loại bởi
  `no_speech_prob`.
- Sửa luật `no_speech`.
- Khi câu chốt khác ngôn ngữ với chữ mờ, giải mã lại câu bằng ngôn ngữ của chữ mờ.

**Kết quả.** Câu rỗng giảm một nửa; câu lệch ngôn ngữ với chữ mờ giảm mạnh.

**Bằng chứng.**
- 30 phút họp: 68 đoạn bị loại vì `no_speech`, cả 68 đều giải mã tự tin; câu rỗng
  91 → 45 (khôi phục 173 s nội dung).
- Nhật ký (28/08 → 11/09 00:37 → 11/09 10:41): `language_disagrees` 20 → 14 → 7;
  `far_from_partial` 75 → 34 → 41; `near_block_list` 13 → 16 → 16.

---

## Giai đoạn 4 — Hai lượt nói bằng hai thứ tiếng trong một câu (nhánh `improve`, 11/09)

**Vấn đề cần giải quyết (VĐ2).** Người Việt và người Nhật đáp nhau nhanh hơn 500 ms
nên nằm chung một câu. LID chọn một ngôn ngữ, và lượt kia **biến mất** chứ không bị
dịch sai.

**Đã thay đổi.**
- Phép cắt theo ngôn ngữ thăm dò lùi vào 300 ms từ hai đầu câu, cần biên chắc chắn
  0.50.
- Trước khi cắt, thăm dò lại chính hai nửa sẽ gửi đi.
- Không để lại mảnh dưới 1.2 s tiếng nói, có tính cả khoảng lặng cuối câu.
- Mở rộng từ vựng mồi.

**Kết quả.** Số lần cắt sai giảm mạnh; cắt ra mảnh vụn (Whisper tự điền) gần như
hết.

**Bằng chứng.**
- 30 phút họp: lần cắt 76 → 27; cặp cắt nhìn thấy trên màn hình 41 → 9; cặp cắt mà
  hai nửa cùng một ngôn ngữ 26 → 3.
- Nhật ký 11/09 14:40: `language_disagrees` 8, `lost_turns` 11,
  `refused_translations` 9, `far_from_partial` 48.

---

## Giai đoạn 5 — Ổn định chữ mờ bằng ASR streaming, đổi model dịch (nhánh `alt`, 13–14/09)

**Vấn đề cần giải quyết (VĐ1).**
- Chữ mờ là cửa sổ trượt 4 giây: đầu câu trôi mất, và 2/3 số lần cập nhật xoá quá
  nửa dòng đang đọc.
- Câu chốt giải mã lại từ đầu nên có thể mất điều chữ mờ đã nghe đúng.

**Đã thay đổi.**
- Áp dụng cách tiếp cận ASR streaming: từ nào hai lần giải mã liên tiếp đồng ý thì
  **chốt vĩnh viễn**; câu chốt chỉ giải mã phần chưa chốt.
- Đổi model dịch sang `gemma-4-12b-it`.

**Kết quả.** Chữ mờ ổn định gấp đôi và câu chốt bám sát chữ mờ. Nhưng xuất hiện lỗi
hiển thị nặng, và 77 unit test hỏng.

**Bằng chứng.** Nhật ký 10/09 của cách tiếp cận này, so với 11/09 14:40:

| Chỉ số | 11/09 14:40 | 10/09 (streaming) |
| --- | --- | --- |
| `update_survival` | 35.6% | 64.3% |
| `far_from_partial` | 48 | 18 |
| `japanese_spaced` | 9 | 144 |
| `with_repeats` | 16 | 131 |
| `refused_translations` | 9 | 19 |

---

## Giai đoạn 6 — Hoàn thiện ASR streaming và hợp nhất các sửa lỗi (nhánh `alt`, 16/09)

**Vấn đề cần giải quyết (VĐ1, VĐ2, VĐ3).** Giữ độ ổn định của streaming nhưng bỏ
lỗi hiển thị, và đưa các sửa lỗi của giai đoạn 1–4 vào cùng một bản.

**Đã thay đổi.**
- **Dựng văn bản:** giữ nguyên chuỗi từ của Whisper (không tự chèn khoảng trắng);
  bỏ từ lặp ở chỗ nối giữa chữ đã chốt và phần đuôi.
- **Câu chốt:** chỉ giải mã tới chỗ tiếng nói thật sự dừng (bỏ khoảng lặng cuối
  câu), beam 5, có từ vựng mồi.
- **Dịch:** lịch sử dịch là **bản chụp lúc câu được chốt**. Trước đó, lịch sử chứa
  cả câu đang dịch, và mỗi câu nằm trong đó hai lần.
- Đưa sang từ các giai đoạn trước: chặn câu bịa, xử lý lỗi theo tầng, phép cắt theo
  ngôn ngữ, cửa sổ họp, nhật ký.
- Gom cụm người nói: giới hạn 12 người, và chỉ đổi nhãn khi hai lần gom cùng đề
  xuất.
- Thêm real test phát lại cuộc họp trên pod.

**Kết quả.**
- Lỗi hiển thị hết, bản dịch bị từ chối giảm mạnh.
- Lộ ra hai vấn đề mới: câu lệch ngôn ngữ tăng (có câu trộn hai thứ tiếng), và gom
  cụm người nói làm nhãn nhảy liên tục.

**Bằng chứng.** Nhật ký 16/09:
- `japanese_spaced` 144 → 13 (đều là khoảng trắng do chính Whisper đặt),
  `with_repeats` 131 → 8.
- `refused_translations` 19 → 3, `near_block_list` 12 → 4,
  `update_survival` 64.3%.
- `language_disagrees` 37, `mixed_language` 2.
- 149 lần sửa nhãn trên 290 câu.

---

## Giai đoạn 7 — Giữ đủ các lượt nói trong câu dài (nhánh `alt`, 17/09)

**Vấn đề cần giải quyết (VĐ1, VĐ2).**
- Chữ mờ bị chốt ngôn ngữ ngay từ cửa sổ đầu (0.6–1.2 s), nên bịa câu tiếng Việt
  trên nền người nói tiếng Nhật.
- Câu chốt trộn hai thứ tiếng.
- Lượt nói thứ hai trong câu vẫn bị mất.

**Đã thay đổi (qua 4 lần chạy thật liên tiếp, mỗi lần sửa theo từng câu đọc được
trong nhật ký).**
- **Ngôn ngữ của chữ mờ:** chốt khi 2 cửa sổ liên tiếp chắc chắn (biên ≥ 0.50);
  đổi khi 2 cửa sổ chắc chắn nói ngôn ngữ kia. Chỉ các lần giải mã cùng ngôn ngữ
  mới được ghép với nhau.
- **Ngôn ngữ của câu chốt:** theo LID của cả câu khi LID chắc chắn; nếu khác chữ
  mờ thì giải mã lại cả câu. Khoảng trắng giữa hai chữ tiếng Nhật bị bỏ.
- **Cắt câu ngay khi chữ mờ đổi ngôn ngữ.** Phép cắt ở cuối câu phải khớp với ngôn
  ngữ các cửa sổ chữ mờ đã nghe.
- **Nửa câu cùng ngôn ngữ với chữ mờ** được chốt từ chính chữ mờ, không giải mã lại
  từ đầu. Giải mã lại từ đầu từng cho ra vòng lặp "TACCAP, TACCAP…" và làm mất trắng
  nửa câu.
- **Người nói:** nhãn sửa sau gom cụm không còn gửi về client (chỉ đo), vì với cuộc
  họp 4 người, bộ gán nhãn trực tiếp cho đúng 4 nhãn chính.
- Chặn thêm các câu bịa kiểu "Các bạn có thể nhận thêm… phần bình luận".

**Kết quả.** Không còn câu trộn hai thứ tiếng hay câu tiếng Nhật bị cách chữ. Lượt
nói bị mất còn khoảng một nửa so với bản cửa sổ trượt. Câu chốt bám chữ mờ sát nhất
từ đầu dự án. Real test phát lại đạt 15/15 kiểm tra.

**Bằng chứng.** Nhật ký cùng cuộc họp:

| Chỉ số | 11/09 14:40 | 16/09 | 17/09 08:18 | 17/09 11:40 | 17/09 13:54 | **17/09 15:57** |
| --- | --- | --- | --- | --- | --- | --- |
| `lost_turns` | 11 | 11 | 21 | 8 | 8 | **6** |
| `language_disagrees` | 8 | 37 | 24 | 14 | 14 | **9** |
| `mixed_language` | 0 | 2 | 0 | 0 | 0 | **0** |
| `japanese_spaced` | 9 | 13 | 0 | 0 | 0 | **0** |
| `far_from_partial` | 48 | 46 | 29 | 24 | 26 | **18** |
| `refused_translations` | 9 | 3 | 7 | 5 | 5 | **3** |
| `near_block_list` | 16 | 4 | 5 | 2 | 3 | **2** |
| `update_survival` | 35.6% | 64.3% | 61.9% | 63.0% | 64.0% | **63.9%** |

Người nói ở lần chạy cuối:
- Bộ gán nhãn trực tiếp: 4 nhãn chính (103 / 98 / 63 / 17 câu, cộng 3 nhãn chỉ
  có 1 câu).
- Gom cụm lại (chỉ đo): 10 cụm (179 / 63 / 13 / …); nếu được gửi, 113 câu sẽ bị đổi
  nhãn.

---

## Tổng kết theo vấn đề khởi điểm

| Vấn đề | Trạng thái | Đã giải quyết | Còn mở |
| --- | --- | --- | --- |
| 1. Câu dài bị thiếu hoặc sai ý | **Đã giải quyết** | Chữ đã chốt không bị viết lại; câu chốt bám chữ mờ (`far_from_partial` 75 → 18); bản dịch bị từ chối 10 → 3; lịch sử dịch đúng câu | — |
| 2. Nhiều người nói chồng lấn | **Giải quyết một phần** | Hai lượt nói hai thứ tiếng trong một câu: `lost_turns` 20 → 6, không còn câu trộn ngôn ngữ; nhãn người nói ổn định | Hai người nói **cùng lúc**: Overlap Resolver chưa từng được đo hợp lệ. Hai lượt **cùng một thứ tiếng** trong một câu: cắt theo giọng đã thử và phải tắt |
| 3. Âm lượng thấp, tín hiệu không ổn định | **Giải quyết một phần** | Whisper tự điền nội dung trên đoạn nhỏ tiếng: `near_block_list` 13 → 2, chặn theo mẫu, luật mẩu ngắn, bỏ khoảng lặng cuối câu | Chưa có chuẩn hoá âm lượng và chưa chỉnh VAD cho người nói nhỏ; chưa đo khoảng trống transcript; chưa có biện pháp riêng cho việc LLM tự thêm ý |
