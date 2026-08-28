# Nhánh `translate-gemma`

Thay model dịch từ `Qwen/Qwen3.5-9B` sang họ Gemma, **không đổi gì khác**. Tám
tầng xử lý âm thanh, các guard, và client đều y nguyên. Client không biết —
và không cần biết — server đang dùng model nào.

> **Điều tôi không chắc:** tôi không có kiến thức xác thực về checkpoint
> `gemma-4-12b-it` cụ thể. Những gì dưới đây đúng cho **họ Gemma** như tôi
> biết, và mọi thứ đặc thù phiên bản đều **cấu hình được** chứ không hardcode.
> Con số nào chưa đo thì được ghi rõ là chưa đo.

## Chọn nhánh nào khi chạy

Không phải chọn nhánh code — chỉ chọn **profile** lúc khởi động server, và
khởi động vLLM với đúng model.

```bash
# vLLM, terminal riêng
python3.11 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-12b-it \
    --port 8001 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 4096

# server âm thanh, terminal riêng
python3.11 -m server.run --profile gemma
```

Quay lại Qwen thì đổi đúng hai chỗ đó:

```bash
python3.11 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.55
python3.11 -m server.run --profile qwen
```

`uvicorn server.app:app` vẫn chạy được và vẫn mặc định `qwen`, nên mọi lệnh cũ
không đổi hành vi.

Kiểm bằng `/health`:

```json
"translation_profile": "gemma",
"translation_loaded": true
```

Sai profile thì server **từ chối khởi động** tầng dịch: `choose_model` không
tìm thấy checkpoint nó chờ. Nếu bạn ép `TRANSLATE_MODEL` cho khớp mà để sai
profile thì log ra cảnh báo — vLLM trả lời một request sai template rất vui
vẻ, và triệu chứng duy nhất là bản dịch tệ hơn mức đáng có.

## Hai khác biệt kỹ thuật, và vì sao chỉ có hai

Cả hai đều là **lỗi thẳng**, không phải suy giảm chất lượng.

**Gemma không có vai `system`.** Chat template của nó chỉ có `user` và
`model`. Gửi một message `system` là lỗi template, không phải cảnh báo. Nên
`ModelProfile.messages()` gộp phần hướng dẫn vào lượt `user` đầu tiên.

**Gemma không có chế độ thinking.** `chat_template_kwargs.enable_thinking` là
của Qwen3 — Qwen3 suy luận ra tiếng trước khi trả lời nếu không bị tắt, và lần
chạy đầu tiên nó tiêu trọn 512 token vào khối `<think>` rồi không trả bản dịch
nào. Template của Gemma không khai báo kwarg đó, và gửi một kwarg không khai
báo là lỗi trên một số bản vLLM. Nên nó nằm trong profile, không nằm trong
hàm dựng request.

**Mọi thứ còn lại dùng chung có chủ ý** — system prompt, lịch sử hội thoại,
`temperature = 0`, ngân sách token, các guard từ chối, phép bóc `<think>`.
Một profile bắt đầu chứa nội dung prompt là một profile sẽ trôi khỏi cái nó
đang được đem ra so sánh. Có test canh điều đó.

## So sánh hai model

Một card H100 không cùng lúc chở nổi 12B, 9B và Whisper. Nên phép so là **hai
lần chạy và một lần diff**, không phải một lần chạy.

```bash
# vLLM đang chạy Qwen
python3.11 server/tests_real/test_real_translate.py \
    --profile qwen --save server/tests_real/output/qwen.json

# khởi động lại vLLM với Gemma
python3.11 server/tests_real/test_real_translate.py \
    --profile gemma --save server/tests_real/output/gemma.json

python3.11 server/tests_real/compare_translate.py \
    server/tests_real/output/qwen.json \
    server/tests_real/output/gemma.json
```

Script quyết định được **ba** thứ, và đó đúng là ba cách một model thay thế có
thể tệ đi mà không ai nhận ra cho tới lúc họp:

| | |
| --- | --- |
| từ chối nhiều hơn | câu nào Qwen dịch được mà Gemma không |
| chậm hơn | trung vị độ trễ, cảnh báo khi chậm hơn 1.5 lần **và** hơn 0.5 giây |
| trượt check | check nào bản kia qua mà bản này trượt |

Thoát mã `1` nghĩa là **tệ hơn ở phần đo được**. Nhưng nó **không** quyết được
tiếng Nhật có hay hơn không — nó in cả hai câu trả lời cạnh nhau để bạn đọc.
Đúng thoả thuận mà mọi real test trong dự án này đều theo.

## Chưa đo — phải làm trước khi chốt

- **VRAM.** Gemma 12B ở bf16 cần khoảng gấp rưỡi Qwen 9B. `0.60` ở trên là
  điểm khởi đầu, **không phải số đã đo**. Nếu Whisper hết chỗ thì hạ xuống,
  hoặc chạy vLLM với lượng tử hoá.
- **Độ trễ.** Model lớn hơn thì mỗi câu chậm hơn. Tầng dịch chạy ngoài luồng
  audio nên chậm là *trễ phụ đề*, không phải *nghẽn tiếng nói* — nhưng
  `worst translation lag` ở dòng tổng kết là chỗ phải nhìn.
- **Tỉ lệ echo.** Qwen trả lại nguyên văn câu nguồn 18 lần trong 30 phút, một
  nửa là do lịch sử hội thoại. Gemma có thể khác hẳn theo cả hai chiều. Dòng
  `echoes: N retried, N rescued` so trực tiếp được.
- **`ありがとうございました` và các câu bịa.** Đó là tầng ASR, không phải tầng
  dịch — đổi model dịch không đụng tới chúng.
