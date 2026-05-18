# Pattern Calibration Framework

## Mục tiêu

Framework này dùng để nội địa hóa từng mẫu hình cho thị trường Việt Nam mà
không biến quá trình nghiên cứu thành tham số hóa hậu nghiệm.

Nguyên tắc chính:

```text
Giữ rule nhận diện từ sách gốc làm provenance
-> chạy trên active Market Stats universe
-> đo target family cố định theo từng pattern family
-> chọn base target bằng rule công khai
-> kiểm tra liquidity/regime/overlap/path-quality
-> mới viết chapter PDF
```

Đây là calibration, không phải optimization. Một target chỉ được chọn nếu qua
cổng thống kê đã khóa trước.

## Target family hiện tại

| Pattern family | Target family | Ghi chú |
|---|---:|---|
| Bull Flag / Flags | `0.46x`, `0.5x`, `0.75x`, `1.0x` | `0.46x` là Bulkowski-adjusted benchmark; `1.0x` chỉ là legacy full-pole benchmark |
| Broadening Bottoms | `0.65x`, `0.75x`, `1.0x` | `0.65x` là adjusted benchmark; `1.0x` chỉ để so sánh |
| Unknown / draft pattern | `0.5x`, `0.75x`, `1.0x`, `1.25x` | Chỉ dùng diagnostic cho đến khi có provenance family riêng |

## Rule chọn base target

Base target của chapter phải là target đầu tiên trong target family vượt đủ các
cổng dưới đây:

| Gate | Ngưỡng hiện tại |
|---|---:|
| `N` tối thiểu | `100` cho headline pattern |
| Wilson lower bound của target hit | `>= 55%` |
| Target-first-before-adverse-5% | `>= 35%` |
| Failure 5% | `<= 30%` |

Nếu nhiều target cùng pass, chọn target xuất hiện sớm hơn trong family order để
giữ đúng nguyên tắc Bulkowski-adjusted trước, target tham vọng sau.

Nếu không target nào pass, chapter không được phong base target. Khi đó chỉ báo
target sensitivity và ghi trạng thái `no_base_target_pass`.

## Split bắt buộc cho Bull Flag

Bull Flag hiện được kiểm tra theo các lát cắt:

- `liquidity_bucket`: `high`, `mid`, `low`
- `primary_60d`: event chính sau cooldown 60 ngày và repeat event
- `path_proxy_clean` / `path_proxy_flagged`
- `corp_proxy_clean` / `corp_proxy_flagged`

Các split này không được dùng để cherry-pick target headline. Chúng chỉ dùng để
kiểm tra robustness và viết caveat.

## Kết quả Bull Flag hiện tại

Theo artifact `artifacts/scanner_v2/research_support/target_calibration_decisions.json`:

| Metric | Giá trị |
|---|---:|
| Selected target | `0.46x` |
| Role | `bulkowski_adjusted_base` |
| N | `110` |
| Target hit | `70.00%` |
| Wilson lower bound | `60.88%` |
| Target-first-before-adverse-5% | `42.73%` |
| Failure 5% | `24.55%` |
| MFE/MAE median ratio | `1.56` |

Kết luận vận hành: Bull Flag có thể dùng `0.46x pole` làm base target cho bản
chapter tiếp theo, trong khi `0.5x` là rounded local base và `1.0x` chỉ là
legacy benchmark.

## Cảnh báo

- Không dùng target family để tối ưu PnL.
- Không xếp hạng toàn bộ pattern trước khi mỗi pattern có calibration riêng.
- Không dùng historical VN30/VN100 membership làm headline khi dữ liệu không đủ.
- Không claim full point-in-time universe trong phạm vi `available_series_descriptive`.
- Bearish/downside pattern trên cash equities Việt Nam mặc định là
  informational/defensive reference cho đến khi có instrument và execution
  model phù hợp.
