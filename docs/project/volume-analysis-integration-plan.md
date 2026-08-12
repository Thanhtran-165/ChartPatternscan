# Volume Analysis Integration Plan

Nguồn đọc: `Buff Pelz Dormeier-Investing with Volume Analysis_ Identify Follow and Profit from Trends-FT Press 2011.pdf`.

Mục tiêu của note này không phải tóm tắt sách, mà là chuyển các ý tưởng dùng được từ phân tích khối lượng vào dự án mẫu hình giá Việt Nam: scanner, thống kê hậu mẫu, tradable layer và BUY Candidate Scan.

## Kết luận ứng dụng

Tài liệu của Dormeier củng cố một nguyên tắc phù hợp với dự án hiện tại: giá cho biết hình thái, khối lượng cho biết mức độ tham gia và sức xác nhận. Với dữ liệu hiện có là daily OHLCV, phần có giá trị nhất không phải tick-flow hay block-trade analysis, mà là một lớp `volume_confirmation_layer` chạy trên từng mã và từng event.

Nên triển khai theo ba tầng:

1. **Pre-breakout setup**: kiểm tra mẫu đang hình thành có được khối lượng ủng hộ hay đang chỉ là nhiễu.
2. **Breakout confirmation**: kiểm tra phiên xác nhận có đủ lực so với nền khối lượng gần đây.
3. **Post-breakout health**: kiểm tra sau phá vỡ giá đi tiếp trên volume lành mạnh hay chỉ bùng nổ một phiên rồi suy yếu.

## Những phần dùng được ngay

| Ý tưởng từ tài liệu | Cách lượng hóa trong dự án | Nơi áp dụng đầu tiên | Trạng thái |
|---|---|---|---|
| Volume xác nhận xu hướng giá | `price_volume_phase`: giá tăng + volume tăng, giá tăng + volume giảm, giá giảm + volume tăng, giá giảm + volume giảm | BUY_SETUP, BUY_PULLBACK, symbol profile | Nên làm ngay |
| Volume surge ở breakout | `breakout_volume_ratio_20` hoặc `breakout_volume_ratio_50`; cờ `volume_confirmed` | confirmed event scanner, BUY_PULLBACK email/PDF | Đã có rời rạc, cần chuẩn hóa |
| Volume co lại trong consolidation | `pattern_volume_slope`, `pattern_volume_contraction_ratio`, `flag_volume_to_pole_ratio` | Flags, Pennants, Triangles, Rectangles, Cup handle | Đã có một phần, cần đưa vào common layer |
| OBV / VPT divergence | slope OBV/VPT trong pattern so với slope close | setup-quality before breakout | Nên làm P1 |
| Money Flow Index | MFI 14/20 và slope MFI trong mẫu | filter phụ cho BUY_SETUP | Nên làm P1 |
| VWMA/VW-MACD | so sánh SMA/VWMA ngắn-dài; `vwma_trend_confirmed` | trend continuation và after-buy health | Nên làm P1/P2 |
| Volume at Price | volume profile theo vùng giá daily gần trigger | hỗ trợ/kháng cự quanh trigger/invalidation | P2, vì cần thiết kế cẩn thận |
| Anti-volume stop/risk idea | không lấy nguyên công thức; chuyển thành cảnh báo stop khi giá giảm trên volume tăng | BUY report, bear-trap caution | P2 |

## Những phần chưa nên dùng

| Nhóm | Lý do |
|---|---|
| Tick VWAP chuẩn | Cần dữ liệu intraday/tick; daily OHLCV chỉ có thể làm proxy thô |
| Block vs nonblock money flow | Không có dữ liệu giao dịch theo lệnh/lô lớn |
| Capital Weighted Volume | Cần vốn hóa/free-float point-in-time; dữ liệu hiện tại chưa đủ sạch để làm headline |
| Market breadth nâng cao | Có thể làm cho market dashboard, nhưng chưa nên trộn vào từng event mẫu hình |
| Công thức TTI/VPCI đầy đủ làm tín hiệu mua bán | Có thể nghiên cứu sau; không nên đưa ngay vào BUY scanner như một rule mạnh trước khi backtest |

## Volume feature contract v1

Lớp kỹ thuật nên tạo một module chung, ví dụ `scanner/volume_features.py`, nhận daily OHLCV đã chuẩn hóa và trả ra các trường sau:

### Trường cơ bản

| Field | Ý nghĩa |
|---|---|
| `volume_ratio_20` | Volume hiện tại / median volume 20 phiên trước |
| `value_ratio_20` | Giá trị giao dịch hiện tại / median value 20 phiên trước, nếu có value |
| `volume_z20` | Độ lệch chuẩn hóa của volume so với 20 phiên |
| `volume_surge_flag` | True nếu volume vượt ngưỡng mạnh, ví dụ >= 1.5x hoặc z-score cao |
| `zero_volume_rate_20` | Tỷ lệ phiên không có thanh khoản trong 20 phiên |
| `liquidity_bucket` | high / mid / low theo giá trị giao dịch và zero-volume |

### Trường setup

| Field | Ý nghĩa |
|---|---|
| `pattern_volume_slope` | Độ dốc volume trong vùng hình thái |
| `pattern_volume_contraction_ratio` | median volume trong mẫu / median volume trước mẫu |
| `quiet_setup_flag` | Mẫu đang nén với volume giảm hoặc không bùng nổ bất thường |
| `noisy_setup_flag` | Mẫu bị nhiễu vì volume spike trái chiều hoặc zero-volume cao |

### Trường confirmation

| Field | Ý nghĩa |
|---|---|
| `breakout_volume_ratio_20` | Volume phiên xác nhận / median volume 20 phiên trước |
| `breakout_value_ratio_20` | Value phiên xác nhận / median value 20 phiên trước |
| `breakout_volume_confirmed` | True nếu phá vỡ đi kèm participation đủ rõ |
| `breakout_close_volume_phase` | Một trong bốn pha: up_confirmed, up_weak, down_confirmed, down_drying |

### Trường hậu phá vỡ

| Field | Ý nghĩa |
|---|---|
| `post_breakout_volume_decay_10d` | Volume sau phá vỡ có xẹp ngay không |
| `post_breakout_accumulation_score` | Điểm cộng nếu giá giữ vùng xác nhận và volume không phân phối |
| `adverse_volume_warning` | Cảnh báo nếu giá giảm bất lợi đi kèm volume tăng |

### Trường indicator P1

| Field | Ý nghĩa |
|---|---|
| `obv_slope_20` | Độ dốc OBV 20 phiên |
| `vpt_slope_20` | Độ dốc Volume Price Trend 20 phiên |
| `mfi_14` | Money Flow Index 14 phiên |
| `mfi_slope_10` | Momentum của MFI |
| `vwma_fast_minus_slow` | VWMA ngắn hạn trừ VWMA dài hạn |
| `vwma_trend_confirmed` | True nếu giá và VWMA cùng xác nhận hướng |

## Tác động vào scanner

### BUY_SETUP

BUY_SETUP hiện đã có `median_value_20` và `zero_volume_rate_20`, nhưng chưa có khái niệm participation/confirmation. Nên nâng `_score_candidate` theo hướng:

- Cộng điểm nếu setup đang gần trigger nhưng volume trong thân mẫu co lại lành mạnh.
- Trừ điểm nếu volume spike bất thường trong vùng sideway khiến mẫu dễ là nhiễu.
- Trừ điểm nếu zero-volume cao dù mẫu hình đẹp.
- Cộng điểm nếu OBV/VPT/MFI đang tích lũy trong khi giá chưa phá trigger.

Không nên dùng volume làm hard gate cho mọi mẫu, vì nhiều pattern tốt vẫn có thể phá vỡ trên volume vừa phải. Volume nên là quality modifier, trừ những mẫu mà source gốc nhấn mạnh volume là điều kiện rất quan trọng như cup-with-handle hoặc một số breakout continuation.

### BUY_PULLBACK / confirmed event

Với các event đã xác nhận, nên thêm:

- `volume_quality_label`: mạnh / vừa / yếu / nhiễu.
- `pullback_volume_health`: kéo ngược trên volume thấp lành mạnh hơn kéo ngược trên volume cao.
- `adverse_volume_warning`: nếu MAE đang tăng và volume cũng tăng, giảm mức ưu tiên.

Điều này phù hợp với hướng hiện tại: mail chỉ là danh sách mở chart kiểm tra, không phải tín hiệu mua.

## Tác động vào thống kê

Nên chạy lại thống kê theo các nhóm volume:

| Split | Câu hỏi |
|---|---|
| `breakout_volume_ratio_20 >= 1.5` vs thấp hơn | Breakout có volume cao thật sự hit target nhiều hơn không? |
| `pattern_volume_contraction_ratio <= 1.0` vs cao hơn | Mẫu nén volume có sạch hơn mẫu nhiễu không? |
| `obv_slope_20 > 0` vs <= 0 | Dòng tiền tích lũy trước breakout có cải thiện target-first không? |
| `mfi_14` bucket | Mẫu quá nóng có fail nhiều hơn không? |
| `post_adverse_volume_warning` | Kéo ngược kèm volume cao có dự báo failure không? |

Chỉ sau khi các split này có hiệu quả qua walk-forward mới được đưa vào tradable layer như filter mạnh. Trước đó, chúng chỉ là diagnostic.

## Tác động vào BUY Candidate Scan

Email nên giữ đơn giản, nhưng PDF đính kèm có thể thêm một block “Khối lượng nói gì?”.

### Email nhanh

Thêm tối đa 2 cột/nhãn:

- `Sức xác nhận`: Mạnh / Vừa / Yếu.
- `Cảnh báo khối lượng`: Không / Kéo ngược có volume / Thanh khoản mỏng / Volume nhiễu.

Không nên đưa OBV, MFI, VWMA thô vào mail vì người đọc phổ thông sẽ khó hiểu.

### PDF chi tiết

Thêm một bảng nhỏ dưới chart:

| Dòng | Ý nghĩa đọc |
|---|---|
| Khối lượng trước xác nhận | Mẫu đang nén hay nhiễu |
| Khối lượng tại xác nhận | Có lực tham gia hay không |
| Khối lượng sau xác nhận | Đi tiếp khỏe hay đang phân phối |
| Dòng tiền phụ | OBV/MFI/VWMA ủng hộ hay mâu thuẫn |

## Ưu tiên triển khai

### P0 - Nên làm trước

1. Tạo `scanner/volume_features.py`.
2. Thêm test unit cho `volume_ratio`, `volume_phase`, OBV/VPT/MFI/VWMA cơ bản.
3. Gắn features vào `run_buy_setup_scan_watchlist.py`.
4. Gắn `volume_quality_label` vào `send_realtime_scan_email.py`.
5. Gắn block “Khối lượng nói gì?” vào `build_realtime_scan_pdf_report.py`.
6. Chạy lại BUY scanner và so sánh số ứng viên trước/sau.

### P1 - Sau khi P0 ổn

1. Chạy phân tích historical uplift theo volume split cho tất cả BUY-eligible chapters.
2. Thêm volume split vào realtime history ledger để học theo thời gian.
3. Thêm volume diagnostics vào stock historical profile.

### P2 - Nghiên cứu thêm

1. VPCI proxy trên daily OHLCV.
2. Volume-at-price quanh trigger/invalidation.
3. Market breadth/sector volume context.
4. Anti-volume stop-loss localized cho after-buy risk note.

## Rủi ro cần chặn

- Không được biến volume indicator thành “mua vì volume đẹp”.
- Không dùng tick/intraday logic khi dữ liệu chỉ là daily.
- Không tối ưu nhiều ngưỡng volume để ép score tradable tăng.
- Không trộn volume liquidity với volume confirmation: thanh khoản đủ giao dịch và volume xác nhận mẫu là hai khái niệm khác nhau.
- Không đưa chỉ báo phức tạp vào email nếu chưa chuyển thành ngôn ngữ đọc.

## Đề xuất quyết định

Nên triển khai `volume_confirmation_layer_v1` vào BUY Candidate Scan trước, vì đây là nơi lợi ích rõ nhất và ít rủi ro nhất: nó giúp lọc/giải thích ứng viên hiện tại, không đòi hỏi viết lại toàn bộ 63 chapter.

Sau đó mới chạy backtest phân nhóm volume để xem indicator nào thật sự nâng hiệu quả. Nếu kết quả tốt, đưa volume layer trở lại scanner/tradable layer như một filter chính thức.
