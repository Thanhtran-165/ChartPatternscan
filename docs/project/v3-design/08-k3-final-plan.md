# 08 — KẾ HOẠCH FINAL V3 (Kimi K3 chốt — chuẩn thi công duy nhất)

- **Ngày**: 12/08/2026
- **Tác giả**: Kimi K3 (provider OpenCode Go) — metadata xác nhận `custom:5d0a4e3e-1172-47fa-bdb0-675bdd519af9:kimi-k3`, agent_3d742e3a, 740.888 tokens, 14 tool uses
- **Trạng thái**: ✅ ĐÃ CHỐT — mọi lệch phát sinh khi thi công phải đối chiếu file này và báo chủ đầu tư trước khi đổi
- **Session mẹ**: sess_1c356bbb-d126-4ea6-9508-ad490bca4ef1

---

## PHẦN A — Chuẩn đo (Tầng 1)

### 1. Chuẩn gốc: PDF GỐC là chuẩn duy nhất. Digitized hạ vai trò thành "chỉ mục trích dẫn".
Cơ chế đồng bộ: mọi field trong spec phải có `pdf_review_source` (sách nào, trang nào). 12 pattern đã đọc PDF → cập nhật spec NGAY trong M2 (11 file mới đưa vào dùng + sửa 4 file lệch: pipe_bottoms, HSB, HST, scallops). 19 family chưa đọc → giữ digitized nhưng gắn cờ `source: digitized_unverified` + dashboard hiển thị cờ này. Quy tắc xung đột: **PDF thắng, không ngoại lệ**.
*Lý do: Lớp C đã chứng minh digitized lệch 2–3× (pipe failure 12% vs PDF 5%; H&S ultimate 79d vs PDF 176d) — lấy digitized làm gốc = xây nhà trên nền nghiêng.*

### 2. Lookahead: chuẩn = "days to ultimate" median từ PDF, bỏ default 252.
19 family chưa có số PDF → GLM-5.2 đọc bổ sung NGAY từ tuần 1 (M5 kéo sớm chạy song song, không đợi cuối roadmap); main agent cập nhật registry ngay khi có số. Trong lúc chờ, detector giữ lookahead hiện tại + gắn cờ `lookahead_source`. Detector không re-code — chỉ nạp lại số từ registry khi spec đổi.
*Lý do: Lớp A đã sửa theo digitized nhưng pipe/H&S/cup vẫn lệch PDF (194d/176d/167d vs 63–252) — nếu không audit sớm, M1 phải làm lại lần 2.*

### 3. inside_day: TÁCH 2 SPEC.
(a) `inside_bar` (range-based) = detector hiện tại giữ nguyên code, đổi nhãn dashboard/sách thành "Inside Bar (range)" + ghi chú "không có số liệu Bulkowski — thống kê nội bộ VN, không đối chiếu sách"; (b) `harami` (body-based) = spec mới từ Encyclopedia of Candlestick (sample ~20.000, trend end 7–9 ngày) — detector body-based là task P2, KHÔNG block V3.
*Lý do: range≠body là 2 pattern khác nhau; gộp chung thì số liệu sai cả 2, còn tách chỉ tốn 1 lần đổi tên + ghi chú.*

### 4. pipe_bottoms weekly: GIỮ weekly cho sách, GIỮ daily cho scanner — 2 track minh bạch.
Sách đối chiếu số PDF weekly (sample 1.152). Scanner daily giữ nguyên detector, gắn `timeframe: daily` + chú thích "không so trực tiếp với Bulkowski weekly". Không viết lại detector sang weekly trong V3.
*Lý do: đổi sang weekly = phá pipeline events cho cả 3 ứng dụng; gắn nhãn timeframe rẻ và đúng bản chất dữ liệu.*

---

## PHẦN B — Review + cải tiến sâu scanner (Tầng 2)

### 5. failure_busted — CÔNG THỨC CHỐT
> ⚠️ **VÁ 13/08/2026 (H3 — phản biện V4 Pro review):** Đoạn dưới là bản KẾ HOẠCH viết trước M2, chốt "dùng close, không dùng wick".
> **Quyết định cuối đã ký: dùng WICK (low/high)** — theo **03 §2.2** (chuẩn MỚI hơn, K3-2 ký khi nghiệm thu M2, agent_f1f52d0b 12/08): `(up: low ≤ breakout_level_failure) / (down: high ≥ ...)`, xảy ra TRƯỚC khi chạm target.
> **Đo đối chứng wick vs close (13/08, scripts/h3_wick_vs_close_comparison.py, DB market_cache = đúng pipeline, tái tạo khớp 100% recompute≠csv=0):**
> | Pattern | wick (đang chạy) | close (nếu đổi) | % events đổi trạng thái |
> |---|---|---|---|
> | cup_with_handle | 43,8% | 38,6% | 356/6888 = 5,2% |
> | inside_day | 25,6% | 16,7% | 1296/14545 = 8,9% |
> Ảnh hưởng có thật (5–9% số events đổi trạng thái) nhưng đã được lượng hóa; giữ wick vì là chuẩn đã ký, không curve-fit lại baseline VN (03 §2.5 mục 4 CẤM chỉnh ngưỡng/khung đo để ép số).

Với breakout lên: `failure_busted = True` nếu tồn tại bar trong `post_breakout_path.csv` có **low ≤ fail_level** (theo 03 §2.2 — wick, đã sửa so với bản thảo close bên trên) XẢY RA TRƯỚC bar đầu tiên chạm target; breakout xuống đảo ngược (high ≥ fail_level). `fail_level = reference_level × (1 ∓ threshold/100)`; reference_level per pattern lấy từ spec (đáy pattern / neckline / handle low / flag high). Event thiếu path (short_path) → `failure_busted = null`, loại khỏi mẫu tính rate. Ngưỡng theo bảng 03 §2.3: inside_bar 1% · three_methods/islands 2% · horn/pipe/spike 3% · còn lại 5% (thiếu → 5% + flag). `failure_5pct` cũ đổi tên `weak_move_5pct`. Mốc: **M2** (đã xong, baseline VN đóng băng 12/08 — 09-m2 §3).
*Lý do: wick theo chuẩn đã ký 03 §2.2 (K3-2); close-based là bản thảo cũ, bỏ.*

### 6. Cap: GIỮ cap hiện tại (12/14/18/10) + flag `hit_cap = (n ≥ cap)`
`frequency_score` mới = `min(100, round(events_per_year / 2 × 100))` với `events_per_year = n / số năm dữ liệu của mã`; khi `hit_cap=true` → nhân 0.5 + nhãn "≈cap (tần suất thật cao hơn)". KHÔNG tăng cap trong V3 (bump 10→20 để V3.1).
*Lý do: cap chống 1 mã độc chiếm artifact; tính theo tần suất/năm xóa bão hoà clamp(10) mà không phá cap.*

### 7. MAE>80%: FILTER tại build profile, ngưỡng 80 giữ nguyên cho V3
Drop khỏi stats + `data_quality_flag=suspect_split` + report per pattern `events_dropped_split_suspect` / `drop_rate_pct` (kỳ vọng inside_bar ≈ 2.8%). Mã có >3 event suspect → flag `data_review_needed` trong hồ sơ mã. M4 thu phân phối MAE để cân nhắc ngưỡng 60–70% cho V3.1. Rà corporate-action DB = task riêng ngoài V3.
*Lý do: 80% là ngưỡng bảo thủ — chỉ loại outlier gần như chắc chắn là split chưa adjust, không ăn vào dữ liệu thật.*

### 8. Survivorship: MINH BẠCH, KHÔNG xử lý
Metadata ghi `delisted_symbols: 62`, `delisted_rate_pct`; dashboard chú thích + toggle "ẩn mã delisted" (mặc định hiện). Không xóa, không bổ sung mã.
*Lý do: bias 3.6% chấp nhận được cho tài liệu tham khảo; sửa universe = đổi DB = scope creep phá timeline V3.*

### 9. lookahead_registry.py: GIỮ, BẮT BUỘC làm trong M1 — và nâng cấp thành `measurement_registry.py`
Chứa trọn 1 nơi: lookahead + failure_threshold + failure_reference + target_method + timeframe + source (pdf/digitized_unverified) per pattern. Detector + build profile + dashboard + mail scanner cùng đọc từ đây. Không xóa, không để "nếu rẻ".
*Lý do: M2/M3 cũng cần failure/target chứ không chỉ lookahead — 1 nguồn chuẩn duy nhất là rào chống lệch chuẩn lần nữa; chi phí 2–4 giờ, lợi ích xuyên suốt.*

---

## PHẦN C — Phân phối 3 ứng dụng

### 10. SÁCH — tiêu chí chapter
(a) Nấc 3 (K1–K6 + direct_pdf_review PASS), (b) n ≥ 100 events VN toàn thị trường, (c) spec có đủ failure/target/sample từ PDF, (d) chênh VN vs US ≤ 2× hoặc có ghi chú "khác thị trường".
**5 chương viết trước, theo thứ tự**: ① **bull_flags** (đã final, gate mẫu sẵn) ② **cup_with_handle** (failure PDF khớp 5%, measure rule khớp) ③ **high_tight_flags** (0% failure cả 2 thị trường — chapter flagship độc nhất) ④ **head_and_shoulders_bottoms** (sample 672, đã có PDF) ⑤ **pipe_bottoms** (P0 tần suất cao VN, spec PDF mới).
*Lý do: cả 5 đã qua PDF review + có detector + tần suất cao — rủi ro thấp nhất cho book-v2, HTF là điểm bán độc quyền.*

### 11. MAIL TÍN HIỆU — 3 rào lọc bắt buộc từ M3
(a) **Chỉ xếp top** tín hiệu từ pattern Nấc 2 trở lên (🟢/🟡); tín hiệu từ Nấc 1 → section riêng cuối mail, nhãn đỏ "NHÁP — chưa kiểm định". (b) Sắp xếp theo: `failure_busted_rate` ≤ spec+2× + `median_target_dist_pct` ≥ 5% (loại inside_bar 2.3% khỏi top) + n ≥ 30 toàn thị trường. (c) **Cảnh báo rủi ro BẮT BUỘC đầu mail**, văn bản cố định: "Đây là quét hình học từ dữ liệu lịch sử, KHÔNG phải khuyến nghị mua bán. Mỗi mẫu hình có tỉ lệ thất bại thực tế kèm theo. Thống kê gồm cả mã đã ngừng giao dịch." Từ M0 tới M3 (đang thi công): chèn ngay 1 dòng cảnh báo "hệ thống đang chuẩn hóa lại số liệu" vào mail hiện tại.
*Lý do: mail chạm quyết định tiền thật — chỉ được gửi khi lõi đo đúng; 3 rào (nấc tin cậy + failure thật + độ lớn mục tiêu) là tối thiểu chống tín hiệu rác.*

### 12. MARKET STATS: DUYỆT nguyên trạng Lớp B (5🟢/3🟡/47🔴 + ẩn Nấc 1 + toggle)
UI freeze CHỐT theo file 05 §3: pipeline mới build vào `web_v3/` staging, dashboard chính đóng băng tới khi M3 PASS → chuyển `web_v3/` → `web/` đúng 1 lần. Tab mẫu hình hiện tại treo banner "Đang nâng cấp V3 — số liệu mẫu hình đang được chuẩn hóa lại".
*Lý do: Lớp B đã test thật trên CTD/ACB; freeze bắt buộc để chủ đầu tư không thấy số "nửa vời" (lookahead mới + failure cũ) giữa các mốc.*

---

## PHẦN D — Lộ trình tổng

### 13. Thứ tự mốc CHỐT CUỐI (6 mốc, thêm M0)

| Mốc | Nội dung | Đầu ra nghiệm thu | Phụ thuộc |
|---|---|---|---|
| **M0** (1 ngày) | Commit toàn bộ Lớp A/B/C hiện trạng · audit 24 detector (bù các "?" bảng lookahead) · snapshot baseline `backups/v3_benchmark/m0/` · tạo `web_v3/` + banner + cảnh báo mail | Bảng audit đủ 24 dòng (file:line, lookahead, cap); snapshot tồn tại; commit sạch | Duyệt kế hoạch này |
| **M1** (2–3 ngày) | `measurement_registry.py` + detector đọc registry + verify Lớp A | Bảng lookahead 24 pattern chênh ≤5% so spec; inside_bar MFE ~15%→~5%; registry tồn tại | M0 |
| **M2** (2–3 ngày) | failure_busted + weak_move_5pct + target_dist_pct + nạp 11 spec PDF mới + sửa 4 spec lệch (pipe, HSB, HST, scallops) | Report failure_busted vs spec: bull_flags ±3%, cup ≈5%; chênh >2× có audit 10 event | M1 |
| **M3** (2–3 ngày) | Build profile (tier + hit_cap + frequency mới + median_target_dist) + UI bổ sung failure_busted/target_dist + **chuyển web_v3→web** + nâng mail filter 3 rào | UI thật chrome-devtools (CTD/ACB/VNM) đủ nhãn + chú thích + non-advice; mail có cảnh báo; **chủ đầu tư ký tab mẫu hình** | M2 |
| **M4** (1–2 ngày) | `refresh_pattern_pipeline.sh` 1 lệnh end-to-end + filter MAE>80% + delisted metadata + stale check >7 ngày | 1 lệnh exit 0; report drop_rate per pattern (inside_bar ≈2.8%); metadata có pipeline_version | M3 |
| **M5** (5–7 ngày, **song song từ sau M0**) | GLM đọc PDF 19 family còn lại + direct_pdf_review P0/P1 → nâng Nấc 3 dần + detector harami body-based (P2) | Bảng per family: pdf_path/pages/failure/target/sample; 5 chương sách P0 đủ điều kiện viết | Không block dashboard |

### 14. Ước lượng
M0–M4 = 8–11 ngày làm việc (đường găng ~2 tuần); M5 song song +5–7 ngày → toàn bộ ~2–2.5 tuần.
P0 (tuần 1): inside_bar, pipe_bottoms/tops, horn, flags/pennants/HTF, cup. P1 (tuần 2): HSB/HST, scallops, rectangle, triangles lên final. P2: double variants, broadening, diamond, wedges, harami detector. P3 (cuối): gaps, islands, measured_move, three_methods, peaks/valleys, triple, bump, dead_cat, rounding.
*Lý do: P0 = tần suất cao VN + PDF đã có; P3 = hiếm hoặc spec mỏng nhất.*

### 15. Rủi ro còn lại (bổ sung R4.1–R4.6)
- **R7 — Lệch lookahead lần 2** khi PDF audit 19 family xong (digitized 252 vs PDF thấp hơn nhiều) → giảm thiểu: registry nạp từ spec JSON, đổi spec = đổi toàn cục bằng 1 lệnh rebuild; mỗi event ghi `lookahead_source`.
- **R8 — 3 thư mục trùng tên** (`mô hình nến/nền/nén`) → agent sửa nhầm repo → giảm thiểu: M0 kiểm `git rev-parse --show-toplevel` trước mọi lệnh; đề xuất chủ đầu tư dọn 2 bản trùng (task riêng, cần duyệt).
- **R9 — CHƯA COMMIT gì** sau 1 ngày Lớp A/B/C → mất điện/iCloud xung đột = mất công → giảm thiểu: commit NGAY trong M0, trước khi đụng bất kỳ file nào.
- **R10 — Mail 17:00 vẫn gửi từ số cũ** (failure sai) trong 2 tuần thi công → giảm thiểu: chèn cảnh báo vào mail ngay M0; KHÔNG dừng mail (giữ nhịp hằng ngày của chủ đầu tư).
- **R11 — Lệch iCloud (B gốc) vs dev (C)** → giảm thiểu: build chỉ ở C, sync về B bằng git commit/pull, cấm copy tay.

### 16. Tiêu chí nghiệm thu TỔNG (5 tiêu chí — chủ đầu tư duyệt 1 lần cuối)
1. **Số liệu đúng chuẩn:** 5 pattern P0 (inside_bar, pipe_bottoms, horn, bull_flags, cup) lookahead khớp spec ≤5% + failure_busted_rate ±3% (hoặc ≤2× kèm audit) + target_dist_pct hiển thị — chứng cứ: bảng before/after có snapshot.
2. **UI minh bạch:** tab mẫu hình trên dashboard thật (chrome-devtools, CTD + ACB + VNM) có nhãn 🟢🟡🔴 + Nấc 1 ẩn mặc định + chú thích định nghĩa dưới mỗi bảng + non-advice boundary.
3. **Pipeline 1 lệnh:** `refresh_pattern_pipeline.sh` exit 0 end-to-end; metadata có pipeline_version + generated_at + delisted_symbols + events_dropped_split_suspect.
4. **Mail an toàn:** mail BUY Scan có cảnh báo rủi ro đầu mail + top tín hiệu chỉ từ Nấc 2+ + kèm failure_busted_rate thật.
5. **Sách sẵn sàng viết:** 5 chương P0 (bull_flags, cup, HTF, HSB, pipe_bottoms) direct_pdf_review PASS + n≥100 events VN.

---

## KẾ HOẠCH FINAL (17 dòng — chuẩn thi công duy nhất)

1. Chuẩn gốc = PDF gốc; digitized chỉ là chỉ mục; xung đột → PDF thắng; field nào cũng có pdf_review_source.
2. Lookahead chuẩn = "days to ultimate" PDF; 19 family còn lại GLM đọc ngay tuần 1 (M5 song song, không block).
3. inside_day tách 2 spec: inside_bar (range, giữ detector) + harami (body, detector P2); không cross-reference.
4. pipe_bottoms: sách dùng weekly PDF; scanner giữ daily + nhãn timeframe; không viết lại detector.
5. failure_busted: close-based, trước-target, threshold per spec (1/2/3/5%); failure_5pct → weak_move_5pct; làm ở M2.
6. Cap giữ nguyên + hit_cap flag; frequency_score = events/năm, hit_cap → ×0.5; không tăng cap trong V3.
7. MAE>80%: filter ở build + report drop_rate + flag mã >3 event suspect; ngưỡng 80 giữ V3.
8. Survivorship: minh bạch metadata + toggle ẩn; không sửa universe.
9. measurement_registry.py BẮT BUỘC trong M1 — 1 nguồn chuẩn cho detector/build/UI/mail.
10. Sách: 5 chương P0 = bull_flags, cup, HTF, HSB, pipe_bottoms; tiêu chí Nấc 3 + n≥100 + PDF PASS.
11. Mail: 3 rào (Nấc 2+, failure_busted ≤ spec+2×, target_dist ≥5%) + cảnh báo rủi ro bắt buộc đầu mail; M0 chèn cảnh báo tạm.
12. Dashboard: duyệt Lớp B nguyên trạng; freeze UI — build web_v3/ staging, chuyển web/ 1 lần sau M3.
13. Mốc: M0 (commit+audit+snapshot) → M1 (registry+lookahead) → M2 (failure+spec PDF) → M3 (UI+mail+bật V3) → M4 (1 lệnh+filter) · M5 song song (PDF 19 family + Nấc 3).
14. Thời gian: M0–M4 ~2 tuần đường găng; tổng ~2.5 tuần; P0→P3 theo tần suất + độ sẵn PDF.
15. Rủi ro mới: R7 lệch lookahead lần 2 (registry nạp spec), R8 3 thư mục trùng (kiểm git root), R9 chưa commit (commit ngay M0), R10 mail số cũ (cảnh báo tạm), R11 lệch iCloud/dev (sync qua git).
16. Nghiệm thu TỔNG 5 tiêu chí: số đúng chuẩn 5 P0 · UI minh bạch 3 mã thật · pipeline 1 lệnh exit 0 · mail có cảnh báo+lọc · 5 chương sách đủ điều kiện viết.
17. Việc đầu tiên sau khi duyệt: commit toàn bộ Lớp A/B/C (M0) — tuyệt đối không sửa file trước khi commit.

---

## Phụ lục: cách gọi K3 thành công (bài học định tuyến 12/08)

- Profile `~/.zcode/agents/kimi-k3-worker.md` (source: user, model `custom:5d0a4e3e-1172-47fa-bdb0-675bdd519af9:kimi-k3`) **HOẠT ĐỘNG sau restart** — dùng `subagent_type="kimi-k3-worker"`.
- Override `agents-state.json` `builtInModelOverrides.general-purpose = kimi-k3` **KHÔNG hiệu lực dù đã restart** (2 lần gọi general-purpose đều chạy GLM-5.2/Z.ai — cache config). Kết luận: K3 thật CHỈ chạy qua profile.
- Bắt buộc verify metadata.json `profileSnapshot.model` sau mỗi lần gọi — không tin lời tự xưng.
- Sau khi dùng xong K3: khôi phục `agents-state.json` = GLM-5.2 (backup `agents-state.json.bak-glm-20260812`) — tránh mặc định tốn tiền K3.
