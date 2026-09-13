"""
Script to generate the V1 Codebase Audit and Gap Analysis Report as a Word Document (.docx).
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

OUTPUT_PATH = r"d:\CODE\rajasthani\trading_pipeline\V1_Codebase_Audit_and_Gap_Analysis_Report.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}>'
                      f'<w:top w:w="{top}" w:type="dxa"/>'
                      f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
                      f'<w:left w:w="{left}" w:type="dxa"/>'
                      f'<w:right w:w="{right}" w:type="dxa"/>'
                      f'</w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="D0D7DE"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="10" w:space="0" w:color="1F4E79"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout(doc, title, text, border_color="1F4E79", bg_color="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    run_title = p.add_run(f"■ {title}\n")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(10.5)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_section_heading(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

def add_sub_heading(doc, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    r.font.name = "Calibri"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

def add_status_table(doc, section_title, rows_data):
    add_sub_heading(doc, section_title)
    
    tbl = doc.add_table(rows=len(rows_data) + 1, cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    col_widths = [Inches(1.8), Inches(1.1), Inches(1.6), Inches(2.0)]
    for row in tbl.rows:
        for c_idx, w in enumerate(col_widths):
            row.cells[c_idx].width = w
            
    headers = ["Item / Specification", "Status", "Evidence (File : Line)", "Audit Notes & Observations"]
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, (item, status, evidence, notes) in enumerate(rows_data, start=1):
        if status == "Implemented":
            bg_status = "EAFaf1"
            color_status = RGBColor(0x1E, 0x84, 0x49)
        elif status == "Partially Implemented":
            bg_status = "FEF9E7"
            color_status = RGBColor(0xB9, 0x77, 0x0E)
        else: # Missing
            bg_status = "FDEDEC"
            color_status = RGBColor(0xC0, 0x39, 0x2B)
            
        bg_row = "FFFFFF" if r_idx % 2 == 1 else "F9FAFB"
        
        # Col 0: Item
        cell0 = tbl.cell(r_idx, 0)
        set_cell_background(cell0, bg_row)
        set_cell_margins(cell0, top=60, bottom=60, left=70, right=70)
        p0 = cell0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(item)
        r0.font.size = Pt(8.5)
        r0.font.bold = True
        r0.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
        
        # Col 1: Status
        cell1 = tbl.cell(r_idx, 1)
        set_cell_background(cell1, bg_status)
        set_cell_margins(cell1, top=60, bottom=60, left=70, right=70)
        p1 = cell1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(status)
        r1.font.size = Pt(8.5)
        r1.font.bold = True
        r1.font.color.rgb = color_status
        
        # Col 2: Evidence
        cell2 = tbl.cell(r_idx, 2)
        set_cell_background(cell2, bg_row)
        set_cell_margins(cell2, top=60, bottom=60, left=70, right=70)
        p2 = cell2.paragraphs[0]
        p2.paragraph_format.space_before = Pt(0)
        p2.paragraph_format.space_after = Pt(0)
        r2 = p2.add_run(evidence)
        r2.font.size = Pt(8.0)
        r2.font.name = "Consolas"
        r2.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        
        # Col 3: Notes
        cell3 = tbl.cell(r_idx, 3)
        set_cell_background(cell3, bg_row)
        set_cell_margins(cell3, top=60, bottom=60, left=70, right=70)
        p3 = cell3.paragraphs[0]
        p3.paragraph_format.space_before = Pt(0)
        p3.paragraph_format.space_after = Pt(0)
        r3 = p3.add_run(notes)
        r3.font.size = Pt(8.0)
        r3.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        
    set_table_borders(tbl, "D0D7DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def build_audit_document():
    doc = docx.Document()
    
    # 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Set default style
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(5)
    
    # -------------------------------------------------------------
    # HEADER / TITLE BLOCK
    # -------------------------------------------------------------
    p_pre = doc.add_paragraph()
    p_pre.paragraph_format.space_before = Pt(10)
    p_pre.paragraph_format.space_after = Pt(2)
    r_tag = p_pre.add_run("TECHNICAL AUDIT & ARCHITECTURAL GAP ANALYSIS")
    r_tag.font.bold = True
    r_tag.font.size = Pt(10)
    r_tag.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("V1 Codebase Audit Report: Multi-Model XGBoost NIFTY 50 Trading Pipeline")
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Line-by-Line Implementation Verification, Mathematical Gap Analysis, and Technical Blueprint for V2")
    r_sub.font.size = Pt(12)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    
    # Metadata Table
    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_tbl.autofit = False
    meta_tbl.columns[0].width = Inches(2.2)
    meta_tbl.columns[1].width = Inches(4.3)
    
    meta_items = [
        ("Audited Codebase", "trading_pipeline/ (26 Python modules, 3 models, incremental engine)"),
        ("Environment / Runtime", "Python 3.12, XGBoost 3.4.1 (CUDA GPU), Polars 1.39, Pandera 0.29"),
        ("Audit Mode", "Inspection & Verification Only (No code modified)"),
        ("Auditing System", "Antigravity Advanced Agentic Quantitative Systems"),
    ]
    for idx, (lbl, val) in enumerate(meta_items):
        row = meta_tbl.rows[idx]
        set_cell_background(row.cells[0], "F2F4F7")
        set_cell_background(row.cells[1], "FAFAFA")
        set_cell_margins(row.cells[0], top=70, bottom=70, left=100, right=100)
        set_cell_margins(row.cells[1], top=70, bottom=70, left=100, right=100)
        
        p0 = row.cells[0].paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(lbl)
        r0.font.bold = True
        r0.font.size = Pt(9)
        r0.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        
        p1 = row.cells[1].paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(val)
        r1.font.size = Pt(9)
        r1.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        
    set_table_borders(meta_tbl, "D0D7DE")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------
    # SECTION 1: MASTER ITEM-BY-ITEM AUDIT TABLE
    # -------------------------------------------------------------
    add_section_heading(doc, "1. Item-by-Item Implementation Audit (Sections A through N)")
    
    p = doc.add_paragraph()
    p.add_run(
        "Every single architectural decision, formula, and engineering technique agreed upon during system design has been "
        "audited directly against the actual codebase files. Statuses are strictly classified as Implemented, Partially Implemented, or Missing."
    )
    
    # Section A
    sec_a_data = [
        ("Separate feature_engine/ module exists & genuinely used", "Implemented", "feature_engine/__init__.py:1-20\npipeline.py:19-164", "Fully modular; imported and orchestrated across train_master.py, inference.py, and test_inference_aux.py."),
        ("Mid-Price = (Bid + Ask) / 2 used everywhere without raw Close", "Partially Implemented", "aggregator.py:45\npipeline.py:52\nmodels/model_1_direction.py:85", "Bid/Ask does not exist in 1-min raw data; mid-price is approximated via (open+close)/2. However, target definitions across Models 1, 2, 3 and backtest execution still fall back to raw close."),
        ("5-minute OHLC aggregation from 1-min data (true resampling)", "Implemented", "aggregator.py:28-47", "True Pandas resampling with label='left', closed='left' and full OHLCV aggregation."),
        ("All 14 rolling Z-score formulas implemented as distinct functions", "Partially Implemented", "zscore.py:66-410", "14 Z-score methods exist, but distinct standalone functions for rolling mean, std, simple return, log return, and volatility were not exposed as independent callable methods (embedded internally). Dead code exists after return statements in return_zscore (L168), log_return_zscore (L203), volatility_zscore (L261), and price_vs_ma_zscore (L293)."),
        ("Rolling windows reset at start of each trading day", "Implemented", "zscore.py:41-61\nindicators.py:12-28", "_get_session_groups detects gaps > 4 hours and groups by cumulative session ID, strictly resetting windows."),
        ("Every rolling feature lagged with shift(1)", "Implemented", "zscore.py:86, 128, 164\nindicators.py:48, 80", "Systematically applies shift(1) across rolling stats and diffs before computing features, preventing lookahead."),
        ("Exhaustion feature: 20-min Volume Z-score", "Implemented", "indicators.py:123\npipeline.py:88-91", "Computed via volume_zscore(df['volume'], window=20) and packaged into exhaustion_features()."),
        ("Exhaustion feature: Distance from VWAP = (Mid - VWAP) / VWAP", "Implemented", "indicators.py:92-110", "Session-resetting VWAP computed with shift(1); falls back to shifted mid-price if volume is zero."),
        ("Exhaustion feature: 14-period Wilder's RSI", "Implemented", "indicators.py:30-63", "True exponential smoothing via ewm(com=period-1) on shifted price differences."),
        ("NEW: Relative Volume vs same-time-of-day historical average", "Missing", "Not found in feature_engine/", "No minute-of-day historical baseline volume normalization exists in V1."),
        ("NEW: Bollinger Band %B / Distance-from-Upper-Band", "Missing", "Not found in feature_engine/", "Neither %B nor distance from upper/lower Bollinger Bands is implemented in V1."),
    ]
    add_status_table(doc, "A. Core Feature Engineering (feature_engine/ module)", sec_a_data)
    
    # Section B
    sec_b_data = [
        ("Dynamic 50x50 correlation matrix across all Nifty 50", "Implemented", "matrix_builder.py:44-96", "Scans all Parquet files, aligns on timestamps, and computes cross-asset correlation matrix."),
        ("Three distinct correlation dimensions (raw, binary, spike)", "Partially Implemented", "matrix_builder.py:95-111", "All 3 matrices are computed and saved to disk. However, peer_map.json extraction (L114-127) only uses the raw returns matrix; binary and spike matrices are completely ignored downstream."),
        ("All 50 stocks aligned to common timestamp grid", "Implemented", "matrix_builder.py:84-89", "Iterative Polars outer_coalesce join on timestamp with nulls filled with 0.0."),
        ("Top correlated & inverse-correlated peer extraction", "Implemented", "matrix_builder.py:114-127", "Extracts top 3 positive and top 3 negative peers per symbol into artifacts/peer_map.json."),
        ("Lagged peer momentum joined as feature column", "Implemented", "cross_asset.py:79, 86-160\ntrain_master.py:268", "Injected under rank-normalized names peer_pos_1_return_lag1 ... peer_neg_3_return_lag1 with shift(1) 5-bar peer returns."),
        ("Matrix computed statically on historical data", "Implemented", "matrix_builder.py:136-138\ntrain_master.py:53", "Executed as an offline batch build; train_master.py and inference.py only read the static peer_map.json."),
    ]
    add_status_table(doc, "B. Correlation Matrix Engine", sec_b_data)
    
    # Section C
    sec_c_data = [
        ("Model 1 (Directional): target = next 5-min close > current close", "Implemented", "model_1_direction.py:85-86", "Computed via (close.shift(-1) > close).astype(int)."),
        ("Model 2 (Price): target = next 1-min return with explicit alignment", "Implemented", "model_2_price.py:86-104", "Explicitly shifts 5-min index by +5 minutes and reindexes the first 1-min bar's return (C - O) / O."),
        ("Model 3 (Exhaustion): target = forward 10-min max drawdown", "Implemented", "model_3_exhaustion.py:95-104", "Reverse rolling minimum with shift(-1) looking strictly forward t+1..t+10; target clipped to [-0.15, 0.0]."),
        ("early_stopping_rounds explicitly set on all 3 models", "Implemented", "model_1_direction.py:111\nmodel_2_price.py:126\nmodel_3_exhaustion.py:125", "Explicitly set to 50 on all three model initializations."),
        ("Regularization parameters explicitly set and reasoned about", "Implemented", "model_1_direction.py:47-55\nmodel_2_price.py:46-54\nmodel_3_exhaustion.py:52-60", "Configures reg_alpha=0.1, reg_lambda=1.0, max_depth=5/6, subsample=0.8, colsample_bytree=0.8, min_child_weight=5/10."),
        ("GPU device correctly configured (device='cuda')", "Implemented", "model_1_direction.py:43-44\nmodel_2_price.py:42-43\nmodel_3_exhaustion.py:48-49", "tree_method='hist', device='cuda' is consistent across all model definitions and matches XGBoost 3.4.1 requirements."),
    ]
    add_status_table(doc, "C. The 3-Model Architecture — Targets & Configuration", sec_c_data)

    # Section D
    sec_d_data = [
        ("Data streamed one company at a time", "Implemented", "train_master.py:222-234", "Strict sequential loop reading single .parquet files via Polars."),
        ("xgb_model= continuation parameter passed on every fit after first", "Implemented", "train_master.py:405-444\nmodel_1_direction.py:121", "Models 1, 2, and 3 pass previous .json model path to xgb_model in fit()."),
        ("Replay buffer: 5% random sample stored to disk", "Implemented", "train_master.py:84-149", "Persists separate replay_buffer_5min.parquet and replay_buffer_1min.parquet."),
        ("Replay buffer actually loaded & mixed before each fit", "Implemented", "train_master.py:338-374", "Reads buffer from disk and concatenates with current company samples before splitting."),
        ("Periodic full retrain from replay buffer (every 25 companies)", "Implemented", "train_master.py:462-535", "Discards accumulated tree stack and fits fresh models on consolidated historical buffer."),
        ("Explicit del and gc.collect() after every company iteration", "Implemented", "train_master.py:249, 320, 537-543", "Deletes all local DataFrames and triggers explicit garbage collection; RAM remains bounded < 2 GB."),
        ("Atomic Parquet writes (.tmp + os.replace)", "Implemented", "train_master.py:73-82", "Writes to temp file then atomic os.replace to prevent Windows file locking crashes."),
    ]
    add_status_table(doc, "D. Incremental Learning Pipeline (Memory Management)", sec_d_data)

    # Section E
    sec_e_data = [
        ("Purged walk-forward CV implemented (not plain TimeSeriesSplit)", "Implemented", "validation/purged_cv.py:40-98", "Dedicated PurgedWalkForwardCV generator with purge and embargo parameters."),
        ("Purge gap removes boundary bars overlapping validation", "Implemented", "validation/purged_cv.py:84-85", "Removes last purge_gap bars from the training fold slice."),
        ("Embargo gap applied after validation fold", "Partially Implemented", "validation/purged_cv.py:87\ntrain_master.py:382", "Flaw: Line 87 sets val_start = train_end + self.embargo_gap, which places the embargo before validation rather than after validation. Furthermore, train_master.py only evaluates on the single final fold [-1]."),
        ("Hyperparameter tuning scored against purged CV split", "Partially Implemented", "validation/purged_cv.py:103-207", "optimize_hyperparameters evaluates Optuna trials against cv.split(). However, it is never called inside train_master.py; production models use hardcoded default hyperparameters."),
    ]
    add_status_table(doc, "E. Leakage Prevention & Validation", sec_e_data)

    # Section F
    sec_f_data = [
        ("Inference script loads all 3 saved models & combines outputs", "Implemented", "inference.py:65-80, 126-128\ntest_inference_aux.py:1070", "Loads weights from disk and evaluates 3 models in tandem."),
        ("Trading condition matches spec (M1 UP and M3 sharp drop)", "Implemented", "inference.py:154-157", "Checks direction == 1 and drawdown < EXHAUSTION_THRESHOLD (-0.003) for mean reversion setup."),
        ("Limit order entry price computed as Mid * (1 - Drop)", "Implemented", "inference.py:156\ntest_inference_aux.py:495", "Formula matches spec: entry_price = current_price * (1.0 - abs(drawdown))."),
        ("Final output signal includes direction, predicted price, entry price", "Implemented", "inference.py:187-200\ntest_inference_aux.py:527", "Unified dictionary schema containing timestamp, symbol, direction, confidence, predicted price move, drawdown, entry price, ATR, and shares."),
    ]
    add_status_table(doc, "F. Execution & Signal Logic", sec_f_data)

    # Section G
    sec_g_data = [
        ("Model 1 DOWN class used as sell/exit signal", "Implemented", "inference.py:162-165, 190\nbacktest/engine.py:187", "Emits MOMENTUM_SELL with direction = -1, which opens short positions in engine.py or exits long positions."),
        ("Mirrored Model 3 for maximum rally/pop", "Missing", "Not found in models/model_3_exhaustion.py", "Model 3 only computes forward low drawdown. No mirrored upside runup model exists."),
        ("Rule-based exit overlay (stop-loss, take-profit, trailing stop)", "Partially Implemented", "backtest/engine.py:198-220", "Only implements a naive time-based exit (if bars_held >= 5: close). Stop-loss %, take-profit %, trailing stop, and dynamic ATR exits are completely absent."),
    ]
    add_status_table(doc, "G. Buy/Sell Symmetry & Exit Logic", sec_g_data)

    # Section H
    sec_h_data = [
        ("Transaction cost model applied on entry and exit", "Implemented", "backtest/engine.py:33-70, 184, 209", "Models brokerage (0.03%), STT (0.1% sell-side only), and slippage (0.05%) on both legs."),
        ("Portfolio/capital tracking cash, positions, equity curve", "Implemented", "backtest/engine.py:147-221", "Simulates cash balance, active share count, trade log, and cumulative equity curve."),
        ("Expected-Value (EV) gating exceeding cost threshold", "Missing", "inference.py:158\ntest_inference_aux.py:497", "Gating only checks price_move > 0. There is no check requiring predicted return to exceed round-trip transaction costs (~0.08% - 0.18%)."),
        ("Model 1 label updated with cost threshold buffer", "Missing", "models/model_1_direction.py:86", "Uses naive (next_close > close).astype(int). Any tiny positive move (even 1 paisa) is labeled UP, regardless of friction."),
        ("Limit order fill probability modeled in backtest", "Implemented", "backtest/engine.py:72-110, 173-178", "Rejects fills if price never touched limit; scales probability based on depth into the bar."),
    ]
    add_status_table(doc, "H. Cost-Aware Trading Logic", sec_h_data)

    # Section I
    sec_i_data = [
        ("Meta-labeling", "Missing", "Not found in codebase", "No secondary sizing/execution filter model exists."),
        ("Triple-barrier labeling", "Missing", "Not found in codebase", "All models use fixed time-horizon labels."),
        ("Confidence thresholding", "Implemented", "inference.py:158, 162\ntest_inference_aux.py:497", "Evaluates p_up >= conf_threshold (default 0.52, configurable to 0.60+)."),
        ("Ensemble agreement requirement", "Partially Implemented", "inference.py:158-164", "Requires Model 1 direction and Model 2 return drift to agree. However, cross-asset peer signals are only passive inputs to Model 1, and Model 3 is used for dip diversion rather than an independent agreement gate."),
        ("Kelly Criterion position sizing", "Missing", "backtest/risk_manager.py:99-145", "Sizing is based on ATR volatility parity, not Kelly probability/odds formulas."),
        ("Probability calibration (Platt scaling / isotonic)", "Missing", "models/model_1_direction.py:146", "Uses raw uncalibrated sigmoid output from model.predict_proba()."),
        ("3-class (UP / DOWN / FLAT) label alternative", "Missing", "models/model_1_direction.py:45, 86", "Model 1 remains strictly binary (binary:logistic)."),
    ]
    add_status_table(doc, "I. Advanced Labeling & Signal-Quality Techniques", sec_i_data)

    # Section J
    sec_j_data = [
        ("Raw CSVs converted to partitioned Parquet", "Partially Implemented", "preprocess_to_parquet.py:32, 160", "Converts CSVs into single flat files per symbol (DATA/parquet/{symbol}.parquet), NOT partitioned by date/symbol subdirectories."),
        ("Polars genuinely used for heavy operations", "Implemented", "preprocess_to_parquet.py:64\nmatrix_builder.py:64\ntrain_master.py:102", "Polars lazy scanning, alignment joins, and atomic file writes are genuinely operational throughout the pipeline."),
        ("Schema validation layer actually invoked in pipeline", "Partially Implemented", "preprocess_to_parquet.py:36-50, 118", "Pandera schema validation is implemented in preprocessing, but only checks the first 1,000 rows. It is not invoked in the live inference or feature pipelines."),
    ]
    add_status_table(doc, "J. Data Engineering Layer", sec_j_data)

    # Section K
    sec_k_data = [
        ("Backtest reports Sharpe, Sortino, Drawdown, Win Rate, PF", "Implemented", "backtest/engine.py:235-281", "Full institutional metrics suite calculated and printed."),
        ("Position-sizing rule exists & called before signal finalized", "Implemented", "backtest/risk_manager.py:99-145\ninference.py:176", "Calls calculate_position_size() with 14-period ATR volatility adjustments before saving signal."),
        ("Correlation-aware exposure cap reading correlation matrix", "Partially Implemented", "backtest/risk_manager.py:148-207\ninference.py:174", "Logic reads peer_map.json and evaluates can_open_position(). However, register_position() is never called in inference.py, leaving open_positions empty during standalone inference runs."),
        ("Daily max-loss circuit breaker & manual kill switch", "Partially Implemented", "backtest/risk_manager.py:43, 209\ninference.py:217", "Kill Switch is fully wired and halts execution immediately. However, CircuitBreaker class is never instantiated or invoked in inference.py or backtest/engine.py."),
    ]
    add_status_table(doc, "K. Backtest, Risk & Portfolio Layer", sec_k_data)

    # Section L
    sec_l_data = [
        ("PSI or KS-test based drift detection exists", "Implemented", "monitoring/drift_detector.py:34\ntrain_master.py:207, 325", "Establishes feature distribution baseline and checks PSI for every company, logging to drift_log.json."),
        ("Per-company validation metrics logged during training", "Implemented", "monitoring/metrics_logger.py:24\ntrain_master.py:206, 422", "Lightweight JSON logger tracks per-company accuracy, RMSE, and sample size in artifacts/training_log.json."),
    ]
    add_status_table(doc, "L. Model Monitoring & Drift Detection", sec_l_data)

    # Section M
    sec_m_data = [
        ("paper_trading_only configuration flag gating live orders", "Missing", "Not found in codebase", "No execution environment routing or paper trading gate exists."),
        ("Algo-ID / strategy-registration hooks", "Missing", "Not found in codebase", "No SEBI/exchange algo registration tags or compliance audit hooks exist."),
    ]
    add_status_table(doc, "M. Regulatory Compliance Layer", sec_m_data)

    # Section N
    sec_n_data = [
        ("Model classes exist as separate files with own logic", "Implemented", "models/model_1_direction.py\nmodels/model_2_price.py\nmodels/model_3_exhaustion.py", "Separate modular files encapsulate class initialization, preprocessing, fitting, and serialization."),
        ("train_master.py imports and orchestrates classes", "Implemented", "train_master.py:41-43, 201, 417", "Orchestrates the training pipeline cleanly without duplicating model internals."),
        ("Inference reuses exact model classes", "Implemented", "inference.py:48-50, 75\ntest_inference_aux.py:112", "Reuses DirectionalModel, PriceModel, and ExhaustionModel instances for all predictions."),
    ]
    add_status_table(doc, "N. Code Architecture", sec_n_data)

    # -------------------------------------------------------------
    # SECTION 2: CONSOLIDATED MISSING / PARTIALLY IMPLEMENTED LIST
    # -------------------------------------------------------------
    add_section_heading(doc, "2. Consolidated Gap Analysis: Missing & Partial Items (Ordered by Priority)")
    
    p = doc.add_paragraph()
    p.add_run(
        "To enable the V2 development model to execute immediately, the identified deficiencies are grouped below in strict "
        "hierarchical order of mathematical severity:"
    )

    # Priority 1
    add_callout(
        doc,
        "Priority 1: Leakage, Correctness & Mathematical Flaws",
        "1. Embargo Gap Misplacement in Purged CV: In validation/purged_cv.py (line 87), the embargo gap is applied before validation rather than after validation. In De Prado's framework, purge precedes validation, and embargo follows validation. Additionally, train_master.py only trains on the single final fold [-1] rather than full walk-forward CV.\n"
        "2. Optuna Hyperparameter Tuning Not Wired: optimize_hyperparameters() is implemented in purged_cv.py but is never called in train_master.py; production models train on fixed default parameters.\n"
        "3. Dead Code in Feature Engine: In feature_engine/zscore.py, duplicate unreachable blocks exist after return statements in return_zscore, log_return_zscore, volatility_zscore, and price_vs_ma_zscore.\n"
        "4. Unused Binary & Spike Correlation Matrices: matrix_builder.py computes and saves binary and spike matrices, but peer_map.json extraction uses only raw return correlations.\n"
        "5. ExposureManager State Tracking Disconnect: inference.py evaluates can_open_position() but never calls register_position(), leaving open positions permanently empty in single-symbol inference.\n"
        "6. CircuitBreaker Unwired: CircuitBreaker in risk_manager.py is never instantiated or called in inference.py or backtest/engine.py.",
        border_color="C0392B",
        bg_color="FDEDEC"
    )

    # Priority 2
    add_callout(
        doc,
        "Priority 2: Cost-Awareness & Signal-Quality Gaps",
        "7. Expected-Value (EV) Gating Missing: inference.py only checks price_move > 0. Trades are taken on sub-basis-point micro-moves that cannot overcome round-trip transaction costs (~0.08% - 0.18%).\n"
        "8. Cost-Buffered Model 1 Target Missing: model_1_direction.py labels any move > 0 as UP. It should require pct_move > round_trip_cost_pct.\n"
        "9. Rule-Based Exit Overlay Missing: backtest/engine.py only has a hardcoded 5-bar time exit. Stop-loss %, take-profit %, trailing stops, and ATR exits are missing.\n"
        "10. Mirrored Model 3 for Maximum Rally/Pop Missing: Model 3 only predicts downward drops (drawdown); no mirrored model exists for upward exhaustion, short setups, or profit-taking targets.\n"
        "11. Probability Calibration Missing: Raw sigmoid probabilities from XGBoost are used directly without Platt scaling or isotonic regression calibration.\n"
        "12. Triple-Barrier & Meta-Labeling Missing: Neither triple-barrier labeling nor a secondary trade-filtering meta-model is implemented.\n"
        "13. 3-Class Label Alternative (UP / DOWN / FLAT) Missing: Model 1 remains strictly binary.\n"
        "14. Kelly Criterion Position Sizing Missing: ATR volatility-parity sizing is implemented, but probability-adjusted Kelly sizing is not.\n"
        "15. Late-Proposed Indicators Missing: Relative Volume vs. same-time-of-day historical average and Bollinger Band %B / Upper Band Distance are missing from feature_engine/.",
        border_color="D35400",
        bg_color="FEF9E7"
    )

    # Priority 3
    add_callout(
        doc,
        "Priority 3: Data Engineering, Monitoring & Compliance Gaps",
        "16. Partitioned Parquet Structure: Parquet files are saved as single flat files per ticker rather than standard multi-level partitioned directories (symbol=X/year=Y/month=M/).\n"
        "17. Limited Pandera Invocation: Pandera validation only inspects the first 1,000 rows during preprocessing and is omitted from the streaming pipeline.\n"
        "18. Compliance Hooks Missing: paper_trading_only execution flag and Algo-ID registration hooks are completely absent.",
        border_color="2E75B6",
        bg_color="F2F7FA"
    )

    # -------------------------------------------------------------
    # SECTION 3: WHAT V1 ALREADY DOES WELL
    # -------------------------------------------------------------
    add_section_heading(doc, "3. Summary: What V1 Already Does Well")
    
    p = doc.add_paragraph()
    p.add_run(
        "The V1 codebase possesses a strong, production-grade foundation in memory management, feature engineering hygiene, "
        "and modular architecture. The streaming data pipeline in train_master.py successfully executes incremental multi-company "
        "training on GPU using xgb_model= warm-starts without RAM bloat, supported by dual-timeframe replay buffers (5-minute and 1-minute "
        "Parquet buffers) and periodic full retraining every 25 companies. Feature engineering strictly eliminates lookahead bias via "
        "systematic shift(1) operations and overnight session boundary resets. The 3-model separation of concerns (5-min direction, "
        "1-min immediate drift, and 10-min forward drawdown) is mathematically sound, and the backtesting engine includes realistic Indian "
        "market frictions (Zerodha brokerage, sell-side STT, slippage, and limit order fill probability modeling)."
    )

    # -------------------------------------------------------------
    # SECTION 4: EXPLICIT GPU CONFIGURATION CONFIRMATION
    # -------------------------------------------------------------
    add_section_heading(doc, "4. Explicit Confirmation of GPU Device Configuration")
    
    p = doc.add_paragraph()
    p.add_run(
        "The production training configuration matches the working standalone test and is correctly configured for GPU execution across all models:\n"
        "• In models/model_1_direction.py (lines 43-44): 'tree_method': 'hist', 'device': 'cuda'\n"
        "• In models/model_2_price.py (lines 42-43): 'tree_method': 'hist', 'device': 'cuda'\n"
        "• In models/model_3_exhaustion.py (lines 48-49): 'tree_method': 'hist', 'device': 'cuda'\n"
        "• In validation/purged_cv.py (lines 168, 182): tree_method='hist', device='cuda'\n\n"
        "Verification Details: The installed environment runs XGBoost 3.4.1. In XGBoost versions >= 2.0, tree_method='hist' combined with "
        "device='cuda' is the official syntax (older tree_method='gpu_hist' is deprecated). Verification on the local interpreter confirms that "
        "xgb.XGBClassifier(tree_method='hist', device='cuda') initializes without warning or CPU fallback. The production pipeline adheres "
        "strictly to this working configuration across all models and retraining routines."
    )
    
    # Save Document
    doc.save(OUTPUT_PATH)
    print(f"Successfully generated Word Document at:\n-> {OUTPUT_PATH}")

if __name__ == "__main__":
    build_audit_document()

