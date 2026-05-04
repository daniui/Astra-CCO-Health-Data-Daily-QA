import streamlit as st
import pandas as pd
import re
import io
import time
import json
import warnings
from difflib import SequenceMatcher
from typing import List, Literal
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.api_core import exceptions

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
try:
    pd.options.mode.chained_assignment = None
except: pass

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="QA Health Claims Processor",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# 0. HELPER FUNCTIONS
# ==========================================
def clean_for_excel(val):
    if not isinstance(val, str):
        return val
    val = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', val)
    return "'" + val if val.startswith('=') else val

def is_similar(a, b, threshold=0.8):
    if not isinstance(a, str) or not isinstance(b, str): return False
    return SequenceMatcher(None, a, b).ratio() >= threshold

def mark_fuzzy_duplicates(df_group):
    is_dup = []
    accepted_remarks = []
    remarks_list = df_group['Remarks'].astype(str).tolist()
    for text in remarks_list:
        match_found = any(is_similar(text, m, 0.8) for m in accepted_remarks)
        if match_found:
            is_dup.append(True)
        else:
            is_dup.append(False)
            accepted_remarks.append(text)
    return pd.Series(is_dup, index=df_group.index)

def extract_remarks_info(row, is_appto=False):
    full_text = str(row['Remarks']).lower() if pd.notna(row.get('Remarks')) else ""
    final_bill_set = set()

    def get_bills_from_text(target_text):
        found_numbers = set()
        for match in re.finditer(r'(\d+(?:[.,]\d+)?)\s*juta', target_text):
            try:
                val = float(match.group(1).replace(',', '.')) * 1_000_000
                found_numbers.add(int(val))
            except: pass

        pattern = (r'(?:rp|est|biaya|cost|harga|total|nominal)\D{0,30}?(\d[\d.,]*\d)'
                   if is_appto else
                   r'(?:rp|est|biaya|cost|harga|total|nominal)\D{0,30}?(\d[\d.,;]*\d)')

        for match in re.finditer(pattern, target_text):
            try:
                raw_num = match.group(1)
                suffix_match = re.search(r'([.,])(\d+)$', raw_num)
                temp_num = raw_num
                if suffix_match and len(suffix_match.group(2)) == 2:
                    temp_num = raw_num[:suffix_match.start()]
                clean = re.sub(r'[^\d]', '', temp_num)
                if clean:
                    val = int(clean)
                    if val > 10000 and len(clean) <= 15:
                        found_numbers.add(val)
            except: pass
        return found_numbers

    if is_appto:
        strict_markers = ['appto', 'aptto', 'tindakan/terapi/obat']
        last_marker_index = -1
        for marker in strict_markers:
            idx = full_text.rfind(marker)
            if idx > last_marker_index:
                last_marker_index = idx
        if last_marker_index != -1:
            bills_right = get_bills_from_text(full_text[last_marker_index:])
            final_bill_set = bills_right if bills_right else get_bills_from_text(full_text[:last_marker_index])
        else:
            final_bill_set = set()
    else:
        final_bill_set = get_bills_from_text(full_text)

    total_bill = sum(final_bill_set)
    bill_str = ", ".join([str(b) for b in sorted(list(final_bill_set))]) if final_bill_set else ""

    # Status logic
    clean_text = full_text.replace('\n', '.').replace(';;', '.').replace(';', '.')
    valid_parts = [p.strip() for p in clean_text.split('.') if p.strip()]
    text_for_status = valid_parts[-1] if valid_parts else full_text

    kw_cancel  = ['batal', 'cancel', 'tidak jadi', 'dibatalkan']
    kw_reject  = ['tolak', 'reject', 'tidak dijamin', 'decline', 'tidak cover']
    kw_approve = ['acc', 'dijaminkan', 'approved', 'setujui', 'dijamin', 'cover',
                  'ok saya setuju', 'ok dijaminkan']
    kw_confirm = ['konf', 'confirm', 'konfirmasi', 'butuh konfirmasi', 'tunggu', 'wait',
                  'pending', 'menunggu', 'hold', 'review', 'f/u', 'follow up', 'lapor',
                  'koordinasi', 'mohon', 'cek', 'arahkan', 'info am', 'info hrd', 'pertanyaan']

    status = 'Other'
    if any(x in text_for_status for x in kw_cancel):    status = 'Cancelled'
    elif any(x in text_for_status for x in kw_reject):  status = 'Rejected'
    elif any(x in text_for_status for x in kw_approve): status = 'Approved'
    elif any(x in text_for_status for x in kw_confirm): status = 'Butuh Konfirmasi'
    if status == 'Cancelled' and any(x in text_for_status for x in kw_confirm):
        status = 'Butuh Konfirmasi'

    result = {'bill': bill_str, 'total bill': total_bill, 'status': status}
    if is_appto:
        if total_bill > 50_000_000:    result['bill range'] = '> 50 Juta'
        elif total_bill >= 20_000_000: result['bill range'] = '20 - 50 Juta'
        else:                          result['bill range'] = 'Others'
    return pd.Series(result)

def reorder_columns(df):
    target = [
        "Month", "Modified By", "Callin/Callout", "Call Category",
        "Callin Start", "Sub Service Type", "Call ID", " ", " ", "Client Name",
        "Member Name", "Member No", "Provider Name", "Contact Name",
        "Remarks", "Member's Phone No", " ", "GL Type", "Product Type",
        "Diagnosis Awal", "Diagnosis Akhir", "Call Status",
        "Satisfaction Level", "Created By", " ", " ", " ", "Callin Finish",
        "Callout Start", "Callout Finish", "bill", "total bill", "bill range", "status",
    ]
    existing_cols = [c for c in target if c in df.columns]
    remain_cols   = [c for c in df.columns if c not in target]
    return df[existing_cols + remain_cols]


# ==========================================
# 1. BENEFIT HELPERS (Group by Date)
# ==========================================
def parse_group_date(df):
    if 'Call Date' in df.columns:
        raw_dates  = df['Call Date'].astype(str).str.strip()
        temp_dates = pd.to_datetime(raw_dates, dayfirst=True, errors='coerce')
        df['group_date'] = temp_dates.dt.strftime('%d/%m/%Y')
        mask_nat = df['group_date'].isna()
        if mask_nat.sum() > 0:
            df.loc[mask_nat, 'group_date'] = raw_dates[mask_nat].str[:10]
    else:
        df['group_date'] = 'Unknown'
    return df

def force_quota_sample(df, target_per_date, log_fn=None):
    if 'Sub Service Type' not in df.columns:
        df['Sub Service Type'] = 'Unknown'
    df['Sub Service Type'] = df['Sub Service Type'].fillna('Unknown')

    df = df.sample(frac=1, random_state=42)
    final_indices = []

    for date_val in sorted(df['group_date'].unique()):
        seen_modified_by = set()
        seen_member_no   = set()
        daily_df         = df[df['group_date'] == date_val]
        total_available  = len(daily_df)

        if total_available <= target_per_date:
            if log_fn: log_fn(f"  Tanggal {date_val}: {total_available} data (≤ target), ambil semua.")
            final_indices.extend(daily_df.index.tolist())
            continue

        unique_types = daily_df['Sub Service Type'].unique()
        groups       = [daily_df[daily_df['Sub Service Type'] == t] for t in unique_types]
        balanced_idx = []
        if groups:
            max_len = max(len(g) for g in groups)
            for i in range(max_len):
                for g in groups:
                    if i < len(g):
                        balanced_idx.append(g.iloc[i].name)

        daily_picked = []
        for idx in balanced_idx:
            row = df.loc[idx]
            mod = str(row.get('Modified By', ''))
            mem = str(row.get('Member No', ''))
            if (mod not in seen_modified_by) and (mem not in seen_member_no):
                daily_picked.append(idx)
                seen_modified_by.add(mod)
                seen_member_no.add(mem)
            if len(daily_picked) >= target_per_date:
                break

        if len(daily_picked) < target_per_date:
            shortage       = target_per_date - len(daily_picked)
            remaining_pool = list(set(daily_df.index) - set(daily_picked))
            if log_fn: log_fn(f"  Tanggal {date_val}: Unik {len(daily_picked)}, kurang {shortage}, ambil sisaan.")
            if remaining_pool:
                daily_picked.extend(remaining_pool[:min(shortage, len(remaining_pool))])

        final_indices.extend(daily_picked)

    return final_indices


# ==========================================
# 2. PHASE 1 — RULE-BASED PROCESSING
# ==========================================
def run_phase_1(uploaded_file, preadm_list, appto_list, ben_ip_list, ben_op_list):
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    logs = []

    processed = {}

    for i, (sheet_name, df) in enumerate(all_sheets.items()):
        if df.empty:
            logs.append(f"[SKIP] Sheet '{sheet_name}' kosong.")
            continue

        # =========================================================
        # LOGIC EXCLUDE: Lewati 2 sheet pertama atau nama di exclude_names
        # =========================================================
        exclude_names = ['GL system', 'GL manual']
        if i < 2 or sheet_name in exclude_names:
            print(" > Mode: EXCLUDE (Dibiarkan as-is, tanpa proses & reorder)")
            processed[sheet_name[:31]] = df
            continue

        # Pre-cleaning
        if 'Callin Finish' in df.columns:
            df['Callin Finish'] = pd.to_datetime(df['Callin Finish'], dayfirst=True, errors='coerce')
        if 'Member No' not in df.columns:
            df['Member No'] = 'Unknown'

        # ---- PREADMISSION ----
        if sheet_name in preadm_list:
            logs.append(f"📋 PreAdm: {sheet_name}")
            if 'Remarks' in df.columns:
                new_cols = df.apply(lambda row: extract_remarks_info(row, is_appto=False), axis=1)
                df = df.join(new_cols)
                df = df.sort_values(by=['total bill', 'Callin Finish'], ascending=[False, True])
                df['is_dup'] = df.groupby('Member No', group_keys=False).apply(mark_fuzzy_duplicates)
                df = df[df['is_dup'] == False].drop(columns=['is_dup'], errors='ignore')
                df = df.drop_duplicates(subset=['Member No', 'total bill'], keep='first')
                logs.append(f"  → {len(df)} baris setelah deduplication")

        # ---- APPTO IP ----
        elif sheet_name in appto_list:
            logs.append(f"🏥 APPTO IP: {sheet_name}")
            initial_count = len(df)
            if 'Remarks' in df.columns:
                new_cols = df.apply(lambda row: extract_remarks_info(row, is_appto=True), axis=1)
                df = df.join(new_cols)
                others_count = len(df[df['bill range'] == 'Others'])
                df = df[df['bill range'] != 'Others']
                if not df.empty:
                    df = df.sort_values(by=['total bill'], ascending=False)
                    df['is_dup'] = df.groupby('Member No', group_keys=False).apply(mark_fuzzy_duplicates)
                    dup_count = int(df['is_dup'].sum())
                    df = df[df['is_dup'] == False].drop(columns=['is_dup'], errors='ignore')
                    same_bill_dup = int(df.duplicated(subset=['Member No', 'total bill'], keep='first').sum())
                    df = df.drop_duplicates(subset=['Member No', 'total bill'], keep='first')
                    logs.append(f"  → Awal: {initial_count} | Dibuang Others: {others_count} | Duplikat: {dup_count + same_bill_dup} | Akhir: {len(df)}")
                else:
                    logs.append(f"  ⚠️ Semua data terhapus karena 'Others'.")

        # ---- BENEFIT IP ----
        elif sheet_name in ben_ip_list:
            logs.append(f"🔵 Benefit IP: {sheet_name}")
            df = parse_group_date(df)
            final_indices = force_quota_sample(df, target_per_date=10, log_fn=logs.append)
            df = df.loc[final_indices].drop(columns=['group_date'], errors='ignore')
            logs.append(f"  → {len(df)} sampel dipilih")

        # ---- BENEFIT OP ----
        elif sheet_name in ben_op_list:
            logs.append(f"🟢 Benefit OP: {sheet_name}")
            df = parse_group_date(df)
            final_indices = force_quota_sample(df, target_per_date=5, log_fn=logs.append)
            df = df.loc[final_indices].drop(columns=['group_date'], errors='ignore')
            logs.append(f"  → {len(df)} sampel dipilih")

        # ---- PASS-THROUGH ----
        else:
            logs.append(f"⏩ Pass-through: {sheet_name}")

        # Final clean & store
        df_final = reorder_columns(df)
        for col in df_final.select_dtypes(include=['object']).columns:
            df_final[col] = df_final[col].apply(clean_for_excel)
        processed[sheet_name[:31]] = df_final

    return processed, logs, all_sheets


# ==========================================
# 3. GEMINI SCHEMA
# ==========================================
def preprocess_text_gemini(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\.00(?!\d)', '', text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '', text)
    text = text.replace("Rp.", "Rp ")
    return text

class MedicalItem(BaseModel):
    name:   str   = Field(..., description="Nama tindakan")
    amount: float = Field(..., description="Biaya tindakan ini")
    type:   str   = Field(..., description="'PRIMARY' untuk tindakan utama, 'SUPPORTING' untuk penunjang")

class CaseExtraction(BaseModel):
    row_id:        int
    final_status:  Literal['Approved', 'Butuh Konfirmasi', 'Ditolak', 'Others']
    billing_logic: str   = Field(..., description="Penjelasan naratif: Kondisi Awal -> Perubahan -> Keputusan Akhir.")
    final_bill:    float
    items:         List[MedicalItem]

class BatchResult(BaseModel):
    results: List[CaseExtraction]

def process_batch_gemini(model, batch_data, max_retries=3):
    prompt = f"""
    Anda adalah Senior Claim Analyst. Tugas: Ekstrak Final Bill & Logic.
    INPUT: Teks sudah dipreprocessing (titik ribuan dihapus).

    ATURAN LOGIC (STORYTELLING):
    Gunakan pola: "Kondisi Awal... Namun/Kemudian... Sehingga Keputusan..."

    ATURAN BILLING:
    - Status: 'Approved', 'Butuh Konfirmasi', 'Ditolak', 'Others'.
    - Ambil biaya Tindakan Utama (PRIMARY) sebagai Final Bill.

    DATA INPUT:
    {json.dumps(batch_data)}
    """
    gen_config = genai.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.0,
        response_schema=BatchResult
    )
    wait_time = 10
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt, generation_config=gen_config)
            parsed = json.loads(response.text)
            if isinstance(parsed, list):   return parsed
            elif isinstance(parsed, dict): return parsed.get('results', [])
            return []
        except exceptions.ResourceExhausted:
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            time.sleep(2)
    return []


# ==========================================
# 4. GEMINI PROCESSING — RETURNS DICT MAP
# ==========================================
def run_gemini_on_sheet(model, df_source, progress_bar=None, status_text=None):
    """
    Jalankan Gemini pada df_source.
    Return dict: { row_position -> {final_status, final_bill, billing_logic, breakdown_str} }
    """
    COL_REMARKS  = 'Remarks'
    COL_TOTALBILL= 'total bill'

    if COL_REMARKS not in df_source.columns:
        return {}

    df_source = df_source.copy()
    df_source['_pos'] = range(len(df_source))

    # Hanya proses baris total bill != 0
    if COL_TOTALBILL in df_source.columns:
        df_work = df_source[df_source[COL_TOTALBILL] != 0].copy()
    else:
        df_work = df_source.copy()

    if len(df_work) == 0:
        return {}

    BATCH_SIZE  = 5
    results_map = {}
    total_rows  = len(df_work)

    for i in range(0, total_rows, BATCH_SIZE):
        batch   = df_work.iloc[i: i + BATCH_SIZE]
        payload = []
        pos_map = {}

        for _, row in batch.iterrows():
            rid        = int(row['_pos'])
            raw_text   = str(row[COL_REMARKS]) if pd.notna(row[COL_REMARKS]) else ""
            clean_text = preprocess_text_gemini(raw_text)
            payload.append({"row_id": rid, "text": clean_text})
            pos_map[rid] = True

        ai_results = process_batch_gemini(model, payload)

        if ai_results:
            for res in ai_results:
                if not isinstance(res, dict): res = res.dict()
                rid       = res.get('row_id')
                items_obj = res.get('items', [])
                item_lines = []
                for item in items_obj:
                    if not isinstance(item, dict): item = item.dict()
                    item_lines.append(
                        f"- {item.get('name','-')} ({item.get('type','-')}): Rp {item.get('amount',0):,.0f}"
                    )
                results_map[rid] = {
                    'final_status':  res.get('final_status', ''),
                    'final_bill':    res.get('final_bill', 0),
                    'billing_logic': res.get('billing_logic', ''),
                    'breakdown_str': "\n".join(item_lines),
                }

        # Update progress
        if progress_bar is not None:
            progress_bar.progress(min((i + BATCH_SIZE) / total_rows, 1.0))
        if status_text is not None:
            done = min(i + BATCH_SIZE, total_rows)
            status_text.text(f"Memproses baris {done}/{total_rows}...")

        time.sleep(2)

    return results_map


# ==========================================
# 5. MERGE GEMINI → APPTO IP
# ==========================================
def merge_gemini_appto(df, gemini_map):
    df = df.copy()
    df['_pos'] = range(len(df))

    for col in ['bill', 'total bill', 'status']:
        if col not in df.columns:
            df[col] = ''

    def _apply(row):
        rid = int(row['_pos'])
        if rid not in gemini_map:
            row['Breakdown APPTO'] = row.get('bill', '')
            row['Final Bill']      = row.get('total bill', '')
            row['Status']          = row.get('status', '')
            row['Reason By AI']    = ''
        else:
            g = gemini_map[rid]
            row['Breakdown APPTO'] = g['breakdown_str']
            row['Final Bill']      = g['final_bill']
            row['total bill']      = g['final_bill']   # overwrite angka
            row['Status']          = g['final_status']
            row['Reason By AI']    = g['billing_logic']
        return row

    df = df.apply(_apply, axis=1)
    df.drop(columns=['bill', 'status', '_pos'], inplace=True, errors='ignore')

    priority_end = ['total bill', 'Final Bill', 'bill range', 'Breakdown APPTO', 'Status', 'Reason By AI']
    other_cols   = [c for c in df.columns if c not in priority_end]
    return df[other_cols + [c for c in priority_end if c in df.columns]]


# ==========================================
# 6. MERGE GEMINI → PRE ADM
# ==========================================
def merge_gemini_preadm(df, gemini_map):
    df = df.copy()
    df['_pos'] = range(len(df))

    for col in ['bill', 'total bill', 'status']:
        if col not in df.columns:
            df[col] = ''

    def _apply(row):
        rid = int(row['_pos'])
        if rid not in gemini_map:
            row['Diagnosis_'] = ''
            row['Status']     = row.get('status', '')
        else:
            g = gemini_map[rid]
            row['total bill'] = g['final_bill']     # overwrite angka, nama tetap
            row['Diagnosis_'] = g['breakdown_str']
            row['Status']     = g['final_status']
        return row

    df = df.apply(_apply, axis=1)
    df.drop(columns=['status', '_pos'], inplace=True, errors='ignore')

    priority_end = ['bill', 'Diagnosis_', 'total bill', 'Status']
    other_cols   = [c for c in df.columns if c not in priority_end]
    return df[other_cols + [c for c in priority_end if c in df.columns]]


# ==========================================
# 7. WRITE ALL SHEETS → BYTES BUFFER
# ==========================================
def write_to_buffer(processed_sheets):
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        for sheet_name, df in processed_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output_buffer.seek(0)
    return output_buffer


# ==========================================
# 8. STREAMLIT UI
# ==========================================
st.title("🏥 QA Health Claims Processor — Astra v4")

st.markdown("""
**Cara pakai:**
1. Upload file Excel raw di bawah.
2. Atur konfigurasi nama sheet di sidebar.
3. Klik **Start Processing** — Phase 1 (rule-based) akan berjalan otomatis.
4. Jika API Key Gemini diisi, Phase 2 (AI) akan berjalan untuk sheet APPTO IP & Pre Adm.
5. Download hasil.
""")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    pic_name = st.text_input("Nama PIC", placeholder="contoh: Dani")
    api_key  = st.text_input("Gemini API Key (opsional)", type="password",
                              help="Isi untuk mengaktifkan AI extraction. Dapatkan di https://aistudio.google.com/api-keys")

    st.divider()
    st.subheader("Nama Sheet")
    st.caption("Pisahkan dengan koma. Harus sama persis (case-sensitive).")

    preadm_in  = st.text_input("PreAdm Sheets",    "PreAdm, Pre Adm, Preadmission")
    appto_in   = st.text_input("APPTO Sheets",     "APPTO, APPTO IP, Appto")
    ben_ip_in  = st.text_input("Benefit IP Sheets","Benefit IP")
    ben_op_in  = st.text_input("Benefit OP Sheets","Benefit OP, Benefit OP Dll, Benefit OP dll")

    st.divider()
    st.subheader("Sheet Target AI")
    st.caption("Sheet mana yang diproses Gemini (harus ada di APPTO / PreAdm list).")
    gemini_appto_sheet  = st.text_input("AI → APPTO sheet", "APPTO IP")
    gemini_preadm_sheet = st.text_input("AI → PreAdm sheet", "Pre Adm")

# ---- MAIN AREA ----
uploaded_file = st.file_uploader("📂 Upload File Excel Raw", type=["xlsx"])

if uploaded_file:
    col1, col2 = st.columns([1, 3])
    with col1:
        start_btn = st.button("🚀 Start Processing", use_container_width=True, type="primary")

    if start_btn:
        # Parse list input
        preadm_list  = [x.strip() for x in preadm_in.split(',')]
        appto_list   = [x.strip() for x in appto_in.split(',')]
        ben_ip_list  = [x.strip() for x in ben_ip_in.split(',')]
        ben_op_list  = [x.strip() for x in ben_op_in.split(',')]

        # ============================================================
        # PHASE 1 — RULE-BASED
        # ============================================================
        st.markdown("---")
        st.subheader("Phase 1 — Cleaning, Regex & Deduplication")

        with st.spinner("Memproses semua sheet..."):
            processed_sheets, logs1, all_sheets_raw = run_phase_1(
                uploaded_file, preadm_list, appto_list, ben_ip_list, ben_op_list
            )

        st.success(f"✅ Phase 1 selesai — {len(processed_sheets)} sheet diproses.")
        with st.expander("📋 Log Phase 1", expanded=False):
            for log in logs1:
                st.text(log)

        # ============================================================
        # PHASE 2 — GEMINI AI (opsional)
        # ============================================================
        if api_key:
            st.markdown("---")
            st.subheader("Phase 2 — AI Extraction (Gemini)")

            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(
                    'gemini-2.5-flash',
                    generation_config={"response_mime_type": "application/json"}
                )
                st.success("🔗 Gemini API terhubung.")
            except Exception as e:
                st.error(f"Gagal koneksi Gemini: {e}")
                gemini_model = None

            logs2 = []

            if gemini_model:
                appto_key  = gemini_appto_sheet[:31]
                preadm_key = gemini_preadm_sheet[:31]

                # ---- APPTO IP ----
                if appto_key in processed_sheets:
                    st.markdown(f"**🤖 AI → {gemini_appto_sheet}**")
                    appto_rows = len(processed_sheets[appto_key])
                    p_bar  = st.progress(0)
                    s_text = st.empty()
                    s_text.text(f"Memproses 0/{appto_rows}...")

                    gemini_map = run_gemini_on_sheet(
                        gemini_model,
                        processed_sheets[appto_key],
                        progress_bar=p_bar,
                        status_text=s_text
                    )
                    if gemini_map:
                        processed_sheets[appto_key] = merge_gemini_appto(
                            processed_sheets[appto_key], gemini_map
                        )
                        s_text.text(f"✅ {len(gemini_map)} baris di-merge ke sheet '{appto_key}'.")
                        logs2.append(f"APPTO AI: {len(gemini_map)} baris berhasil diproses.")
                    else:
                        s_text.text("⚠️ Tidak ada hasil AI untuk sheet ini.")
                        logs2.append(f"APPTO AI: Tidak ada hasil.")
                else:
                    st.info(f"Sheet '{gemini_appto_sheet}' tidak ditemukan, skip AI APPTO.")
                    logs2.append(f"Sheet '{gemini_appto_sheet}' tidak ada, skip.")

                # ---- PRE ADM ----
                if preadm_key in processed_sheets:
                    st.markdown(f"**🤖 AI → {gemini_preadm_sheet}**")
                    preadm_rows = len(processed_sheets[preadm_key])
                    p_bar2  = st.progress(0)
                    s_text2 = st.empty()
                    s_text2.text(f"Memproses 0/{preadm_rows}...")

                    gemini_map2 = run_gemini_on_sheet(
                        gemini_model,
                        processed_sheets[preadm_key],
                        progress_bar=p_bar2,
                        status_text=s_text2
                    )
                    if gemini_map2:
                        processed_sheets[preadm_key] = merge_gemini_preadm(
                            processed_sheets[preadm_key], gemini_map2
                        )
                        s_text2.text(f"✅ {len(gemini_map2)} baris di-merge ke sheet '{preadm_key}'.")
                        logs2.append(f"PreAdm AI: {len(gemini_map2)} baris berhasil diproses.")
                    else:
                        s_text2.text("⚠️ Tidak ada hasil AI untuk sheet ini.")
                        logs2.append(f"PreAdm AI: Tidak ada hasil.")
                else:
                    st.info(f"Sheet '{gemini_preadm_sheet}' tidak ditemukan, skip AI PreAdm.")
                    logs2.append(f"Sheet '{gemini_preadm_sheet}' tidak ada, skip.")

                with st.expander("📋 Log Phase 2 (AI)", expanded=False):
                    for log in logs2:
                        st.text(log)

        else:
            st.info("ℹ️ Gemini API Key tidak diisi — Phase 2 dilewati. Hasil hanya dari rule-based.")

        # ============================================================
        # PREVIEW & DOWNLOAD
        # ============================================================
        st.markdown("---")
        st.subheader("📊 Preview Hasil")

        sheet_names = list(processed_sheets.keys())
        if sheet_names:
            selected_sheet = st.selectbox("Pilih sheet untuk preview:", sheet_names)
            st.dataframe(
                processed_sheets[selected_sheet].head(50),
                use_container_width=True,
                height=350
            )
            st.caption(f"Menampilkan 50 baris pertama dari {len(processed_sheets[selected_sheet])} total baris.")

        # Write to buffer
        output_buffer = write_to_buffer(processed_sheets)

        # Filename
        from datetime import datetime
        today_str = datetime.now().strftime("%d%m%y")
        out_name  = f"Processed_QA_{today_str}_{pic_name}.xlsx" if pic_name else f"Processed_QA_{today_str}.xlsx"

        st.download_button(
            label="📥 Download Hasil (.xlsx)",
            data=output_buffer.getvalue(),
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        st.success(f"File siap didownload: **{out_name}**")