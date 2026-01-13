import streamlit as st
import pandas as pd
import re
import io
import time
import json
from difflib import SequenceMatcher
import google.generativeai as genai
from google.api_core import exceptions
from typing import List, Literal
import typing_extensions

# ==========================================
# 0. CONFIG & HELPER FUNCTIONS
# ==========================================
st.set_page_config(page_title="Claims Data Processor", layout="wide")

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
        match_found = False
        for master_text in accepted_remarks:
            if is_similar(text, master_text, threshold=0.8):
                match_found = True
                break
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
            
        pattern = r'(?:rp|est|biaya|cost|harga|total|nominal)\D{0,30}?(\d[\d.,]*\d)' if is_appto else r'(?:rp|est|biaya|cost|harga|total|nominal)\D{0,30}?(\d[\d.,;]*\d)'
        
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
            text_after = full_text[last_marker_index:]
            bills_right = get_bills_from_text(text_after)
            if bills_right:
                final_bill_set = bills_right
            else:
                text_before = full_text[:last_marker_index]
                final_bill_set = get_bills_from_text(text_before)
        else:
            final_bill_set = set()
    else:
        final_bill_set = get_bills_from_text(full_text)

    total_bill = sum(final_bill_set)
    bill_str = ", ".join([str(b) for b in sorted(list(final_bill_set))]) if final_bill_set else ""

    # Status Logic
    clean_text = full_text.replace('\n', '.').replace(';;', '.').replace(';', '.')
    parts = clean_text.split('.')
    valid_parts = [p.strip() for p in parts if p.strip()]
    last_sentence = valid_parts[-1] if valid_parts else full_text
    
    status = 'Other'
    text_for_status = last_sentence 
    kw_cancel = ['batal', 'cancel', 'tidak jadi', 'dibatalkan']
    kw_reject = ['tolak', 'reject', 'tidak dijamin', 'decline', 'tidak cover']
    kw_approve = ['acc', 'dijaminkan', 'approved', 'setujui', 'dijamin', 'cover', 'ok saya setuju', 'ok dijaminkan']
    kw_confirm = ['konf', 'confirm', 'konfirmasi', 'butuh konfirmasi', 'tunggu', 'wait', 'pending', 'menunggu', 'hold', 'review', 'f/u', 'follow up', 'lapor', 'koordinasi', 'mohon', 'cek', 'arahkan', 'info am', 'info hrd', 'pertanyaan']
    
    if any(x in text_for_status for x in kw_cancel): status = 'Cancelled'
    elif any(x in text_for_status for x in kw_reject): status = 'Rejected'
    elif any(x in text_for_status for x in kw_approve): status = 'Approved'
    elif any(x in text_for_status for x in kw_confirm): status = 'Butuh Konfirmasi'
    if status == 'Cancelled' and any(x in text_for_status for x in kw_confirm): status = 'Butuh Konfirmasi'
    
    result = {'bill': bill_str, 'total bill': total_bill, 'status': status}
    if is_appto:
        if total_bill > 50_000_000: result['bill range'] = '> 50 Juta'
        elif total_bill >= 20_000_000: result['bill range'] = '20 - 50 Juta'
        else: result['bill range'] = 'Others'
    return pd.Series(result)

def reorder_columns(df):
    target = ["Month", "Modified By", "Callin/Callout", "Call Category", "Callin Start", "Sub Service Type", "Call ID", " ", " ", "Client Name", "Member Name", "Member No", "Provider Name", "Contact Name", "Remarks", "Member’s Phone No", " ", "GL Type", "Product Type", "Diagnosis Awal", "Diagnosis Akhir", "Call Status", "Satisfaction Level", "Created By", " ", " ", " ", "Callin Finish", "Callout Start", "Callout Finish", "bill", "total bill", "bill range", "status"]
    existing_cols = [c for c in target if c in df.columns]
    remain_cols = [c for c in df.columns if c not in target]
    return df[existing_cols + remain_cols]

# ==========================================
# 1. PHASE 1: LOGIC & REGEX PROCESSING
# ==========================================
def run_phase_1(input_file, preadm_list, appto_list, benefit_ip_list, benefit_op_list):
    all_sheets = pd.read_excel(input_file, sheet_name=None)
    output_buffer = io.BytesIO()
    
    logs = []
    
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        for sheet_name, df in all_sheets.items():
            if df.empty: continue
            
            # Pre-cleaning
            if 'Callin Finish' in df.columns:
                df['Callin Finish'] = pd.to_datetime(df['Callin Finish'], dayfirst=True, errors='coerce')
            if 'Member No' not in df.columns:
                df['Member No'] = 'Unknown'

            # Logic Selection
            if sheet_name in preadm_list:
                logs.append(f"Processing PreAdm: {sheet_name}")
                if 'Remarks' in df.columns:
                    new_cols = df.apply(lambda row: extract_remarks_info(row, is_appto=False), axis=1)
                    df = df.join(new_cols)
                    df = df.sort_values(by=['total bill', 'Callin Finish'], ascending=[False, True])
                    df['is_dup'] = df.groupby('Member No', group_keys=False).apply(mark_fuzzy_duplicates)
                    df = df[df['is_dup'] == False].drop(columns=['is_dup'], errors='ignore')
                    df = df.drop_duplicates(subset=['Member No', 'total bill'], keep='first')

            elif sheet_name in appto_list:
                logs.append(f"Processing APPTO: {sheet_name}")
                initial_count = len(df)
                if 'Remarks' in df.columns:
                    new_cols = df.apply(lambda row: extract_remarks_info(row, is_appto=True), axis=1)
                    df = df.join(new_cols)
                    df = df[df['bill range'] != 'Others']
                    
                    if not df.empty:
                        df = df.sort_values(by=['total bill'], ascending=False)
                        df['is_dup'] = df.groupby('Member No', group_keys=False).apply(mark_fuzzy_duplicates)
                        df = df[df['is_dup'] == False].drop(columns=['is_dup'], errors='ignore')
                        df = df.drop_duplicates(subset=['Member No', 'total bill'], keep='first')
                        logs.append(f"  -> APPTO Reduced from {initial_count} to {len(df)}")
            
            elif sheet_name in benefit_ip_list:
                logs.append(f"Processing Benefit IP: {sheet_name}")
                # Balanced & Unique logic
                if 'Sub Service Type' not in df.columns: df['Sub Service Type'] = 'Unknown'
                df['Sub Service Type'] = df['Sub Service Type'].fillna('Unknown')
                unique_types = df['Sub Service Type'].unique()
                groups = [df[df['Sub Service Type'] == t].sample(frac=1, random_state=42) for t in unique_types]
                balanced_rows = []
                if groups:
                    max_len = max(len(g) for g in groups)
                    for i in range(max_len):
                        for g in groups:
                            if i < len(g): balanced_rows.append(g.iloc[[i]])
                if balanced_rows: df = pd.concat(balanced_rows, ignore_index=True)
                
                # Greedy Selection
                selected_indices = []
                seen = set()
                for idx, row in df.iterrows():
                    key = (str(row.get('Modified By')), str(row.get('Member No')))
                    if key not in seen:
                        selected_indices.append(idx)
                        seen.add(key)
                    if len(selected_indices) >= 20: break
                df = df.loc[selected_indices]

            elif sheet_name in benefit_op_list:
                logs.append(f"Processing Benefit OP: {sheet_name}")
                # Similar logic to Benefit IP but target 10
                if 'Sub Service Type' not in df.columns: df['Sub Service Type'] = 'Unknown'
                df['Sub Service Type'] = df['Sub Service Type'].fillna('Unknown')
                unique_types = df['Sub Service Type'].unique()
                groups = [df[df['Sub Service Type'] == t].sample(frac=1, random_state=42) for t in unique_types]
                balanced_rows = []
                if groups:
                    max_len = max(len(g) for g in groups)
                    for i in range(max_len):
                        for g in groups:
                            if i < len(g): balanced_rows.append(g.iloc[[i]])
                if balanced_rows: df = pd.concat(balanced_rows, ignore_index=True)
                
                selected_indices = []
                seen = set()
                for idx, row in df.iterrows():
                    key = (str(row.get('Modified By')), str(row.get('Member No')))
                    if key not in seen:
                        selected_indices.append(idx)
                        seen.add(key)
                    if len(selected_indices) >= 10: break
                df = df.loc[selected_indices]

            # Save
            df_final = reorder_columns(df)
            for col in df_final.select_dtypes(include=['object']).columns:
                df_final[col] = df_final[col].apply(clean_for_excel)
            final_name = sheet_name[:31]
            df_final.to_excel(writer, sheet_name=final_name, index=False)
            
    return output_buffer, logs

# ==========================================
# 2. PHASE 2: GEMINI AI PROCESSING
# ==========================================
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\.00(?!\d)', '', text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '', text)
    text = text.replace("Rp.", "Rp ")
    return text

def run_phase_2(excel_buffer, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-2.0-flash-exp',
        generation_config={"response_mime_type": "application/json"}
    )
    
    # Read the processed file from memory
    excel_buffer.seek(0)
    xls = pd.ExcelFile(excel_buffer)
    
    output_buffer = io.BytesIO()
    logs = []
    
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Only process APPTO IP sheets with AI
            if "APPTO" in sheet_name.upper():
                logs.append(f"AI Processing for: {sheet_name}")
                results = []
                
                # Batch processing
                BATCH_SIZE = 5
                progress_bar = st.progress(0)
                
                for i in range(0, len(df), BATCH_SIZE):
                    batch = df.iloc[i : i + BATCH_SIZE]
                    payload = []
                    
                    for idx, row in batch.iterrows():
                        text = preprocess_text(str(row.get('Remarks', '')))
                        payload.append({"row_id": idx, "text": text})
                    
                    prompt = f"""
                    You are a Senior Claim Analyst. Extract Final Bill & Logic.
                    INPUT: {json.dumps(payload)}
                    OUTPUT JSON Schema: {{ "results": [ {{ "row_id": int, "final_status": "Approved"|"Butuh Konfirmasi"|"Ditolak"|"Others", "billing_logic": string, "final_bill": float, "items": [{{ "name": string, "amount": float, "type": "PRIMARY"|"SUPPORTING" }}] }} ] }}
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        parsed = json.loads(response.text)
                        
                        # Handle list vs dict response
                        batch_res = parsed if isinstance(parsed, list) else parsed.get('results', [])
                        
                        for res in batch_res:
                            rid = res.get('row_id')
                            # Find original row
                            if rid is not None:
                                items_str = "\n".join([f"- {x['name']}: {x['amount']}" for x in res.get('items', [])])
                                results.append({
                                    'row_id_temp': rid,
                                    'AI_Status': res.get('final_status'),
                                    'AI_Bill': res.get('final_bill'),
                                    'AI_Logic': res.get('billing_logic'),
                                    'AI_Items': items_str
                                })
                    except Exception as e:
                        logs.append(f"Error batch {i}: {e}")
                    
                    progress_bar.progress(min((i + BATCH_SIZE) / len(df), 1.0))
                    time.sleep(1) # Rate limit handling
                
                # Merge AI results back to DF
                if results:
                    df_ai = pd.DataFrame(results)
                    # Join based on index/row_id
                    df['row_id_temp'] = df.index
                    df = pd.merge(df, df_ai, on='row_id_temp', how='left')
                    df.drop(columns=['row_id_temp'], inplace=True)

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_buffer, logs

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.title("🏥 Daily QA for Health Claim Data with AI")

st.markdown("""
**Steps:**
1. Upload your Raw Excel file.
2. Define Sheet Names for each logic category.
3. (Optional) Enter Gemini API Key for AI-powered bill extraction.
4. Click Process.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key (Optional)", type="password")
    
    st.subheader("Sheet Name Mapping")
    st.caption("Use exact names (case-sensitive). Separate with comma.")
    
    preadm_in = st.text_input("PreAdm Sheets", "PreAdm, Pre Adm, Preadmission")
    appto_in = st.text_input("APPTO Sheets", "APPTO, APPTO IP, Appto")
    ben_ip_in = st.text_input("Benefit IP Sheets", "Benefit IP")
    ben_op_in = st.text_input("Benefit OP Sheets", "Benefit OP, Benefit OP Dll, Benefit OP dll")

# --- Main Area ---
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file and st.button("Start Processing"):
    # Parse inputs
    preadm_list = [x.strip() for x in preadm_in.split(',')]
    appto_list = [x.strip() for x in appto_in.split(',')]
    ben_ip_list = [x.strip() for x in ben_ip_in.split(',')]
    ben_op_list = [x.strip() for x in ben_op_in.split(',')]

    # Phase 1
    with st.spinner("Running Phase 1: Cleaning, Regex & Deduplication..."):
        phase1_buffer, logs1 = run_phase_1(uploaded_file, preadm_list, appto_list, ben_ip_list, ben_op_list)
    
    st.success("Phase 1 Complete!")
    with st.expander("View Phase 1 Logs"):
        for log in logs1: st.write(log)

    final_buffer = phase1_buffer
    
    # Phase 2 (Optional)
    if api_key:
        with st.spinner("Running Phase 2: AI Smart Extraction (Gemini)..."):
            final_buffer, logs2 = run_phase_2(phase1_buffer, api_key)
        st.success("Phase 2 (AI) Complete!")
        with st.expander("View AI Logs"):
            for log in logs2: st.write(log)
    else:
        st.info("Skipping Phase 2 (No API Key provided).")

    # Download
    st.write("---")
    st.download_button(
        label="📥 Download Processed File",
        data=final_buffer.getvalue(),
        file_name="Processed_Claims_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )


