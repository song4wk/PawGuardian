"""PawGuardian: 車内ペット安全監護 Agent アプリ。"""

import datetime
import json
import os
import re
from typing import Optional
from dotenv import load_dotenv

import google.auth
import streamlit as st
from google.auth.transport.requests import Request
from google.cloud import secretmanager, storage
from twilio.rest import Client
import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

# ==========================================
# 1. 構成と環境設定 (Tokyo Region)
# ==========================================

# ローカル実行時: .env から環境変数を読み込む（Cloud では .env が無いためスキップ）
load_dotenv()

# 現在の環境の認証情報を自動取得
credentials, project_id = google.auth.default()

# クラウド: サービスアカウントのメールを取得 / ローカル: 環境変数から取得
if hasattr(credentials, "service_account_email"):
    sa_email = credentials.service_account_email
else:
    sa_email = os.environ.get("SERVICE_ACCOUNT_EMAIL")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "sentientcarguard")
LOCATION = "asia-northeast1"  # 東京リージョン


@st.cache_resource
def load_secrets() -> dict:
    secrets = {}
    required_secrets = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER",
        "TWILIO_SMS_NUMBER",
        "OWNER_PHONE_NUMBER",
    ]

    try:
        client = secretmanager.SecretManagerServiceClient()
        for secret_id in required_secrets:
            name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
            try:
                response = client.access_secret_version(request={"name": name})
                secrets[secret_id] = response.payload.data.decode("UTF-8")
            except Exception as e:
                print(f"Warning: Could not load {secret_id}: {e}")
                secrets[secret_id] = None
    except Exception as e:
        st.error(f"Secret Manager Connection Error: {e}")

    return secrets


SECRETS = load_secrets()

TWILIO_SID = SECRETS.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = SECRETS.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_PHONE = SECRETS.get("TWILIO_PHONE_NUMBER")
TWILIO_SMS_NUMBER = SECRETS.get("TWILIO_SMS_NUMBER")
TWILIO_TO = SECRETS.get("OWNER_PHONE_NUMBER")


MODEL_ID = "gemini-2.5-flash"

try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    st.error(f"Vertex AI Init Failed: {e}")
    st.stop()

VIDEOS = {
    "シナリオ A: 待機中 (Relax)": {
        "uri": "gs://paw-guardian-tokyo/Relax.mp4",
        "desc": "安全：リラックス状態",
    },
    "シナリオ B: 軽度の不安 (Low Anxiety)": {
        "uri": "gs://paw-guardian-tokyo/Low Anxiety.mp4",
        "desc": "注意：初期の不安兆候",
    },
    "シナリオ C: 重度の不安 (High Anxiety)": {
        "uri": "gs://paw-guardian-tokyo/High Anxiety.mp4",
        "desc": "警告：重度の分離不安",
    },
    "シナリオ D: 空席 (Nothing)": {
        "uri": "gs://paw-guardian-tokyo/Nothing.mp4",
        "desc": "待機：車内無人",
    },
}

# ==========================================
# 2. ツール定義（Agent が呼び出す Python 関数）
# ==========================================


def send_sms_alert(message: str) -> str:
    if not all([TWILIO_SID, TWILIO_TOKEN]):
        return "エラー: Twilio が設定されていません。"
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=message, from_=TWILIO_SMS_NUMBER, to=TWILIO_TO)
        return "SMS を送信しました。"
    except Exception as e:
        return f"エラー: SMS 送信に失敗しました - {e}"


def make_emergency_call(message: str) -> str:
    if not all([TWILIO_SID, TWILIO_TOKEN]):
        return "エラー: Twilio が設定されていません。"
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        twiml = (
            f'<Response><Say language="ja-JP" voice="alice">{message}</Say></Response>'
        )
        client.calls.create(twiml=twiml, to=TWILIO_TO, from_=TWILIO_FROM_PHONE)
        return "電話を発信しました。"
    except Exception as e:
        return f"エラー: 電話発信に失敗しました - {e}"


def open_car_windows(level: int) -> str:
    return f"窓を {level}% 開きます。"


def play_music(track_type: str) -> str:
    return f"音楽を再生します。{track_type}"


guardian_tools = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="send_sms_alert",
            description="Send an SMS alert to the owner.",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
        FunctionDeclaration(
            name="make_emergency_call",
            description="Make an emergency phone call to the owner.",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        ),
        FunctionDeclaration(
            name="open_car_windows",
            description="Open the car windows. Level: 0-100 (0 = closed, 100 = fully open).",
            parameters={
                "type": "object",
                "properties": {"level": {"type": "integer"}},
                "required": ["level"],
            },
        ),
        FunctionDeclaration(
            name="play_music",
            description="Play soothing music. track_type: 'relax' or 'white_noise'.",
            parameters={
                "type": "object",
                "properties": {"track_type": {"type": "string"}},
                "required": ["track_type"],
            },
        ),
    ]
)


@st.cache_data(ttl=3600)
def get_signed_url_cached(gcs_uri: str) -> Optional[str]:
    try:
        path = gcs_uri.replace("gs://", "")
        bucket_name, blob_name = path.split("/", 1)

        credentials, project = google.auth.default()

        if not credentials.valid:
            credentials.refresh(Request())

        storage_client = storage.Client(credentials=credentials, project=project)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
            service_account_email=sa_email,
            access_token=credentials.token,
        )
        return url
    except Exception as e:
        st.error(f"Video Signing Error: {e}")
        return None


def clean_json_text(text: Optional[str]) -> str:
    if not text:
        return "{}"
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


# ==========================================
# 3. Streamlit インターフェース
# ==========================================

st.set_page_config(page_title="PawGuardian", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models() -> tuple:
    return (
        GenerativeModel(MODEL_ID),
        GenerativeModel(MODEL_ID, tools=[guardian_tools]),
    )


observer_model, agent_model = load_models()

with st.sidebar:
    st.header("🐶 ペット情報")
    with st.form("pet_config_form"):
        pet_name = st.text_input("名前", "Lucky")
        pet_breed = st.selectbox(
            "犬種",
            ["コーギー", "柴犬", "チワワ", "シュナウザー", "ポメラニアン", "カスタム"],
        )
        if pet_breed == "カスタム":
            pet_breed = (
                st.text_input("犬種名", placeholder="例: ミックス") or "カスタム"
            )
        pet_age = st.number_input("年齢（歳）", 0.5, 20.0, 4.5, 0.5)
        pet_weight = st.number_input("体重（kg）", 5.0, 30.0, 13.5, 0.5)
        is_brachy = "Yes" if pet_breed in ["フレンチブルドッグ", "パグ"] else "No"
        medical_history = st.text_area(
            "既往症", placeholder="気管虚脱、皮膚病、耳炎など"
        )
        sensitivity = st.slider("不安感度（高いほど不安を検知しやすい）", 1, 10, 8)
        st.form_submit_button("設定を保存")

    breed_note = ""
    if "フレンチブルドッグ" in pet_breed:
        breed_note = "⚠️ クリティカル: これはブラシェペシック（短頭種）の犬種です。極端に低い熱耐性を持っています。温度閾値を5°C下げてください。"
    elif "シニア犬" in pet_breed or pet_age > 10:
        breed_note = (
            "⚠️ クリティカル: シニア犬。不安反応が低いです。反応が速くなります。"
        )

    pet_context = f"Name: {pet_name}, Breed: {pet_breed}, Age: {pet_age}, Owner Sensitivity: {sensitivity}/10. {breed_note}"

    if breed_note:
        st.warning(f"特別コンテキスト有効化:\n{breed_note}")

st.title("🛡️ PawGuardian: Pet Safety Agent")
st.caption(f"Architecture: Native Function Calling (Tools) • Model: {MODEL_ID}")
st.markdown("---")

col_video, col_agent = st.columns([4, 5])

with col_video:
    st.subheader("📹 監視フィード")
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = list(VIDEOS.keys())[0]
    sel = st.selectbox(
        "シナリオ選択",
        list(VIDEOS.keys()),
        index=list(VIDEOS.keys()).index(st.session_state.selected_scenario),
    )
    st.session_state.selected_scenario = sel

    video_uri = VIDEOS[sel]["uri"]
    st.info(VIDEOS[sel]["desc"])
    car_temp = st.slider("模擬車内温度 (°C)", 15, 45, 26)
    video_placeholder = st.empty()
    url = get_signed_url_cached(video_uri)
    if url:
        video_placeholder.video(url)
    else:
        st.error("ビデオの読み込みに失敗しました")

with col_agent:
    st.subheader("🧠 エージェント思考プロセス")

    if st.button("🚀 システム起動", type="primary", use_container_width=True):
        video_part = Part.from_uri(video_uri, mime_type="video/mp4")

        # ---------------------------------------------------------
        # Agent 1: Observer (視覚分析)
        # ---------------------------------------------------------
        with st.status("🕵️‍♂️ Agent 1: 視覚分析中...", expanded=True) as s1:
            obs_prompt = f"""
            Analyze the video based on these criteria:
            - Relax: Body relaxed, sitting quietly, no exploration.
            - Low Anxiety: Licking nose, ears back, looking around restlessly.
            - High Anxiety: Scratching windows, heavy panting, continuous barking.

            [CRITICAL RULE]
            If NO dog is visible in the car, set "subject_detected" to false AND "anxiety_level" to "None".

            Context: {pet_context}.
            Output JSON: {{
                "subject_detected": bool,
                "anxiety_level": "Relax|Low|High",
                "observations": "string (Japanese)",
                "stress_signs": ["string"]
            }}
            """
            try:
                r1 = observer_model.generate_content(
                    [video_part, obs_prompt],
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.0,
                    },
                )
                obs_data = json.loads(clean_json_text(r1.text))
                st.json(obs_data)

                if not obs_data.get("subject_detected"):
                    st.warning("📭 車内にペットが検知されませんでした。")
                    st.success("✅ 監視完了: 介入の必要はありません。")
                    st.stop()

                s1.update(label="✅ 視覚分析完了", state="complete", expanded=False)

            except Exception as e:
                st.error(f"Observer Error: {e}")
                st.stop()

        with st.status("🤖 Agent 2: 自律判断 & 実行中...", expanded=True) as s2:

            system_prompt = f"""
            You are 'PawGuardian' Autonomous AI. Your goal is safety intervention.
            
            [CORE OPERATING PRINCIPLE]
            - Safety interventions (Calling, SMS, Windows) are ONLY permitted if a dog is detected inside the car ("subject_detected": true).
            - If NO dog is detected, your only task is to report "Vehicle is empty and safe." regardless of the temperature.

            [Safety Protocols]
            1. IF subject_detected is true AND temperature > 35: CALL owner AND OPEN windows.
            2. IF subject_detected is true AND pet_breed is 'Brachycephalic' and temp > 30: CALL owner.
            3. IF subject_detected is true AND anxiety == 'High': CALL owner.
            4. IF subject_detected is true AND anxiety == 'Low': PLAY music and SMS owner.
            5. IF subject_detected is true AND anxiety == 'Relax' AND temperature is safe: DO NOT call any tools. Just report "Safe".

            [Constraint]
            - Do NOT perform any actions if the status is 'Relax' and temperature is within normal range.
            - Be concise. Only act when necessary.

            [Language Rule]
            - ALL your responses (Thought and Final Report) MUST be in JAPANESE.
            - Even if the tool execution results are in English, you must summarize them in JAPANESE.
            """
            decision_model = GenerativeModel(
                model_name=MODEL_ID,
                tools=[guardian_tools],
                system_instruction=system_prompt,
            )
            chat = decision_model.start_chat()

            user_msg = f"""
            Current status for evaluation:
            - Visual Data: {json.dumps(obs_data)}
            - Car Temp: {car_temp}°C
            - Pet Context: {pet_context}

            Task:
            Determine if any intervention is required. 

            Rules:
            - If 'anxiety_level' is 'Relax' and temp < 30°C: STOP and report "Pet is safe. No action needed."
            - Do not call tools for 'Relax' state.

            Instruction:
            1. Evaluate 'anxiety_level' and 'Car Temperature' against the rules.
            2. If no rules are triggered, do not use any tools.
            3. Summarize in Japanese.
            """

            try:
                response = chat.send_message(
                    user_msg, generation_config={"temperature": 0.0}
                )
                try:
                    if response.text:
                        st.markdown(f"**🤔 思考プロセス:**\n{response.text}")
                except ValueError:
                    pass
                function_calls = response.candidates[0].function_calls

                if not function_calls:
                    st.success("✅ 介入の必要はありません。ペットは安全な状態です。")
                else:
                    function_responses = []
                    for call in function_calls:
                        f_name = call.name
                        f_args = call.args
                        st.write(f"⚙️ **Action:** `{f_name}`")

                        result = "Error"
                        # 実行ロジック
                        if f_name == "send_sms_alert":
                            result = send_sms_alert(f_args["message"])
                            st.toast("SMSが送信されました", icon="📨")
                        elif f_name == "make_emergency_call":
                            result = make_emergency_call(f_args["message"])
                            st.toast("緊急電話発信中", icon="📞")
                        elif f_name == "open_car_windows":
                            level = int(f_args["level"])
                            result = open_car_windows(level)
                            st.progress(level / 100.0, text="窓を開放中...")
                        elif f_name == "play_music":
                            result = play_music(f_args["track_type"])
                            st.audio(
                                "https://cdn.pixabay.com/audio/2024/11/25/audio_d3038b75e1.mp3"
                            )

                        st.caption(f"👁️ Observation: {result}")
                        function_responses.append(
                            Part.from_function_response(
                                name=f_name, response={"result": result}
                            )
                        )

                    final_res = chat.send_message(function_responses)
                    st.markdown("### 📋 最終レポート")
                    st.info(final_res.text)

                s2.update(
                    label="✅ モニタリングが完了しました",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                st.error(f"Agent Error: {e}")
