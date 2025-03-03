import streamlit as st

from authlib.integrations.requests_client import OAuth2Session

# 配置 OAuth2 参数
CLIENT_ID = "pistonint_cloud"
CLIENT_SECRET = "pi.s#t!on*#cl@oud!@#2021.v5"
AUTHORIZE_URL = "https://sso.pistonint.com/auth/oauth2/authorize"
TOKEN_URL = "https://sso.pistonint.com/auth/oauth/token"


def oauth2_flow():
    client = OAuth2Session(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scope="read"
    )
    uri, state = client.create_authorization_url(AUTHORIZE_URL)
    print(f"Please login to continue.uri:{uri}\nstate:{state}")
    st.session_state.oauth_state = state
    st.markdown(f"[Login here]({uri})")


def handle_callback():
    client = OAuth2Session(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        state=st.session_state.oauth_state
    )
    token = client.fetch_token(
        TOKEN_URL,
        authorization_response=st.query_params["code"][0]
    )
    st.session_state.token = token


if 'token' not in st.session_state:
    if 'code' in st.query_params:
        handle_callback()
    else:
        oauth2_flow()
        st.stop()

# 显示已登录内容
st.title(f"Welcome {st.session_state.token['userinfo']['name']}")
