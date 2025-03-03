import streamlit as st
import streamlit_authenticator as stauth

# 用户配置（建议存储到数据库或加密文件）
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin",
            "password": stauth.Hasher(["admin123"]).generate()[0]  # 自动哈希密码
        },
        "user": {
            "name": "Demo",
            "password": stauth.Hasher(["demo123"]).generate()[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "app_cookie",  # Cookie 名称
    "random_signature_key",  # 签名密钥
    30  # Cookie 有效期（天）
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f'Welcome *{name}*')
    st.title('Application Content')
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
