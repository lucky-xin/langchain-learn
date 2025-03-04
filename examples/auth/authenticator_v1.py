import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader

with open('/Users/luchaoxin/dev/workspace/langchain-learn/examples/auth/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login(
    location="main",
    max_concurrent_users=3
)
authentication_status = st.session_state.get('authentication_status')
print(authentication_status)

# name, authentication_status, username = authenticator.login(
#     location="main",
#     max_concurrent_users=3
# )

# if authentication_status:
#     authenticator.logout(button_name='Logout', location='sidebar')
#     st.sidebar.write(f'Welcome *{name}*')
#     st.title('Application Content')
# elif authentication_status is False:
#     st.error('Username/password is incorrect')
# elif authentication_status is None:
#     st.warning('Please enter your username and password')
