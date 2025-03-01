import os

import requests
from requests_auth import OAuth2ResourceOwnerPasswordCredentials


def create_oauth2() -> OAuth2ResourceOwnerPasswordCredentials:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": os.getenv("OAUTH2_BASIC_AUTH_HEADER"),
        }
    )
    return OAuth2ResourceOwnerPasswordCredentials(
        token_url=os.getenv("OAUTH2_ENDPOINT"),
        username=os.getenv("OAUTH2_USERNAME"),
        password=os.getenv("OAUTH2_PASSWORD"),
        header_name="Authorization",
        header_value="Oauth2 {token}",
        scope="read",
        session=session
    )

resp = requests.post(
    url='https://openapi.pistonint.com/evaluate',
    headers={
        "Content-Type": "application/json"
    },
    json={
        "datas": [
            {
                "cityId": "cit00790",
                "mileage": "3.70",
                "trimId": "tri64271",
                "regTime": "20210401",
                "colorId": "Col09",
            }
        ]
    },
    auth=create_oauth2()
)

print(resp.json())
