import datetime

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document

urls = [
    "https://www.youtube.com/watch?v=dA1CHGACXCo",
    "https://www.youtube.com/watch?v=ZCEMLz27SL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhLPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibosL1ps",
    "https://www.youtube.com/watch?v=281C4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-Uc",
    "https://www.youtube.com/watch?v=WLRHWKUKVOE",
    "https://www.youtube.com/watch?v=0bILtMaRJVY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M"
]

docs: [Document] = [YoutubeLoader.from_youtube_url(url, add_video_info=True).load() for url in urls]

for doc in docs:
    doc.metadata['publish_year'] = int(
        datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y')
    )
    print(doc)
