from examples.agent.postgres_saver_factory import create_checkpointer

checkpointer = create_checkpointer()

resp = checkpointer.list(
    {"configurable": {
        "thread_id": "6191116288"
    }}
)
for item in resp:
    print(item)
