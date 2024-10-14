def pretty_print_messages(messages):
    for message in messages:
        # Check if 'content' is not None and 'sender' is in the message
        if message.get("content") is not None and "sender" in message:
            return f"{message['sender']}: {message['content']}"
