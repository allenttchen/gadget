import re


def toLowerCase(text):
    return text.lower()


def replaceEmailToken(text, email_token="EMAIL_TOKEN"):
    return re.sub(
        "([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", 
        email_token, 
        text, 
    )
