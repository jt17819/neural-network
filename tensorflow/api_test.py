import requests, base64

URL = "https://tensorflow-neural-network.onrender.com"

def hello_world():
    response = requests.get(f"{URL}/")
    print(response.status_code)
    print(response.text)
    return


def predict(payload):
    response = requests.post(f"{URL}/predict", json={"payload": payload})
    print(response.status_code)
    print(response.text)
    return 

# test predict
with open("tensorflow/digit3-1.png", "rb") as f:
    img_bytes = f.read()
    payload = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
    result = predict(payload)
    print(result)
    