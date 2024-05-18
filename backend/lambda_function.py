import json, base64
from predict import predict, test

def lambda_handler(event, context):
    responseHeaders = {
      "Access-Control-Allow-Origin": "*"
    }
    # TODO implement
    if event.get("httpMethod") == "POST":
        # print(event["body"])
        img_data = event["body"].split(",")[1]
        img = base64.b64decode(img_data)
        with open("/tmp/img.jpg", "wb") as file:
            file.write(img)
        ans, probs = predict()
    else:        
        ans, probs = test()
    # print(ans)
    # print(probs[0].tolist())
    
    return {
        "isBase64Encoded": False,
        "statusCode": 200,
        "headers": responseHeaders,
        "body": json.dumps({"message": "Hello from Lambda!", "ans": str(ans), "raw": probs[0].tolist()})
    }
