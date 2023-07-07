Create a Brev environment with the following link:
https://console.brev.dev/environment/new?repo=https://github.com/brevdev/falcon-api&setupPath=setup.sh&instance=a2-highgpu-1g

Start server:
```
python app.py
```
Curl command:

```
curl -X POST -H "Content-Type: application/json" -d '{"text":"Once upon a time"}' http://localhost:8888/predict
```
