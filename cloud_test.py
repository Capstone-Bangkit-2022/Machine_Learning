import requests

response = requests.post('https://getpredictions-3plyzvalqq-uc.a.run.app', files={'file': open('sampel1.jpg', 'rb')})
# response1 = requests.post('https://getpredictions-3plyzvalqq-uc.a.run.app', files={'file': open('sampel2.jpg', 'rb')})
# response2 = requests.post('https://getpredictions-3plyzvalqq-uc.a.run.app', files={'file': open('sampel3.jpg', 'rb')})
# response3 = requests.post('https://getpredictions-3plyzvalqq-uc.a.run.app', files={'file': open('sampel5.jpg', 'rb')})

print(response.json())
# print(response1.json())
# print(response2.json())
# print(response3.json())