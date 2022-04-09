import requests
import json
from docarray import Document

doc = Document(uri='https://picsum.photos/id/222/300/300.jpg')
doc.load_uri_to_blob()
blob = doc.blob
doc.convert_blob_to_tensor()
reqUrl = "http://0.0.0.0:63745/post"


headersList = {
#  "Accept": "application/json",
 "Content-Type": "application/json" 
#  'Content-Type': 'image/jpeg'
}
null = None
payload = json.dumps({
    "execEndpoint": "/foo",
    "data": [
        {
            "id": "2f0fae7fe7e2b34cd1106bf7b2ef5821", 
            "parent_id": null, 
            "granularity": null, 
            "adjacency": null, 
            "blob": null, 
            "tensor": doc.tensor, 
            "mime_type": null, 
            "text": "hello", 
            "weight": null, 
            "uri": 'https://picsum.photos/id/222/300/300.jpg', 
            "tags": null, 
            "offset": null, 
            "location": null, 
            "embedding": null, 
            "modality": null, 
            "evaluations": null, 
            "scores": null, 
            "chunks": null, 
            "matches": null,
        }, 
        {
            
        }
    ],
    
})

response = requests.request("POST", reqUrl, data=payload,  headers=headersList)

print(response.text)

















# import requests
# import json
# reqUrl = "http://0.0.0.0:63745/post"


# headersList = {
# #  "Accept": "application/json",
#  "Content-Type": "application/json" 
# #  'Content-Type': 'image/jpeg'
# }
# null = None
# payload = json.dumps({
#     "execEndpoint": "/foo",
#     "data": [
#         {
#             "id": "2f0fae7fe7e2b34cd1106bf7b2ef5821", 
#             "parent_id": null, 
#             "granularity": null, 
#             "adjacency": null, 
#             "blob": null, 
#             "tensor": null, 
#             "mime_type": null, 
#             "text": "hello", 
#             "weight": null, 
#             "uri": null, 
#             "tags": null, 
#             "offset": null, 
#             "location": null, 
#             "embedding": null, 
#             "modality": null, 
#             "evaluations": null, 
#             "scores": null, 
#             "chunks": null, 
#             "matches": null,
#         }, 
#         {
#             "text": "world"
#         }
#     ],
    
# })

# response = requests.request("POST", reqUrl, data=payload,  headers=headersList)

# print(response.text)