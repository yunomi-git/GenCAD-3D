from . import ProcessUrl as RequestUrlCreator
from . import OnshapeAPI
import json


class UploadBlobElement(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(UploadBlobElement, self).__init__(OnshapeAPI.ApiMethod.POST)
        self.request_url = RequestUrlCreator.get_api_url("v6/blobelements",
                                                         None,
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         wvm=url.wvm)
        self.raw_url = url
        self.file = None
        self.encodedFilename = None

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': "application/vnd.onshape.v1+json;charset=UTF-8;qs=0.1",
                'Content-Type': 'multipart/form-data'}

    def upload_new_file(self):
        # This does not make usage apparent. refactor
        self.request_url = RequestUrlCreator.get_api_url("v6/blobelements",
                                                         None,
                                                         document=self.raw_url.documentID,
                                                         workspace=self.raw_url.workspaceID,
                                                         wvm=self.raw_url.wvm)
        payload = {
            "file": self.file,
            "encodedFilename": self.encodedFilename,
            "notifyUser": True,
            "locationPosition": -1,
            "storeInDocument": True,
            "translate": False,
            "importAppearances": False,
            "splitAssembliesIntoMultipleDocuments": False,
            "onePartPerDoc": False,
        }
        self.make_request(payload=payload, use_post_param=True)

    def update_file(self):
        # update targets an element id
        # This does not make usage apparent. refactor
        self.request_url = RequestUrlCreator.get_api_url("v6/blobelements",
                                                         None,
                                                         document=self.raw_url.documentID,
                                                         workspace=self.raw_url.workspaceID,
                                                         element=self.raw_url.elementID,
                                                         wvm=self.raw_url.wvm)
        payload = {
            "file": self.file,
            "encodedFilename": self.encodedFilename,
            "notifyUser": True,
            "locationPosition": -1,
            "storeInDocument": True,
            "translate": False,
            "importAppearances": False,
            "splitAssembliesIntoMultipleDocuments": False,
            "onePartPerDoc": False,
        }
        self.make_request(payload=payload, use_post_param=True)


if __name__ == "__main__":
    json = json.dumps({"adaa123": 1})
    url = RequestUrlCreator.OnshapeUrl("https://cad.onshape.com/documents/c3b4576ef97b70b3e09ba2f0/w/75bec76c270d0cb4899d9ce4/e/1e160786c96002332ab0abbf")
    uploadBlobElement = UploadBlobElement(url)
    uploadBlobElement.file = json
    uploadBlobElement.encodedFilename = "aaa.jsn"
    uploadBlobElement.update_file()
