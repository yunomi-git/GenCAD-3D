import json
from onshape_client.client import Client
from .ConfigurationEncoder import ConfigurationEncoder
from abc import abstractmethod, ABC
from enum import Enum

with open("onshape_keys.json", "r") as f:
    keys = json.load(f)

base = 'https://cad.onshape.com'
client = Client(configuration={"base_url": base,
                               "access_key": keys["access_key"],
                               "secret_key": keys["secret_key"]})

class ApiMethod(Enum):
    POST = "POST"
    GET = "GET"


class OnshapeAPI(ABC):
    def __init__(self, method : ApiMethod):
        # base = 'https://cad.onshape.com'
        self.client = Client.get_client()
        self.method = method.value

    @abstractmethod
    def _get_api_url(self):
        pass

    @abstractmethod
    def _get_headers(self):
        pass

    # inputs is np array, unitsList is string array
    # returns parsed API request, or None if error occurred
    def make_request(self, configuration: ConfigurationEncoder = None, payload = None, use_post_param: bool = False):
        # Payload can be dict or none
        # Configuration of the request
        params = {}
        if configuration is not None:
            config = configuration.get_encoding()
            params = {'configuration': config}

        # Send the request to onshape
        # multipart post needs to pass post_params
        # normal post passes body
        if self.method == ApiMethod.POST.name:
            if use_post_param:
                response = self.client.api_client.request(method=self.method,  # specific
                                                          url=self._get_api_url(),  # general-specific
                                                          query_params=params,  # general
                                                          headers=self._get_headers(),  # general
                                                          post_params=payload) # specific
            else:
                response = self.client.api_client.request(method=self.method,  # specific
                                                          url=self._get_api_url(),  # general-specific
                                                          query_params=params,  # general
                                                          headers=self._get_headers(),  # general
                                                          body=payload)  # specific
        else: # GET
            response = self.client.api_client.request(method=self.method,  # specific
                                                      url=self._get_api_url(),  # general-specific
                                                      query_params=payload,  # general
                                                      headers=self._get_headers())  # general
        return response





