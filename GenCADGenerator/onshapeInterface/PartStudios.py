from . import ProcessUrl as RequestUrlCreator
from . import OnshapeAPI as OnshapeAPI
import json


class GetStl(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(GetStl, self).__init__(OnshapeAPI.ApiMethod.GET)
        self.stl = None
        self.request_url = RequestUrlCreator.get_api_url("partstudios",
                                                    "stl",
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         element=url.elementID,
                                                         wvm=url.wvm)

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': 'application/vnd.onshape.v1+octet-stream',
                'Content-Type': 'application/json'}

    def send_request(self):
        payload = {
            "units": "millimeter",
            "grouping": "true",
            "scale": 1
        }
        self.stl = self.make_request(payload=payload, use_post_param=False)

    def get_response(self, filename):
        with open(filename, 'wb') as f:
            f.write(self.stl.data.encode())


class GetFeatureList(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(GetFeatureList, self).__init__(OnshapeAPI.ApiMethod.GET)
        self.request_url = RequestUrlCreator.get_api_url("partstudios",
                                                         "features",
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         element=url.elementID,
                                                         wvm=url.wvm)
        self.microversion_id = None

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': 'application/json;charset=UTF-8; qs=0.09',
                'Content-Type': 'application/json'}

    def send_request(self):
        payload = {
            "rollbackBarIndex": -1,
            "includeGeometryIds": True,
            # "featureId": [], # There are issues with sending this array
            "noSketchGeometry": False
        }
        response = self.make_request(payload=payload, use_post_param=False)
        response = json.loads(response.data)
        self.microversion_id = response["sourceMicroversion"]
        return response

class AddFeature(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(AddFeature, self).__init__(OnshapeAPI.ApiMethod.POST)
        self.request_url = RequestUrlCreator.get_api_url("partstudios",
                                                         "features",
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         element=url.elementID,
                                                         wvm=url.wvm)
        self.json_feature = None
        self.source_microversion = None

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': 'application/json;charset=UTF-8; qs=0.09',
                'Content-Type': 'application/json'}

    def send_request(self):
        payload = {
            "sourceMicroversion": self.source_microversion,
            "feature": self.json_feature,
            "rejectMicroversionSkew": False
        }
        response = self.make_request(payload=payload, use_post_param=False)
        return response.data

class EvaluateFeaturescipt(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(EvaluateFeaturescipt, self).__init__(OnshapeAPI.ApiMethod.POST)
        self.request_url = RequestUrlCreator.get_api_url("partstudios",
                                                         "featurescript",
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         element=url.elementID,
                                                         wvm=url.wvm)
        self.script = None
        self.queries = None
        self.source_microversion = None

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': 'application/json;charset=UTF-8; qs=0.09',
                'Content-Type': 'application/json'}

    def send_request(self):
        payload = {
            "sourceMicroversion": self.source_microversion,
            "script": self.script,
            "queries" : self.queries,
            "rejectMicroversionSkew": False
        }
        response = self.make_request(payload=payload, use_post_param=False)
        return json.loads(response.data)

    def get_query_result(self, response):
        values = response["result"]["message"]["value"]
        query_names = [value["message"]["value"][1]["message"]["value"]["message"]["value"] for value in values]
        return query_names

# class FeaturescriptCreator:
    def set_query_sketch_construction(self):
        # query_type: body_type, entity_type, geometry_type
        script = """
        function (context is Context, queries is map)
        {
            var all_query = qSketchFilter(qEverything(EntityType.EDGE), SketchObject.YES);
            all_query = qConstructionFilter(all_query, ConstructionObject.YES);
            var all_encoder = evaluateQuery(context, all_query);
            return all_encoder;
        }
        """
        queries = []
        self.script = script
        self.queries = queries
        # return script, queries

    def set_query_points(self):
        # query_type: body_type, entity_type, geometry_type
        script = """
        function (context is Context, queries is map)
        {
            var all_query = qEverything(EntityType.VERTEX);
            all_query = qBodyType(all_query, BodyType.POINT);
            var all_encoder = evaluateQuery(context, all_query);
            return all_encoder;
        }
        """
        queries = []
        self.script = script
        self.queries = queries

    def set_query_mate_connectors(self):
        # query_type: body_type, entity_type, geometry_type
        script = """
        function (context is Context, queries is map)
        {
            var all_query = qEverything(EntityType.VERTEX);
            all_query = qBodyType(all_query, BodyType.MATE_CONNECTOR);
            var all_encoder = evaluateQuery(context, all_query);
            return all_encoder;
        }
        """
        queries = []
        self.script = script
        self.queries = queries

    def set_query_sketch_faces(self):
        script = """
        function (context is Context, queries is map)
        {
            var all_query = qSketchFilter(qEverything(EntityType.FACE), SketchObject.YES);
            var all_encoder = evaluateQuery(context, all_query);
            return all_encoder;
        }
        """
        queries = []
        self.script = script
        self.queries = queries
        # return script, queries

    def set_query_attribute(self, attribute_name : str):
        script = """
        function (context is Context, queries is map)
        {{
            var instantiatedBodies = qHasAttribute(qEverything(EntityType.BODY), "{attributeName}");

            // This is what we are looking for
            var outputKinematics = getAttribute(context, {{"entity" : instantiatedBodies, "name" : "{attributeName}"}});
            return outputKinematics;
        }}
        """.format(attributeName=attribute_name)
        queries = []
        self.script = script
        self.queries = queries
        # return script, queries

class NewPartStudio(OnshapeAPI.OnshapeAPI):
    def __init__(self, url: RequestUrlCreator.OnshapeUrl):
        super(NewPartStudio, self).__init__(OnshapeAPI.ApiMethod.POST)
        self.request_url = RequestUrlCreator.get_api_url("partstudios",
                                                         "",
                                                         document=url.documentID,
                                                         workspace=url.workspaceID,
                                                         wvm=url.wvm)
        self.name = None
        self.created_element_id = None
        self.microversion = None

    def _get_api_url(self):
        return self.request_url

    def _get_headers(self):
        return {'Accept': 'application/json;charset=UTF-8; qs=0.09',
                'Content-Type': 'application/json'}

    def send_request(self):
        payload = {
            "name": self.name
        }
        response = self.make_request(payload=payload, use_post_param=False)
        response = json.loads(response.data)
        self.created_element_id = response["id"]
        self.microversion = response["microversionId"]
        return response

if __name__ == "__main__":
    # url = RequestUrlCreator.OnshapeUrl("https://cad.onshape.com/documents/c3b4576ef97b70b3e09ba2f0/w/75bec76c270d0cb4899d9ce4/e/2a5362fe0e6cb33b327a98de")
    # getStl = GetStl(url)
    # getStl.send_request()
    # getStl.get_response("got.stl")


    url = RequestUrlCreator.OnshapeUrl("https://cad.onshape.com/documents/c3b4576ef97b70b3e09ba2f0/w/75bec76c270d0cb4899d9ce4/e/8fbf440a3480ee969c03c26f")
    # getFeatures = GetFeatureList(url)
    # response = getFeatures.send_request()
    # print(response)

    with open("sketch.json", 'r') as f:
        s = f.read()
    feature = json.loads(s)

    addFeature = AddFeature(url)
    addFeature.json_feature = feature
    addFeature.source_microversion =  "ddffa8be62383c06c91cfe6f"
    addFeature.send_request()