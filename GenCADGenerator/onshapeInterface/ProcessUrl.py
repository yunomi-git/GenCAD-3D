# This creates the URL request to send to the API, excluding the payload
# Use setRequest() and setCategory() based on the Glassworkd API (google)

# Parses a url copy pasted from browser
class OnshapeUrl:
    def __init__(self, url):
        self.original_url = url
        self.elementID = None
        self.workspaceID = None
        self.documentID = None
        self.wvm = None
        self.parse_studio_url(url)

    def parse_studio_url(self, url):
        index = url.find("document")
        index += 1
        closing_index = url.find("/", index)
        index = closing_index + 1
        closing_index = url.find("/", index)
        self.documentID = url[index: closing_index]

        index = url.find("w/")
        self.wvm = "w"
        if index == -1:
            index = url.find("v/")
            self.wvm = "v"
        if index == -1:
            index = url.find("m/")
            self.wvm = "m"
        index += 1
        closing_index = url.find("/", index)
        index = closing_index + 1
        closing_index = url.find("/", index)
        self.workspaceID = url[index:closing_index]

        index = url.find("e/")
        if index != -1:
            index += 1
            closing_index = url.find("/", index)
            index = closing_index + 1
            self.elementID = url[index:]


# Generates the url related to the api
def get_api_url(category : str, request, document: str, workspace: str, wvm: str, element: str = None):
    s = "https://cad.onshape.com/api/"
    s += category
    s += "/d/" + document
    s += "/" + wvm + "/"
    s += workspace
    if element is not None:
        s += "/e/" + element
    if request is not None:
        s += "/" + request
    return s

