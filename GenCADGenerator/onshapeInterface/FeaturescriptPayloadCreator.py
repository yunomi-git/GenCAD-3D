# Class creates the payload to send via the Featurescript API endpoint

class FeaturescriptCreator:
    def get_sketch_construction(self):
        # query_type: body_type, entity_type, geometry_type
        script = """
        function (context is Context, queries is map)
        {{
            var all_query = qSketchFilter(qEverything(EntityType.EDGE), SketchObject.YES);
            all_query = qConstructionFilter(all_query, ConstructionObject.YES);
            var all_encoder = evaluateQuery(context, all_query);
            return all_encoder;
        }}
        """
        queries = []
        return script, queries

    def get_attribute(attribute_name : str):
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
        return script, queries
