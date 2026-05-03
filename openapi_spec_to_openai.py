import json
import requests


def openapi_spec_to_tools(spec_text: str):
    spec = json.loads(spec_text)

    base_url = spec["servers"][0]["url"]
    tools = []
    callables = {}

    for path, path_config in spec["paths"].items():
        for method, operation in path_config.items():
            operation_id = operation["operationId"]
            summary = operation.get("summary", operation_id)
            parameters = operation.get("parameters", [])

            properties = {}
            required = []

            for param in parameters:
                name = param["name"]
                schema = param.get("schema", {})
                param_type = schema.get("type", "string")

                properties[name] = {
                    "type": param_type,
                    "description": param.get("description", "")
                }

                if param.get("required", False):
                    required.append(name)

            tool = {
                "type": "function",
                "function": {
                    "name": operation_id,
                    "description": summary,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            tools.append(tool)

            def make_callable(method, path, parameters):
                def api_callable(**kwargs):
                    url = base_url + path

                    query_params = {}

                    for param in parameters:
                        name = param["name"]
                        location = param["in"]

                        if location == "path":
                            url = url.replace("{" + name + "}", str(kwargs[name]))
                        elif location == "query" and name in kwargs:
                            query_params[name] = kwargs[name]

                    response = requests.request(
                        method.upper(),
                        url,
                        params=query_params,
                    )

                    response.raise_for_status()

                    if response.text:
                        return response.json()

                    return {}

                return api_callable

            callables[operation_id] = make_callable(method, path, parameters)

    return tools, callables