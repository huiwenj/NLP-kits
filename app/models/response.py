from fastapi.responses import JSONResponse


class R:
    @staticmethod
    def success(data=None):
        return JSONResponse(status_code=200, content={"msg": "success", "data": data})

    @staticmethod
    def bad_request(message=None):
        return JSONResponse(status_code=400, content={"msg": message if not None else "bad request"})

    @staticmethod
    def error(message=None):
        return JSONResponse(status_code=500, content={"msg": message if not None else "sever error", })

    @staticmethod
    def unauthorized():
        return JSONResponse(status_code=401, content={"msg": "unauthorized"})

    @staticmethod
    def forbidden():
        return JSONResponse(status_code=403, content={"msg": "forbidden"})

    @staticmethod
    def response(status_code, message, data):
        return JSONResponse(status_code=status_code, content={"msg": message, "data": data})
